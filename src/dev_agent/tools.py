"""Tool definitions and execution helpers for interacting with MCP."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests


class MCPError(RuntimeError):
    """Raised when an MCP call fails."""


class MCPClient:
    """Lightweight JSON-RPC client for MCP tools with retry logic."""

    def __init__(
        self,
        base_url: str,
        *,
        session: Optional[Any] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._session = session or requests.Session()
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._request_id = 0

    def _call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}

        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                response = self._session.post(self._base_url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                body = response.json()
                if "error" in body:
                    raise MCPError(body["error"])
                return body["result"]
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if attempt < self._max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "MCP call %s failed (attempt %s/%s): %s. Retrying in %ss...",
                        method,
                        attempt + 1,
                        self._max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("MCP call %s failed after retries: %s", method, exc)

        raise last_exception or MCPError("Unknown MCP error")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("tools/call", {"name": name, "arguments": arguments})

    def parallel_explore(
        self,
        project_name: str,
        parent_branch_id: str,
        prompts: List[str],
        agent: str,
        num_branches: int = 1,
    ) -> Dict[str, Any]:
        return self.call_tool(
            "parallel_explore",
            {
                "project_name": project_name,
                "parent_branch_id": parent_branch_id,
                "shared_prompt_sequence": prompts,
                "num_branches": num_branches,
                "agent": agent,
            },
        )

    def get_branch(self, branch_id: str) -> Dict[str, Any]:
        return self.call_tool("get_branch", {"branch_id": branch_id})

    def branch_read_file(self, branch_id: str, file_path: str) -> Dict[str, Any]:
        return self.call_tool("branch_read_file", {"branch_id": branch_id, "file_path": file_path})

logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Raised when a tool call cannot be executed."""


class ToolHandler:
    """Dispatches OpenAI tool calls to MCP client operations."""

    def __init__(
        self,
        client: Any,
        default_project_name: Optional[str],
        *,
        max_branches: int = 4,
    ) -> None:
        self._client = client
        self._default_project_name = default_project_name
        self._max_branches = max_branches

    def handle(self, tool_call: Any) -> Dict[str, Any]:
        """Execute a tool call from OpenAI and return serializable result."""

        name = getattr(getattr(tool_call, "function", None), "name", None)
        arguments_payload = getattr(getattr(tool_call, "function", None), "arguments", "{}")

        if not name:
            return self._error("Missing tool name in call.")

        try:
            arguments = json.loads(arguments_payload or "{}")
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON arguments for tool %s: %s", name, arguments_payload)
            return self._error(f"Invalid JSON arguments: {exc}")

        try:
            if name == "execute_agent":
                result = self._execute_agent(arguments)
            elif name == "check_status":
                result = self._check_status(arguments)
            elif name == "read_artifact":
                result = self._read_artifact(arguments)
            else:
                raise ToolExecutionError(f"Unsupported tool: {name}")
        except ToolExecutionError as exc:
            logger.error("Tool %s failed precondition: %s", name, exc)
            return self._error(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised unexpected error", name)
            return self._error(f"Execution error: {exc}")

        return {"status": "success", "data": result}

    def _execute_agent(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        agent = arguments.get("agent")
        prompt = arguments.get("prompt")
        project_name = arguments.get("project_name") or self._default_project_name
        parent_branch_id = arguments.get("parent_branch_id")
        num_branches = arguments.get("num_branches", 1)

        if not agent or not isinstance(agent, str):
            raise ToolExecutionError("`agent` string argument is required.")
        if not prompt or not isinstance(prompt, str):
            raise ToolExecutionError("`prompt` string argument is required.")
        if not parent_branch_id or not isinstance(parent_branch_id, str):
            raise ToolExecutionError("`parent_branch_id` string argument is required.")
        if not project_name or not isinstance(project_name, str):
            raise ToolExecutionError("`project_name` string argument is required or set via config.")
        if not isinstance(num_branches, int) or num_branches < 1 or num_branches > self._max_branches:
            raise ToolExecutionError(
                f"`num_branches` must be an integer between 1 and {self._max_branches}."
            )

        logger.info(
            "Executing agent %s on project %s from parent %s", agent, project_name, parent_branch_id
        )
        response = self._client.parallel_explore(
            project_name=project_name,
            parent_branch_id=parent_branch_id,
            prompts=[prompt],
            agent=agent,
            num_branches=num_branches,
        )
        branches = response.get("branches") if isinstance(response, dict) else None
        primary_branch = branches[0] if isinstance(branches, list) and branches else None
        branch_id = primary_branch.get("branch_id") if isinstance(primary_branch, dict) else None
        return {
            "parallel_explore": response,
            "branch_id": branch_id,
        }

    def _check_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        branch_id = arguments.get("branch_id")
        if not branch_id or not isinstance(branch_id, str):
            raise ToolExecutionError("`branch_id` string argument is required.")
        logger.info("Checking status for branch %s", branch_id)
        return self._client.get_branch(branch_id)

    def _read_artifact(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        branch_id = arguments.get("branch_id")
        path = arguments.get("path")
        if not branch_id or not isinstance(branch_id, str):
            raise ToolExecutionError("`branch_id` string argument is required.")
        if not path or not isinstance(path, str):
            raise ToolExecutionError("`path` string argument is required.")
        logger.info("Reading artifact %s from branch %s", path, branch_id)
        return self._client.branch_read_file(branch_id, path)

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {"status": "error", "error": message}


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool definitions that surface MCP operations."""

    return [
        {
            "type": "function",
            "function": {
                "name": "execute_agent",
                "description": (
                    "Launch an MCP parallel_explore job for a specialist agent. "
                    "Provide the target agent (claude_code or codex), prompt, "
                    "parent branch id, and project name. Optionally control the "
                    "number of branches to spawn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "description": "Target specialist agent name, e.g. claude_code or codex.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Prompt that describes the task for the agent.",
                        },
                        "project_name": {
                            "type": "string",
                            "description": "Pantheon project name.",
                        },
                        "parent_branch_id": {
                            "type": "string",
                            "description": "Branch UUID to branch from for this run.",
                        },
                        "num_branches": {
                            "type": "integer",
                            "description": "Optional number of sibling branches to create.",
                            "default": 1,
                            "minimum": 1,
                            "maximum": 4,
                        },
                    },
                    "required": ["agent", "prompt", "project_name", "parent_branch_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_status",
                "description": (
                    "Fetch status information for an MCP branch id. Useful for polling "
                    "until a branch run finishes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "branch_id": {
                            "type": "string",
                            "description": "Branch UUID returned from execute_agent.",
                        }
                    },
                    "required": ["branch_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_artifact",
                "description": (
                    "Read a text artifact produced by a branch. "
                    "Pass a branch id and the artifact path or filename."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "branch_id": {
                            "type": "string",
                            "description": "Branch that produced the artifact.",
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Artifact path or filename, e.g. worklog.md or "
                                "artifacts/worklog.md."
                            ),
                        },
                    },
                    "required": ["branch_id", "path"],
                },
            },
        },
    ]
