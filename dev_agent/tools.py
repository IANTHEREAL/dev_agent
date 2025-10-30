"""Tool definitions and execution helpers for interacting with MCP."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import requests


class MCPError(RuntimeError):
    """Raised when an MCP call fails."""


class MCPClient:
    """Lightweight MCP client posting JSON-RPC to a single endpoint.

    Expects base_url to be the Streamable HTTP endpoint (e.g., http://localhost:8000/mcp/sse).
    Optional GET SSE stream is not used by this client.
    """

    def __init__(
        self,
        base_url: str,
        *,
        session: Optional[Any] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._session = session or requests.Session()
        base = (base_url or "http://localhost:8000/mcp/sse").rstrip("/")
        self._rpc_url = base
        self._timeout = timeout
        self._max_retries = max_retries
        self._request_id = 0
        self._session_id = str(uuid.uuid4())

    @staticmethod
    def _parse_sse_json(text: str) -> Dict[str, Any]:
        """Return the first JSON object from an SSE payload or raise."""
        normalized = text.replace("\r\n", "\n")

        # Parse SSE events into dictionaries of {event, data: [chunks]}
        events: List[Dict[str, Any]] = []
        current_event: Dict[str, Any] = {}

        for raw_line in normalized.split("\n"):
            line = raw_line.rstrip("\r")

            if line == "":
                if current_event:
                    events.append(current_event)
                    current_event = {}
                continue

            if line.startswith(":"):
                # SSE comment line
                continue

            if ":" not in line:
                # Not a standard field, skip
                continue

            field, value = line.split(":", 1)
            field = field.strip()
            value = value.lstrip(" ")

            if field == "event":
                current_event["event"] = value
            elif field == "data":
                current_event.setdefault("data", []).append(value)

        if current_event:
            events.append(current_event)

        # Try parsing JSON from each data chunk / combined data payload
        for event in events:
            data_lines = event.get("data") or []

            for chunk in data_lines:
                candidate = chunk.strip()
                if not candidate or candidate in {"[DONE]", "DONE"}:
                    continue
                if candidate[0] in "{[":
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

            joined = "\n".join(data_lines).strip()
            if joined and joined[0] in "{[":
                try:
                    return json.loads(joined)
                except json.JSONDecodeError:
                    pass

        # Fallback: scan the raw payload for the first JSON object/array
        decoder = json.JSONDecoder()
        for start_char in ("{", "["):
            search_index = 0
            while True:
                idx = normalized.find(start_char, search_index)
                if idx == -1:
                    break
                try:
                    obj, _ = decoder.raw_decode(normalized[idx:])
                    return obj
                except json.JSONDecodeError:
                    search_index = idx + 1

        logger.warning("Failed to parse SSE JSON. Raw response:\n%s", text)
        raise ValueError("No JSON data event in SSE response")


    def _rpc_post(self, url: str, body: Dict[str, Any], *, timeout: Optional[float] = None) -> requests.Response:
        headers = {
            # Streamable HTTP requires accepting both JSON responses and SSE
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Mcp-Session-Id": self._session_id,
        }
        return self._session.post(url, json=body, headers=headers, timeout=timeout or self._timeout)

    def _call(self, method: str, params: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        self._request_id += 1

        last_exception: Optional[Exception] = None

        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}

        for attempt in range(self._max_retries):
            try:
                # Primary: Streamable HTTP JSON-RPC at /mcp/sse (or provided endpoint)
                logger.debug("MCP POST %s attempt %s to %s", method, attempt + 1, self._rpc_url)
                primary_resp = self._rpc_post(self._rpc_url, payload, timeout=timeout)
                try:
                    primary_resp.raise_for_status()
                    ct = primary_resp.headers.get("Content-Type", "")
                    logger.debug("MCP response %s CT=%s", primary_resp.status_code, ct)
                    if "text/event-stream" in ct:
                        sse_preview = primary_resp.text[:1000]
                        logger.debug("MCP SSE preview: %r", sse_preview)
                        try:
                            body = self._parse_sse_json(primary_resp.text)
                        except Exception as parse_exc:  # noqa: BLE001
                            logger.error(
                                "Failed to parse SSE JSON for %s. Content-Type: %s, Status: %s",
                                method,
                                ct,
                                primary_resp.status_code,
                            )
                            logger.error("Full SSE response:\n%s", primary_resp.text)
                            raise parse_exc
                    else:
                        try:
                            body = primary_resp.json()
                        except ValueError as json_exc:
                            preview = primary_resp.text[:1000]
                            logger.error(
                                "MCP response not JSON (status %s, CT=%s). First 1000 bytes: %r",
                                primary_resp.status_code,
                                ct,
                                preview,
                            )
                            raise json_exc
                    if "error" in body:
                        raise MCPError(body["error"])  # type: ignore[unreachable]
                    # Some servers wrap JSON-RPC inside SSE data events
                    if isinstance(body, dict) and "result" in body:
                        result_obj = body["result"]
                        if isinstance(result_obj, dict) and "structuredContent" in result_obj:
                            return result_obj["structuredContent"]
                        return result_obj
                    return body
                except requests.HTTPError as http_exc:
                    ct = primary_resp.headers.get("Content-Type", "")
                    preview = primary_resp.text[:500]
                    logger.error(
                        "MCP HTTP error %s for %s (CT=%s): %r",
                        primary_resp.status_code,
                        method,
                        ct,
                        preview,
                    )
                    raise http_exc
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

    def call_tool(self, name: str, arguments: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._call("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

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
        return self.call_tool("get_branch", {"branch_id": branch_id}, timeout=max(self._timeout, 300.0))

    def branch_read_file(self, branch_id: str, file_path: str) -> Dict[str, Any]:
        return self.call_tool("branch_read_file", {"branch_id": branch_id, "file_path": file_path})

logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Raised when a tool call cannot be executed."""


class BranchTracker:
    """Track the starting and latest branch ids observed during tool execution."""

    def __init__(self, start_branch_id: Optional[str] = None) -> None:
        self._start_branch_id = start_branch_id
        self._latest_branch_id = start_branch_id

    def record(self, branch_id: Optional[str]) -> None:
        if not branch_id or not isinstance(branch_id, str):
            return
        if not self._start_branch_id:
            self._start_branch_id = branch_id
        self._latest_branch_id = branch_id

    @property
    def start_branch_id(self) -> Optional[str]:
        return self._start_branch_id

    @property
    def latest_branch_id(self) -> Optional[str]:
        return self._latest_branch_id

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "start_branch_id": self._start_branch_id,
            "latest_branch_id": self._latest_branch_id,
        }


class ToolHandler:
    """Dispatches OpenAI tool calls to MCP client operations."""

    def __init__(
        self,
        client: Any,
        default_project_name: Optional[str],
        *,
        max_branches: int = 4,
        start_branch_id: Optional[str] = None,
    ) -> None:
        self._client = client
        self._default_project_name = default_project_name
        self._max_branches = max_branches
        self._branch_tracker = BranchTracker(start_branch_id)

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

    @property
    def branch_range(self) -> Dict[str, Optional[str]]:
        """Expose the observed branch range."""
        return self._branch_tracker.as_dict()

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
        if isinstance(response, dict) and ("error" in response or response.get("isError")):
            raise ToolExecutionError(str(response.get("error") or response))
        branches = response.get("branches") if isinstance(response, dict) else None
        primary_branch = branches[0] if isinstance(branches, list) and branches else None
        branch_id = self._extract_branch_id(primary_branch or {})
        if not branch_id:
            raise ToolExecutionError("Missing branch id in parallel_explore response.")
        self._branch_tracker.record(branch_id)
        return {
            "parallel_explore": response,
            "branch_id": branch_id,
        }

    def _check_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        branch_id = arguments.get("branch_id")
        if not branch_id or not isinstance(branch_id, str):
            raise ToolExecutionError("`branch_id` string argument is required.")
        timeout_seconds = arguments.get("timeout_seconds", 1800)
        poll_interval = arguments.get("poll_interval_seconds", 3.0)
        max_interval = arguments.get("max_poll_interval_seconds", 30.0)

        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ToolExecutionError("`timeout_seconds` must be a positive number if provided.")
        if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            raise ToolExecutionError("`poll_interval_seconds` must be a positive number if provided.")
        if not isinstance(max_interval, (int, float)) or max_interval < poll_interval:
            raise ToolExecutionError("`max_poll_interval_seconds` must be a number >= poll_interval_seconds.")

        def _status_lower(value: Optional[str]) -> str:
            return (value or "").strip().lower()

        attempt = 0
        deadline = time.monotonic() + float(timeout_seconds)
        sleep_interval = float(poll_interval)

        logger.info("Checking status for branch %s (timeout=%ss)", branch_id, timeout_seconds)

        while True:
            attempt += 1
            response = self._client.get_branch(branch_id)
            status = _status_lower(response.get("status"))
            logger.info(
                "Branch %s response (attempt %s): %s",
                branch_id,
                attempt,
                json.dumps(response, ensure_ascii=False),
            )

            if status in {"succeed", "failed"}:
                self._record_branch_from_status(response)
                return response

            if time.monotonic() >= deadline:
                raise ToolExecutionError(
                    f"Timed out waiting for branch {branch_id} to finish. "
                    f"Last status={status or 'unknown'}."
                )

            logger.info(
                "Branch %s still active (status=%s). Sleeping %.1fs.",
                branch_id,
                status or "unknown",
                sleep_interval,
            )
            self._record_branch_from_status(response)
            time.sleep(sleep_interval)
            sleep_interval = min(sleep_interval * 1.5, float(max_interval))

    def _read_artifact(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        branch_id = arguments.get("branch_id")
        path = arguments.get("path")
        if not branch_id or not isinstance(branch_id, str):
            raise ToolExecutionError("`branch_id` string argument is required.")
        if not path or not isinstance(path, str):
            raise ToolExecutionError("`path` string argument is required.")
        logger.info("Reading artifact %s from branch %s", path, branch_id)
        return self._client.branch_read_file(branch_id, path)

    def _record_branch_from_status(self, response: Dict[str, Any]) -> None:
        branch_id = self._extract_branch_id(response)
        if not branch_id:
            raise ToolExecutionError("Branch status response missing branch identifier.")
        self._branch_tracker.record(branch_id)

    @staticmethod
    def _extract_branch_id(response: Any) -> Optional[str]:
        if not isinstance(response, dict):
            return None
        for key in ("branch_id", "id"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return value
        branch_info = response.get("branch")
        if isinstance(branch_info, dict):
            for key in ("branch_id", "id"):
                value = branch_info.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

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
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "description": "Optional upper bound on how long to poll before failing.",
                            "default": 1800,
                        },
                        "poll_interval_seconds": {
                            "type": "number",
                            "description": "Initial delay between polls.",
                            "default": 3,
                        },
                        "max_poll_interval_seconds": {
                            "type": "number",
                            "description": "Upper bound for the exponential backoff poll interval.",
                            "default": 30,
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
