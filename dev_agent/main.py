"""Unified CLI: headless and chat modes for the CLAUDE.md workflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from .config import AgentConfig
from .tools import MCPClient, ToolHandler, get_tool_definitions

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dev_agent")

MAX_ITERATIONS = 20

SYSTEM_PROMPT = """You are a TDD (Test-Drive Development) workflow orchestrator.

### Agents
* **claude_code**: Implements solutions and tests. Summarizes work in `worklog.md`.
* **codex**: Reviews code for P0/P1 issues. Records findings in `worklog.md` and `codex_review.log`.

### Workflow
1.  **Implement (claude_code)**: Implement the solution and matching tests for the user's task.
2.  **Review (codex)**: Review the implementation for P0/P1 issues.
3.  **Fix (claude_code)**: If issues are found, fix all P0/P1 issues and ensure tests pass.
4.  Repeat **Review** and **Fix** until `codex` reports no P0/P1 issues.

### Your Orchestration Rules
1.  **Call Agents**: For each workflow step, call `execute_agent` with `num_branches=1`. After the call, use `check_status` once to monitor completion.
2.  **Maintain State**: Track branch lineage (`parent_branch_id`) and report any tool errors immediately.
3.  **Handle Review Data**: Before launching a **Fix** run, you **must** use `read_artifact` to get the issues from `codex_review.log`.

### Agent Prompting Requirements
When you call `execute_agent`, your prompt to the agent **must** include:

1.  **The Original User Task**: You **must** state the user's original task *exactly* as given (必须原样转述).
2.  **The Specific Goal**: Clearly state the goal for *this* phase (e.g., "Implement and test the solution," "Review the code for P0/P1 issues," or "Fix the issues found in the review").
3.  **Essential Context**: Provide `workspace_dir`, `parent_branch_id`, and any artifacts to read (like `worklog.md` or `codex_review.log`).
4.  **Focus on "What", Not "How"**: Clearly state the goal and requirements. **Do not** provide detailed step-by-step implementation or review instructions. Let the agent determine the best *how* to achieve the goal.

### Completion
* **Stop Condition**: Stop when a `codex` **Review** run reports no P0/P1 issues.
* **Final Output**: Reply with **JSON only** (no other text):
    {{
      "type": "final_report",
      "task": "<original user task description>",
      "summary": "<Concise outcome, e.g., 'Implementation and review complete. No P0/P1 issues found.'>"
    }}
"""


class LLMBrain:
    """Thin wrapper around the Azure OpenAI Chat Completions API with retry logic."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str,
        max_retries: int = 3,
    ) -> None:
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = deployment
        self._max_retries = max_retries

    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Send messages to OpenAI and return the raw response."""

        last_exception: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self._deployment,
                    "messages": messages,
                    "max_completion_tokens": 4000,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = self._client.chat.completions.create(**kwargs)
                return response
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if attempt < self._max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Azure OpenAI call failed (attempt %s/%s): %s. Retrying in %ss...",
                        attempt + 1,
                        self._max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Azure OpenAI call failed after retries: %s", exc)

        raise last_exception or RuntimeError("Unknown Azure OpenAI API error")


def _print_assistant_message(message: Any) -> None:
    content = getattr(message, "content", None)
    if content:
        print(f"assistant> {content}")


def build_initial_messages(task: str, cfg: AgentConfig, parent_branch_id: str) -> List[Dict[str, Any]]:
    """Create the system and user messages that kick off the orchestrator loop."""
    user_payload = {
        "task": task,
        "parent_branch_id": parent_branch_id,
        "project_name": cfg.project_name,
        "workspace_dir": cfg.workspace_dir,
        "notes": (
            "For every phase: craft an execute_agent prompt "
            "covering task, phase goal, context, and expectations, run with num_branches=1, then call "
            "check_status once. Track branch lineage and stop when codex reports no P0/P1 issues."
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2)},
    ]


def assistant_message_to_dict(message: Any) -> Dict[str, Any]:
    """Convert an OpenAI assistant message to a dict suitable for the next request."""
    content = message.content or ""
    payload: Dict[str, Any] = {"role": "assistant", "content": content}
    if getattr(message, "tool_calls", None):
        payload["tool_calls"] = []
        for call in message.tool_calls:
            payload["tool_calls"].append(
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
    return payload


def parse_final_report(message: Any) -> Optional[Dict[str, Any]]:
    """Try to parse the assistant content as the final report JSON."""
    content = message.content
    if not content:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "final_report":
        return None
    return payload


def orchestrate(
    brain: LLMBrain,
    handler: ToolHandler,
    messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Run the main LLM loop until completion or iteration limit."""
    tools = get_tool_definitions()

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info("LLM iteration %s", iteration)
        response = brain.complete(messages, tools=tools)
        choice = response.choices[0].message
        messages.append(assistant_message_to_dict(choice))

        tool_calls = getattr(choice, "tool_calls", None) or []
        if tool_calls:
            for tool_call in tool_calls:
                result = handler.handle(tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
            continue

        final_report = parse_final_report(choice)
        if final_report:
            return final_report
        logger.info("Assistant response was not a final report; continuing.")

    logger.error("Reached maximum iterations without final report.")
    return None


def chat_loop(
    brain: LLMBrain,
    handler: ToolHandler,
    messages: List[Dict[str, Any]],
    *,
    max_iterations: int = MAX_ITERATIONS,
) -> Optional[Dict[str, Any]]:
    tools = get_tool_definitions()

    for iteration in range(1, max_iterations + 1):
        print(f"[iter {iteration}] requesting completion...")
        response = brain.complete(messages, tools=tools)
        choice = response.choices[0].message
        _print_assistant_message(choice)
        messages.append(
            {
                "role": "assistant",
                "content": choice.content or "",
                **(
                    {
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": c.type,
                                "function": {
                                    "name": c.function.name,
                                    "arguments": c.function.arguments,
                                },
                            }
                            for c in (choice.tool_calls or [])
                        ]
                    }
                    if getattr(choice, "tool_calls", None)
                    else {}
                ),
            }
        )

        tool_calls = getattr(choice, "tool_calls", None) or []
        if tool_calls:
            for tool_call in tool_calls:
                fn = tool_call.function
                print(f"tool> {fn.name} {fn.arguments}")
                result = handler.handle(tool_call)
                print(f"tool< {json.dumps(result)[:2000]}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
            continue

        final_report = parse_final_report(choice)
        if final_report:
            print("assistant< final_report")
            return final_report
        print("assistant< not final yet, continuing...")

    print("error: reached iteration limit without final report", file=sys.stderr)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GPT-5 tool-calling orchestrator (headless or chat)")
    parser.add_argument("--task", help="User task description; prompts if omitted")
    parser.add_argument("--parent-branch-id", required=True, help="Parent branch UUID")
    parser.add_argument("--project-name", help="Optional project name override")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no chat prints)")
    args = parser.parse_args(argv)

    try:
        cfg = AgentConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        logger.error("Configuration error: %s", exc)
        return 1

    if args.project_name:
        cfg = replace(cfg, project_name=args.project_name)

    if not cfg.project_name:
        logger.error("Project name must be provided via PROJECT_NAME or --project-name")
        return 1

    # Single CLI: default to chat; enable headless with flag
    headless = bool(args.headless)

    task = args.task or input("you> Enter task description: ")
    if not task.strip():
        print("error: task is required", file=sys.stderr)
        return 1

    brain = LLMBrain(
        cfg.azure_api_key,
        cfg.azure_endpoint,
        cfg.azure_deployment,
        cfg.azure_api_version,
    )
    mcp_client = MCPClient(cfg.mcp_base_url)
    handler = ToolHandler(mcp_client, cfg.project_name)

    messages = build_initial_messages(task, cfg, args.parent_branch_id)
    final_report = orchestrate(brain, handler, messages) if headless else chat_loop(brain, handler, messages)

    if not final_report:
        return 1

    final_report.setdefault("task", task)
    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
