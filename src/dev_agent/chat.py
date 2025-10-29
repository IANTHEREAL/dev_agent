"""Interactive chat-style CLI for the orchestrator.

This CLI runs the same orchestrator logic but prints assistant
messages and tool activity as a conversational stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .main import LLMBrain, build_initial_messages, parse_final_report
from .tools import MCPClient, ToolHandler, get_tool_definitions


logger = logging.getLogger("dev_agent.chat")


def _print_assistant_message(message: Any) -> None:
    content = getattr(message, "content", None)
    if content:
        print(f"assistant> {content}")


def chat_loop(
    brain: LLMBrain,
    handler: ToolHandler,
    messages: List[Dict[str, Any]],
    *,
    max_iterations: int = 20,
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
    parser = argparse.ArgumentParser(description="Interactive chat to orchestrator")
    parser.add_argument("--task", help="User task description; prompts if omitted")
    parser.add_argument("--parent-branch-id", required=True, help="Parent branch UUID")
    parser.add_argument("--project-name", help="Optional project name override")
    # Workspace dir defaults to /home/pan/workspace; no flag needed
    args = parser.parse_args(argv)

    try:
        cfg = AgentConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        logger.error("Configuration error: %s", exc)
        return 1

    if args.project_name:
        cfg = replace(cfg, project_name=args.project_name)
    # workspace_dir is fixed to /home/pan/workspace by default; env can override if needed

    if not cfg.project_name:
        logger.error("Project name must be provided via PROJECT_NAME or --project-name")
        return 1
    # workspace_dir is provided by config (defaults to /home/pan/workspace)

    task = args.task or input("you> Enter task description: ")
    if not task.strip():
        print("error: task is required", file=sys.stderr)
        return 1

    brain = LLMBrain(cfg.openai_api_key, cfg.openai_model)
    mcp_client = MCPClient(cfg.mcp_base_url)
    handler = ToolHandler(mcp_client, cfg.project_name)

    messages = build_initial_messages(task, cfg, args.parent_branch_id)
    final_report = chat_loop(brain, handler, messages)

    if not final_report:
        return 1

    final_report.setdefault("task", task)
    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
