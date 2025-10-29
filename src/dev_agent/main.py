"""Minimal GPT-5 driven orchestrator for CLAUDE.md workflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import requests
from openai import OpenAI

from .config import AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dev_agent")

REVIEW_LOG_FILENAMES: Sequence[str] = ("codex_review.log", "claude_review.log")


class MCPError(RuntimeError):
    """Raised when MCP call fails."""


@dataclass
class MCPBranch:
    branch_id: str
    status: str


class MCPClient:
    """Tiny JSON-RPC client for MCP tools with retry logic."""

    def __init__(self, base_url: str, timeout: float = 30.0, max_retries: int = 3) -> None:
        self._base_url = base_url
        self._session = requests.Session()
        self._timeout = timeout
        self._request_id = 0
        self._max_retries = max_retries

    def _call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}

        last_exception = None
        for attempt in range(self._max_retries):
            try:
                response = self._session.post(self._base_url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                body = response.json()
                if "error" in body:
                    raise MCPError(body["error"])
                return body["result"]
            except (requests.RequestException, requests.Timeout, MCPError) as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"MCP call failed (attempt {attempt + 1}/{self._max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"MCP call failed after {self._max_retries} attempts: {e}")

        raise last_exception or MCPError("Unknown error occurred")

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


class GPT5Brain:
    """Slim wrapper around OpenAI GPT-5 Chat Completions API with retry logic."""

    def __init__(self, api_key: str, model: str, max_retries: int = 3) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_retries = max_retries

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        last_exception = None

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI returned empty response")
                return content
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self._max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API call failed after {self._max_retries} attempts: {e}")

        raise last_exception or RuntimeError("Unknown OpenAI API error")




PHASE1_SYSTEM_PROMPT = (
    "You help craft concise prompts for a claude_code agent that must follow the TDD "
    "workflow in CLAUDE.md. Output only the final prompt."
)

PHASE2_SYSTEM_PROMPT = (
    "You draft review instructions for a codex agent focusing strictly on P0/P1 issues "
    "based on CLAUDE.md guidance. Output only the final prompt."
)

PHASE3_SYSTEM_PROMPT = (
    "You write a remediation prompt for claude_code to fix high severity issues. "
    "Use context from the review worklog. Output only the final prompt."
)

WORKFLOW_DECISION_SYSTEM_PROMPT = (
    "You analyze claude_code and codex worklogs. Decide if more fixes are required. "
    "Respond with JSON: {\"proceed_to_fix\": bool, \"summary\": \"...\"}."
)


def build_phase1_prompt(task: str) -> str:
    user_prompt = (
        "Task description:\n"
        f"{task}\n\n"
        "Compose a single prompt for claude_code to:\n"
        "- Analyze the existing repo\n"
        "- Design a simple solution\n"
        "- Implement code and write matching tests (TDD)\n"
        "- Update worklog.md with tests and implementation summary\n"
        "Remind the agent to keep it simple and to commit only when tests pass."
    )
    return user_prompt


def build_phase2_prompt(task: str, worklog: str) -> str:
    user_prompt = (
        f"Task description:\n{task}\n\n"
        "Latest worklog.md content from implementation phase:\n"
        f"{worklog}\n\n"
        "Write a prompt for codex to:\n"
        "- Review tests + code for P0/P1 issues only\n"
        "- Provide evidence for each finding\n"
        "- Note findings in worklog.md and do not nitpick style\n"
        "- Read worklog.md before analyzing code\n"
    )
    return user_prompt


def build_phase3_prompt(task: str, review_worklog: str, review_notes: str) -> str:
    user_prompt = (
        f"Task description:\n{task}\n\n"
        "Latest worklog.md from codex review:\n"
        f"{review_worklog}\n\n"
        "Latest codex review notes:\n"
        f"{review_notes}\n\n"
        "Create a prompt for claude_code to:\n"
        "- Read worklog.md to see reported P0/P1 issues\n"
        "- Fix all confirmed problems and extend tests if needed\n"
        "- Keep solution minimal and high impact\n"
        "- Update worklog.md with fixes\n"
    )
    return user_prompt


def wait_for_branch(
    client: MCPClient,
    branch_id: str,
    poll_initial: timedelta,
    poll_max: timedelta,
    timeout: timedelta,
) -> MCPBranch:
    """Poll branch status until completion or timeout."""
    deadline = time.time() + timeout.total_seconds()
    interval = poll_initial.total_seconds()
    while True:
        info = client.get_branch(branch_id)
        status = info.get("status", "").lower()
        logger.info("Branch %s status: %s", branch_id, status)
        if status in {"succeed", "failed"}:
            return MCPBranch(branch_id=branch_id, status=status)
        if time.time() > deadline:
            raise TimeoutError(f"Branch {branch_id} polling exceeded {timeout}")
        time.sleep(interval)
        interval = min(interval * 2, poll_max.total_seconds())


def _extract_manifest_sources(branch_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect manifest dictionaries available in a branch response."""
    manifests: List[Dict[str, Any]] = []
    manifest = branch_info.get("manifest")
    if isinstance(manifest, dict):
        manifests.append(manifest)
    latest = branch_info.get("latest_snap")
    if isinstance(latest, dict):
        latest_manifest = latest.get("manifest")
        if isinstance(latest_manifest, dict):
            manifests.append(latest_manifest)
    return manifests


def _candidate_paths_from_manifests(branch_info: Dict[str, Any], filename: str) -> List[str]:
    """Return artifact paths whose basename matches filename."""
    matches: List[str] = []
    for manifest in _extract_manifest_sources(branch_info):
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Sequence):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            if isinstance(path, str) and path.split("/")[-1] == filename:
                matches.append(path)
    return matches


def _workspace_candidate(workspace_dir: Optional[str], filename: str) -> Optional[str]:
    if not workspace_dir:
        return None
    normalized = filename.lstrip("/")
    if not normalized:
        return None
    return f"{workspace_dir.rstrip('/')}/{normalized}"


def read_branch_text_file(
    client: MCPClient,
    branch_id: str,
    filename: str,
    workspace_dir: Optional[str] = None,
) -> Optional[str]:
    """Fetch a text file from a branch, leveraging workspace hints and manifests."""
    try:
        branch_info = client.get_branch(branch_id)
    except MCPError as exc:
        logger.warning("Failed to inspect branch %s: %s", branch_id, exc)
        branch_info = {}
    candidates: List[str] = []
    workspace_path = _workspace_candidate(workspace_dir, filename)
    if workspace_path:
        candidates.append(workspace_path)
    candidates.extend(_candidate_paths_from_manifests(branch_info, filename))
    basenames = {os.path.basename(path) for path in candidates}
    if filename not in basenames:
        candidates.append(filename)
    ordered: List[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    for path in ordered:
        try:
            result = client.branch_read_file(branch_id, path)
        except MCPError as exc:
            logger.debug("Unable to read %s from %s: %s", path, branch_id, exc)
            continue
        content = result.get("content")
        if isinstance(content, str):
            logger.info("Read %s from branch %s via %s", filename, branch_id, path)
            return content
    logger.info("No %s found in branch %s", filename, branch_id)
    return None


def read_review_notes(
    client: MCPClient,
    branch_id: str,
    workspace_dir: str,
) -> Optional[str]:
    """Try reading known review log filenames."""
    for candidate in REVIEW_LOG_FILENAMES:
        notes = read_branch_text_file(
            client,
            branch_id=branch_id,
            filename=candidate,
            workspace_dir=workspace_dir,
        )
        if notes:
            return notes
    return None


def run_phase(
    client: MCPClient,
    cfg: AgentConfig,
    project_name: str,
    parent_branch_id: str,
    prompt: str,
    agent: str,
) -> MCPBranch:
    """Fire a single-phase parallel_explore and wait for completion."""
    logger.info("Starting %s phase", agent)
    response = client.parallel_explore(
        project_name=project_name,
        parent_branch_id=parent_branch_id,
        prompts=[prompt],
        agent=agent,
        num_branches=1,
    )
    branches = response.get("branches", [])
    if not branches:
        raise RuntimeError("parallel_explore returned no branches")
    branch_id = branches[0]["branch_id"]
    logger.info("Spawned branch %s", branch_id)
    return wait_for_branch(
        client,
        branch_id=branch_id,
        poll_initial=cfg.poll_initial_interval,
        poll_max=cfg.poll_max_interval,
        timeout=cfg.poll_timeout,
    )


def decide_next_step(brain: GPT5Brain, task: str, logs: List[str]) -> Dict[str, Any]:
    combined = "\n\n---\n\n".join(logs)
    user_prompt = (
        f"Task description:\n{task}\n\n"
        "Relevant worklog.md snapshots in chronological order:\n"
        f"{combined}\n\n"
        "Decide if codex reported P0/P1 issues that require fixes. "
        "Return JSON with keys proceed_to_fix (true/false) and summary."
    )
    raw = brain.ask(WORKFLOW_DECISION_SYSTEM_PROMPT, user_prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON decision, defaulting to proceed.")
        return {"proceed_to_fix": True, "summary": raw.strip()}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GPT-5 driven CLAUDE.md orchestrator")
    parser.add_argument("--task", required=True, help="User task description")
    parser.add_argument("--parent-branch-id", required=True, help="Parent branch UUID")
    parser.add_argument("--project-name", required=False, help="Project name override")
    parser.add_argument("--workspace-dir", required=False, help="Workspace directory for branch snapshots")
    args = parser.parse_args(argv)

    try:
        cfg = AgentConfig.from_env()
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        return 1
    cfg = replace(
        cfg,
        project_name=args.project_name or cfg.project_name,
        workspace_dir=args.workspace_dir or cfg.workspace_dir,
    )

    if not cfg.project_name:
        logger.error("Project name must be provided via PROJECT_NAME or --project-name")
        return 1
    if not cfg.workspace_dir:
        logger.error("Workspace directory must be provided via WORKSPACE_DIR or --workspace-dir")
        return 1

    brain = GPT5Brain(cfg.openai_api_key, cfg.openai_model)
    mcp_client = MCPClient(cfg.mcp_base_url)

    # Phase 1 - Implementation
    phase1_prompt = brain.ask(PHASE1_SYSTEM_PROMPT, build_phase1_prompt(args.task))
    impl_branch = run_phase(
        mcp_client,
        cfg,
        project_name=cfg.project_name,
        parent_branch_id=args.parent_branch_id,
        prompt=phase1_prompt,
        agent="claude_code",
    )
    impl_worklog = read_branch_text_file(mcp_client, impl_branch.branch_id, cfg.worklog_filename) or ""

    # Phase 2 - Review
    phase2_prompt = brain.ask(PHASE2_SYSTEM_PROMPT, build_phase2_prompt(args.task, impl_worklog))
    review_branch = run_phase(
        mcp_client,
        cfg,
        project_name=cfg.project_name,
        parent_branch_id=impl_branch.branch_id,
        prompt=phase2_prompt,
        agent="codex",
    )
    review_worklog = read_branch_text_file(mcp_client, review_branch.branch_id, cfg.worklog_filename) or ""
    review_notes = read_review_notes(mcp_client, review_branch.branch_id, cfg.workspace_dir) or ""

    decision_logs = [log for log in [impl_worklog, review_worklog, review_notes] if log]
    decision = decide_next_step(brain, args.task, decision_logs)
    proceed_to_fix = bool(decision.get("proceed_to_fix"))

    final_branch = review_branch
    fix_worklog = ""
    fix_notes = ""

    if proceed_to_fix:
        phase3_prompt = brain.ask(
            PHASE3_SYSTEM_PROMPT,
            build_phase3_prompt(args.task, review_worklog, review_notes),
        )
        fix_branch = run_phase(
            mcp_client,
            cfg,
            project_name=cfg.project_name,
            parent_branch_id=review_branch.branch_id,
            prompt=phase3_prompt,
            agent="claude_code",
        )
        fix_worklog = read_branch_text_file(mcp_client, fix_branch.branch_id, cfg.worklog_filename) or ""
        fix_notes = read_review_notes(mcp_client, fix_branch.branch_id, cfg.workspace_dir) or ""
        final_branch = fix_branch

    # Final summary
    summary_logs = [
        log for log in [impl_worklog, review_worklog, review_notes, fix_worklog, fix_notes] if log
    ]
    final_summary = decision.get("summary", "")
    if not final_summary:
        final_summary = "\n\n".join(summary_logs)

    print(
        json.dumps(
            {
                "task": args.task,
                "final_branch_id": final_branch.branch_id,
                "summary": final_summary,
                "impl_branch_id": impl_branch.branch_id,
                "review_branch_id": review_branch.branch_id,
                "fix_branch_id": final_branch.branch_id if proceed_to_fix else None,
                "impl_status": impl_branch.status,
                "review_status": review_branch.status,
                "fix_status": final_branch.status if proceed_to_fix else None,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
