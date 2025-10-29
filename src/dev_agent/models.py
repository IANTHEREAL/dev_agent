"""Data models used across the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class PhaseType(Enum):
    """Ordered phases for the CLAUDE.md TDD workflow."""

    IMPLEMENT = auto()
    REVIEW = auto()
    FIX = auto()


@dataclass
class TaskRequest:
    """Parsed user request."""

    description: str
    parent_branch_id: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PhasePrompt:
    """Prompt specification for a phase."""

    phase: PhaseType
    agent: str
    prompt: str
    model_hint: Optional[str] = None


@dataclass
class BranchInfo:
    """Minimal representation of a branch returned by parallel_explore."""

    branch_id: str
    status: str
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Result of executing a phase."""

    phase: PhaseType
    branch: BranchInfo
    worklog: Optional[str] = None
    issues_found: List[Dict[str, str]] = field(default_factory=list)
    succeeded: bool = True
    error: Optional[str] = None


@dataclass
class RetryPlan:
    """LLM-provided retry decision."""

    should_retry: bool
    revised_prompt: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class WorkflowDecision:
    """LLM decision for next workflow action."""

    proceed_to_fix: bool = False
    stop: bool = False
    summary: Optional[str] = None
