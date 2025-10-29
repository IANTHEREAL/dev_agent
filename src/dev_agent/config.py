"""Configuration loading for the GPT-5 TDD agent orchestrator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional


def _get_env_timedelta(name: str, default_seconds: int) -> timedelta:
    value = os.getenv(name)
    if not value:
        return timedelta(seconds=default_seconds)
    try:
        seconds = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {value}") from exc
    return timedelta(seconds=seconds)


@dataclass(frozen=True)
class AgentConfig:
    """Holds runtime configuration for the orchestrator."""

    openai_api_key: str
    openai_model: str = "gpt-5"
    mcp_base_url: str = "http://localhost:8082/api/jsonrpc"
    poll_initial_interval: timedelta = field(default_factory=lambda: timedelta(seconds=2))
    poll_max_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    poll_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    poll_backoff_factor: float = 2.0
    worklog_filename: str = "worklog.md"
    project_name: Optional[str] = None
    workspace_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables with validation."""
        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY must be set")
        if not api_key.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY should start with 'sk-'")

        # Validate model
        model = os.getenv("GPT5_AGENT_MODEL", "gpt-5")
        valid_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-5"]
        if model not in valid_models:
            raise ValueError(f"Model '{model}' not in supported models: {valid_models}")

        # Validate base URL
        base_url = os.getenv("MCP_BASE_URL", "http://localhost:8082/api/jsonrpc")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("MCP_BASE_URL must be a valid HTTP/HTTPS URL")

        # Get timing configurations
        poll_initial = _get_env_timedelta("MCP_POLL_INITIAL_SECONDS", 2)
        poll_max = _get_env_timedelta("MCP_POLL_MAX_SECONDS", 30)
        poll_timeout = _get_env_timedelta("MCP_POLL_TIMEOUT_SECONDS", 600)

        # Validate timing relationships
        if poll_initial >= poll_max:
            raise ValueError("MCP_POLL_INITIAL_SECONDS must be less than MCP_POLL_MAX_SECONDS")
        if poll_timeout <= poll_max:
            raise ValueError("MCP_POLL_TIMEOUT_SECONDS must be greater than MCP_POLL_MAX_SECONDS")

        project = os.getenv("PROJECT_NAME")
        workspace_dir = os.getenv("WORKSPACE_DIR")

        try:
            backoff = float(os.getenv("MCP_POLL_BACKOFF_FACTOR", "2.0"))
            if backoff <= 1.0:
                raise ValueError("Backoff factor must be greater than 1.0")
        except ValueError as exc:
            raise ValueError("MCP_POLL_BACKOFF_FACTOR must be a float greater than 1.0") from exc

        return cls(
            openai_api_key=api_key,
            openai_model=model,
            mcp_base_url=base_url,
            poll_initial_interval=poll_initial,
            poll_max_interval=poll_max,
            poll_timeout=poll_timeout,
            poll_backoff_factor=backoff,
            project_name=project,
            workspace_dir=workspace_dir,
        )
