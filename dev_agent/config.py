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

    azure_api_key: str
    azure_endpoint: str
    azure_deployment: str
    azure_api_version: str = "2024-12-01-preview"
    mcp_base_url: str = "http://localhost:8000/mcp/sse"
    poll_initial_interval: timedelta = field(default_factory=lambda: timedelta(seconds=2))
    poll_max_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    poll_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    poll_backoff_factor: float = 2.0
    worklog_filename: str = "worklog.md"
    project_name: Optional[str] = None
    workspace_dir: Optional[str] = "/home/pan/workspace"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables with validation."""
        # Load .env if available (non-destructive)
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "python-dotenv is required to load configuration from .env files"
            ) from exc
        load_dotenv(override=False)
        # Validate Azure OpenAI credentials
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("AZURE_OPENAI_API_KEY must be set")

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise EnvironmentError("AZURE_OPENAI_ENDPOINT must be set")
        if not endpoint.startswith("https://"):
            raise ValueError("AZURE_OPENAI_ENDPOINT must start with 'https://'")

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise EnvironmentError("AZURE_OPENAI_DEPLOYMENT must be set")

        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION must be non-empty")

        # Validate base URL
        base_url = os.getenv("MCP_BASE_URL", "http://localhost:8000/mcp/sse")
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
        # Default workspace directory; allow override via env
        workspace_dir = os.getenv("WORKSPACE_DIR", "/home/pan/workspace")

        try:
            backoff = float(os.getenv("MCP_POLL_BACKOFF_FACTOR", "2.0"))
            if backoff <= 1.0:
                raise ValueError("Backoff factor must be greater than 1.0")
        except ValueError as exc:
            raise ValueError("MCP_POLL_BACKOFF_FACTOR must be a float greater than 1.0") from exc

        return cls(
            azure_api_key=api_key,
            azure_endpoint=endpoint.rstrip("/"),
            azure_deployment=deployment,
            azure_api_version=api_version,
            mcp_base_url=base_url,
            poll_initial_interval=poll_initial,
            poll_max_interval=poll_max,
            poll_timeout=poll_timeout,
            poll_backoff_factor=backoff,
            project_name=project,
            workspace_dir=workspace_dir,
        )
