from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.services.agent.tools import all_agent_tool_names
from app.services.tools.proposals import load_runtime_tool_manifests


def load_agent_dynamic_tool_manifests(
    *,
    settings: Settings,
    agent_settings: Any,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    if not bool(getattr(agent_settings, "dynamic_tools_enabled", False)):
        return []
    try:
        return load_runtime_tool_manifests(
            data_dir=settings.data_dir,
            reserved_names=all_agent_tool_names(),
            deployment_id=str(getattr(agent_settings, "dynamic_tool_deployment_id", "local") or "local"),
        )
    except ValueError as exc:
        if logger is not None:
            logger.warning("dynamic tool manifest loading failed: %s", exc)
        return []
