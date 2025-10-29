from __future__ import annotations

from typing import Any, Dict


def make_agent_config(thread_id: str | None = None) -> Dict[str, Any]:
    if not thread_id:
        return {}
    return {"configurable": {"thread_id": thread_id}}
