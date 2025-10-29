from __future__ import annotations

from typing import Dict, Any, TypedDict, List


class RouterState(TypedDict, total=False):
    message: str
    dialogue: str
    context: str
    tdl: Dict[str, Any]
    annotated: str
    highlights: List[str]
    advice: str
