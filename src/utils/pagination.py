from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List


def apply_filters(items: Iterable[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    return [x for x in items if predicate(x)]


def apply_sort(items: List[Dict[str, Any]], key: str, reverse: bool = False) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: x.get(key), reverse=reverse)


def paginate(items: List[Dict[str, Any]], page: int, size: int) -> List[Dict[str, Any]]:
    if page < 1:
        page = 1
    if size < 1:
        size = 10
    start = (page - 1) * size
    end = start + size
    return items[start:end]
