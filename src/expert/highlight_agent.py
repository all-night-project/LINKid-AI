from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "Select 3-7 highlights from the dialogue strictly by choosing line indices. "
            "Return ONLY JSON: {{\"indices\": [int,…]}} where each index refers to the numbered lines below. "
            "If uncertain, include the first and last line as well. No extra text."
        ),
    ),
    ("human", "Numbered lines (index: text):\n{numbered}\n\nReturn JSON only with 'indices'."),
])

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*?\}")


def _number_lines(lines: List[str]) -> str:
    return "\n".join(f"{i}: {ln}" for i, ln in enumerate(lines))


def _parse_indices_or_highlights(text: str, lines: List[str]) -> List[str]:
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError("no json object")
    data = json.loads(m.group(0))

    # indices → exact lines
    if isinstance(data.get("indices"), list):
        picks: List[str] = []
        for x in data["indices"]:
            try:
                i = int(x)
                if 0 <= i < len(lines):
                    picks.append(lines[i])
            except Exception:
                continue
        if picks:
            return _dedup(picks)

    # highlights (strings) → map back to original lines
    if isinstance(data.get("highlights"), list):
        candidates: List[str] = []
        for s in data["highlights"]:
            s = str(s).strip()
            if not s:
                continue
            if s in lines:
                candidates.append(s)
                continue
            for ln in lines:
                if s in ln:
                    candidates.append(ln)
                    break
        if candidates:
            return _dedup(candidates)

    raise ValueError("no indices/highlights parsed")


def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _fallback_from_lines(lines: List[str]) -> List[str]:
    picks: List[str] = []
    if lines:
        picks.append(lines[0])
    picks.extend([ln for ln in lines if ln.endswith("?")][:3])
    picks.extend([ln for ln in lines if any(k in ln for k in ["싫어", "어려워", "힘들"])][:3])
    if len(lines) > 1:
        picks.append(lines[-1])
    return _dedup(picks)[:7]


def highlight_extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    dialogue = state.get("message") or state.get("dialogue") or ""
    if not dialogue or not str(dialogue).strip():
        return {"highlights": []}

    lines = [ln.strip() for ln in str(dialogue).splitlines() if ln.strip()]
    if not lines:
        return {"highlights": []}

    llm = get_llm(mini=True)
    numbered = _number_lines(lines)
    res = (_PROMPT | llm).invoke({"numbered": numbered})
    content = getattr(res, "content", "") or str(res)

    # Debug print for inspection
    print("\n=== Highlight Raw Model Output ===\n" + content + "\n=== END ===\n")

    try:
        picks = _parse_indices_or_highlights(content, lines)
        if not picks:
            raise ValueError("empty picks after parse")
        return {"highlights": picks[:10]}
    except Exception:
        return {"highlights": _fallback_from_lines(lines)}


if __name__ == "__main__":
    s = {"message": "부모: 숙제 했니?\n아이: 하기 싫어.\n부모: 왜?\n아이: 너무 어려워."}
    print(highlight_extract_node(s))
