from __future__ import annotations

import json
import re
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm

# DPICS 간략 코드 집합
# PR: Praise, RD: Reflection, BD: Behavior Description, NT: Neutral Talk,
# Q: Question, CMD: Command, NEG: Negative/Criticism, IGN: Ignore/Silence, OTH: Other
_ALLOWED = {"PR", "RD", "BD", "NT", "Q", "CMD", "NEG", "IGN", "OTH"}

_DPCS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You annotate each line of a parent-child dialogue using DPICS-style codes. "
            "Return ONLY a JSON array with objects: {{line, code}}. Codes: "
            "PR(Praise), RD(Reflection), BD(Behavior Description), NT(Neutral Talk), "
            "Q(Question), CMD(Command), NEG(Negative/Criticism), IGN(Ignore/Silence), OTH(Other). "
            "Choose the best single code per line. No extra text."
        ),
    ),
    (
        "human",
        (
            "Lines (one per line, keep text exactly):\n{lines}\n\n"
            "Respond with JSON array only."
        ),
    ),
])

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_dpics_json(text: str) -> List[Tuple[str, str]]:
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        raise ValueError("no json array")
    arr = json.loads(m.group(0))
    out: List[Tuple[str, str]] = []
    for item in arr:
        line = str(item.get("line", "")).strip()
        code = str(item.get("code", "OTH")).upper()
        code = code if code in _ALLOWED else "OTH"
        if line:
            out.append((line, code))
    if not out:
        raise ValueError("empty labels")
    return out


def label_lines_dpics_llm(text: str) -> List[Tuple[str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    llm = get_llm(mini=True)
    res = (_DPCS_PROMPT | llm).invoke({"lines": "\n".join(lines)})
    content = getattr(res, "content", "") or str(res)
    try:
        return _parse_dpics_json(content)
    except Exception:
        # 간단 폴백 휴리스틱
        out: List[Tuple[str, str]] = []
        for ln in lines:
            t = ln.strip()
            code = "NT"
            low = t.lower()
            if t.endswith("?") or "왜" in t or "어디" in t or "무엇" in t:
                code = "Q"
            elif any(k in low for k in ["해주세요", "해", "하지마", "그만", "지금", "해라"]):
                code = "CMD"
            elif any(k in low for k in ["잘했", "고마", "멋지", "great", "good", "nice"]):
                code = "PR"
            elif any(k in low for k in ["싫어", "나빠", "짜증", "못해", "미워", "bad", "hate"]):
                code = "NEG"
            out.append((t, code))
        return out


def annotate_dialogue_dpics(text: str) -> str:
    pairs = label_lines_dpics_llm(text)
    if not pairs:
        return text
    return "\n".join(f"[DPICS:{code}] {ln}" for ln, code in pairs)
