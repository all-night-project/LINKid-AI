from __future__ import annotations

import json
import re
from typing import List, Tuple, Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm, get_provider

# 간단 키워드 사전 (ko/en 혼합)
_POSITIVE = {
    "좋아", "잘했", "고마", "행복", "괜찮", "칭찬", "사랑", "응원", "기뻐", "멋지",
    "great", "good", "nice", "love", "thanks", "thank you", "well done", "awesome",
}
_NEGATIVE = {
    "싫어", "나빠", "화나", "짜증", "그만", "몰라", "못하", "미워", "힘들", "어려워",
    "no", "hate", "bad", "angry", "annoy", "stop", "don't", "cannot", "hard",
}
_NEGATIONS = {"아니", "아냐", "not", "don't", "no"}


def _score_line(text: str) -> int:
    lowered = text.lower()
    score = 0
    for w in _POSITIVE:
        if w in lowered:
            score += 1
    for w in _NEGATIVE:
        if w in lowered:
            score -= 1
    for w in _NEGATIONS:
        if w in lowered:
            score -= 1
    return score


def label_lines(text: str) -> List[Tuple[str, str]]:
    lines = [ln.strip() for ln in text.splitlines()]
    labeled: List[Tuple[str, str]] = []
    for ln in lines:
        if not ln:
            continue
        score = _score_line(ln)
        label = "negative" if score < 0 else "positive"
        labeled.append((ln, label))
    return labeled


def annotate_dialogue(text: str) -> str:
    labeled = label_lines(text)
    if not labeled:
        return text
    return "\n".join(f"[{label}] {ln}" for ln, label in labeled)


# ---------------- LLM 기반 라벨링 ---------------- #
_LLM_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You label each line as positive or negative, focusing on tone and affect. "
            "Return ONLY a JSON array of objects with keys 'line' and 'label' (positive|negative). "
            "Do not include extra text."
        ),
    ),
    (
        "human",
        (
            "Lines (keep original text exactly, one per line):\n{lines}\n\n"
            "Respond JSON array only."
        ),
    ),
])

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_labels_json(text: str) -> List[Tuple[str, str]]:
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        raise ValueError("no json array")
    data = json.loads(m.group(0))
    out: List[Tuple[str, str]] = []
    for item in data:
        line = str(item.get("line", "")).strip()
        label = str(item.get("label", "positive")).lower()
        label = "negative" if label == "negative" else "positive"
        if line:
            out.append((line, label))
    if not out:
        raise ValueError("empty labels")
    return out


def label_lines_llm(text: str) -> List[Tuple[str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    llm = get_llm(mini=True)
    res = (_LLM_PROMPT | llm).invoke({"lines": "\n".join(lines)})
    content = getattr(res, "content", "") or str(res)
    try:
        return _parse_labels_json(content)
    except Exception:
        # 폴백: 키워드 기반
        return label_lines(text)


def annotate_dialogue_llm(text: str) -> str:
    pairs = label_lines_llm(text)
    if not pairs:
        return text
    return "\n".join(f"[{label}] {ln}" for ln, label in pairs)
