from __future__ import annotations

import json
import re
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_CHALLENGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert evaluating parent-child interaction challenges. "
            "Evaluate whether the parent met the challenge criteria based on labeled utterances and patterns. "
            "Return ONLY a JSON object with: {challenge_met, score, evidence, feedback, improvement_suggestions}. "
            "challenge_met: boolean, score: 0-100, evidence: list of specific examples. No extra text."
        ),
    ),
    (
        "human",
        (
            "Challenge specification:\n{challenge_spec}\n\n"
            "Labeled utterances:\n{utterances_labeled}\n\n"
            "Detected patterns:\n{patterns}\n\n"
            "Evaluate challenge completion and return JSON object only."
        ),
    ),
])


def challenge_eval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑨ challenge_eval: 챌린지 판정 (패턴/라벨 + spec)
    """
    challenge_spec = state.get("challenge_spec") or {}
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    if not challenge_spec:
        return {
            "challenge_eval": {
                "challenge_met": False,
                "score": 0,
                "evidence": [],
                "feedback": "챌린지 스펙이 제공되지 않았습니다.",
                "improvement_suggestions": []
            }
        }
    
    llm = get_llm(mini=False)
    
    # 포맷팅
    challenge_str = json.dumps(challenge_spec, ensure_ascii=False, indent=2)
    utterances_str = "\n".join([
        f"[{utt.get('speaker')}] [{utt.get('label')}] {utt.get('text')}"
        for utt in utterances_labeled
    ]) if utterances_labeled else "(없음)"
    patterns_str = "\n".join([
        f"- {p.get('pattern_name')}: {p.get('description')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    try:
        res = (_CHALLENGE_PROMPT | llm).invoke({
            "challenge_spec": challenge_str,
            "utterances_labeled": utterances_str,
            "patterns": patterns_str,
        })
        content = getattr(res, "content", "") or str(res)
        
        # JSON 객체 파싱
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            challenge_eval = json.loads(json_match.group(0))
            if isinstance(challenge_eval, dict):
                return {"challenge_eval": challenge_eval}
    except Exception as e:
        print(f"Challenge eval error: {e}")
    
    # 폴백: 패턴 기반 간단한 평가
    negative_patterns = [p for p in patterns if p.get("severity") in ["high", "medium"]]
    challenge_met = len(negative_patterns) == 0
    score = max(0, 100 - len(negative_patterns) * 20)
    
    return {
        "challenge_eval": {
            "challenge_met": challenge_met,
            "score": score,
            "evidence": [p.get("description") for p in negative_patterns[:3]],
            "feedback": f"패턴 기반 평가: {'챌린지를 달성했습니다' if challenge_met else '개선이 필요합니다'}.",
            "improvement_suggestions": [p.get("pattern_name") for p in negative_patterns[:3]]
        }
    }

