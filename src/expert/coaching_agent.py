from __future__ import annotations

import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_COACHING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a professional parenting coach. "
            "Create a personalized coaching plan based on the dialogue analysis. "
            "Return a structured response in Korean with: "
            "1) 핵심 개선 포인트 (3-5개), 2) 구체적 실천 방법 (각 포인트당 2-3개), "
            "3) 다음 대화에서 시도해볼 기법, 4) 장기적 목표. "
            "Be specific, actionable, and encouraging."
        ),
    ),
    (
        "human",
        (
            "대화 요약:\n{summary}\n\n"
            "스타일 분석:\n{style_analysis}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "핵심 순간:\n{key_moments}\n\n"
            "개인화된 코칭 계획을 작성해주세요."
        ),
    ),
])


def coaching_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑧ coaching_plan: 코칭/실천 계획 (LLM)
    """
    summary = state.get("summary") or ""
    style_analysis = state.get("style_analysis") or {}
    patterns = state.get("patterns") or []
    key_moments = state.get("key_moments") or []
    
    if not summary and not patterns:
        return {
            "coaching_plan": {
                "improvement_points": [],
                "action_items": [],
                "next_techniques": [],
                "long_term_goals": []
            }
        }
    
    llm = get_llm(mini=False)
    
    # 포맷팅
    style_str = json.dumps(style_analysis, ensure_ascii=False, indent=2) if style_analysis else "(없음)"
    patterns_str = "\n".join([
        f"- {p.get('pattern_name')}: {p.get('description')}"
        for p in patterns
    ]) if patterns else "(없음)"
    key_moments_str = "\n".join([
        f"- {m.get('description')}"
        for m in key_moments
    ]) if key_moments else "(없음)"
    
    try:
        res = (_COACHING_PROMPT | llm).invoke({
            "summary": summary,
            "style_analysis": style_str,
            "patterns": patterns_str,
            "key_moments": key_moments_str,
        })
        coaching_text = getattr(res, "content", "") or str(res)
        
        # 구조화된 응답 파싱 시도 (간단한 파싱)
        return {
            "coaching_plan": {
                "full_text": coaching_text,
                "improvement_points": _extract_section(coaching_text, "핵심 개선 포인트"),
                "action_items": _extract_section(coaching_text, "실천 방법"),
                "next_techniques": _extract_section(coaching_text, "시도해볼 기법"),
                "long_term_goals": _extract_section(coaching_text, "장기적 목표")
            }
        }
    except Exception as e:
        print(f"Coaching plan error: {e}")
        return {
            "coaching_plan": {
                "full_text": "코칭 계획 생성 중 오류가 발생했습니다.",
                "improvement_points": [],
                "action_items": [],
                "next_techniques": [],
                "long_term_goals": []
            }
        }


def _extract_section(text: str, section_name: str) -> list:
    """텍스트에서 섹션 추출 (간단한 휴리스틱)"""
    lines = text.split('\n')
    in_section = False
    items = []
    
    for line in lines:
        if section_name in line:
            in_section = True
            continue
        if in_section:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.')):
                items.append(line.lstrip('-•1234567890. '))
            elif line and not line.startswith('#'):
                if items:  # 섹션이 끝난 것으로 간주
                    break
    return items

