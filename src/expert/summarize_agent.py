from __future__ import annotations

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a parenting coach analyzing parent-child dialogue. "
            "Provide a comprehensive summary and diagnosis of today's interaction in Korean. "
            "Include: 1) 전체 대화 요약, 2) 주요 이슈, 3) 긍정적 측면, 4) 개선 필요 영역, 5) 오늘의 진단 요약. "
            "Be empathetic, specific, and actionable."
        ),
    ),
    (
        "human",
        (
            "한국어 원문 발화:\n{utterances_ko}\n\n"
            "라벨링된 발화:\n{utterances_labeled}\n\n"
            "탐지된 패턴:\n{patterns}\n\n"
            "오늘의 대화를 진단하고 요약해주세요."
        ),
    ),
])


def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑤ summarize: 오늘의 진단 (LLM)
    """
    utterances_ko = state.get("utterances_ko") or []
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    if not utterances_ko and not utterances_labeled:
        return {"summary": "대화 내용이 없어 분석할 수 없습니다."}
    
    llm = get_llm(mini=False)
    
    # 포맷팅
    utterances_ko_str = "\n".join(utterances_ko) if utterances_ko else "(없음)"
    utterances_labeled_str = "\n".join([
        f"[{utt.get('speaker')}] [{utt.get('label')}] {utt.get('text')}"
        for utt in utterances_labeled
    ]) if utterances_labeled else "(없음)"
    patterns_str = "\n".join([
        f"- {p.get('pattern_name')}: {p.get('description')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    try:
        res = (_SUMMARIZE_PROMPT | llm).invoke({
            "utterances_ko": utterances_ko_str,
            "utterances_labeled": utterances_labeled_str,
            "patterns": patterns_str,
        })
        summary = getattr(res, "content", "") or str(res)
        return {"summary": summary}
    except Exception as e:
        print(f"Summarize error: {e}")
        return {"summary": "요약 생성 중 오류가 발생했습니다."}

