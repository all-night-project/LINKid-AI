from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_STYLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert analyzing parenting communication style. "
            "Analyze the parent's communication style and ratios based on labeled utterances and patterns. "
            "Return ONLY a JSON object with: {style_type, label_distribution, positive_ratio, negative_ratio, "
            "command_ratio, question_ratio, reflection_ratio, overall_assessment}. "
            "style_type: 'authoritative', 'authoritarian', 'permissive', 'uninvolved', 'mixed'. "
            "No extra text."
        ),
    ),
    (
        "human",
        (
            "Labeled utterances:\n{utterances_labeled}\n\n"
            "Detected patterns:\n{patterns}\n\n"
            "Analyze communication style and return JSON object only."
        ),
    ),
])


def analyze_style_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑦ analyze_style: 스타일/비율 분석 (LLM + 패턴/라벨)
    """
    utterances_labeled = state.get("utterances_labeled") or []
    patterns = state.get("patterns") or []
    
    if not utterances_labeled:
        return {
            "style_analysis": {
                "style_type": "unknown",
                "label_distribution": {},
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "command_ratio": 0.0,
                "question_ratio": 0.0,
                "reflection_ratio": 0.0,
                "overall_assessment": "분석할 데이터가 없습니다."
            }
        }
    
    # 패턴/라벨 기반 통계 계산
    parent_utterances = [utt for utt in utterances_labeled if utt.get("speaker") == "Parent"]
    total_parent = len(parent_utterances)
    
    label_counts = {}
    for utt in parent_utterances:
        label = utt.get("label", "OTH")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 비율 계산
    positive_ratio = (label_counts.get("PR", 0) / total_parent) if total_parent > 0 else 0.0
    negative_ratio = (label_counts.get("NEG", 0) / total_parent) if total_parent > 0 else 0.0
    command_ratio = (label_counts.get("CMD", 0) / total_parent) if total_parent > 0 else 0.0
    question_ratio = (label_counts.get("Q", 0) / total_parent) if total_parent > 0 else 0.0
    reflection_ratio = (label_counts.get("RD", 0) / total_parent) if total_parent > 0 else 0.0
    
    # LLM 기반 스타일 분석
    utterances_str = "\n".join([
        f"[{utt.get('speaker')}] [{utt.get('label')}] {utt.get('text')}"
        for utt in utterances_labeled
    ])
    patterns_str = "\n".join([
        f"- {p.get('pattern_name')}: {p.get('description')}"
        for p in patterns
    ]) if patterns else "(없음)"
    
    try:
        llm = get_llm(mini=False)
        res = (_STYLE_PROMPT | llm).invoke({
            "utterances_labeled": utterances_str,
            "patterns": patterns_str,
        })
        content = getattr(res, "content", "") or str(res)
        
        # JSON 객체 파싱
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            style_analysis = json.loads(json_match.group(0))
            if isinstance(style_analysis, dict):
                # 통계 데이터 병합
                style_analysis["label_distribution"] = label_counts
                style_analysis["positive_ratio"] = positive_ratio
                style_analysis["negative_ratio"] = negative_ratio
                style_analysis["command_ratio"] = command_ratio
                style_analysis["question_ratio"] = question_ratio
                style_analysis["reflection_ratio"] = reflection_ratio
                return {"style_analysis": style_analysis}
    except Exception as e:
        print(f"Style analysis error: {e}")
    
    # 폴백: 통계 기반 스타일 추론
    style_type = "mixed"
    if command_ratio > 0.3 and negative_ratio > 0.2:
        style_type = "authoritarian"
    elif positive_ratio > 0.3 and reflection_ratio > 0.2:
        style_type = "authoritative"
    elif command_ratio < 0.1 and negative_ratio < 0.1:
        style_type = "permissive"
    
    return {
        "style_analysis": {
            "style_type": style_type,
            "label_distribution": label_counts,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "command_ratio": command_ratio,
            "question_ratio": question_ratio,
            "reflection_ratio": reflection_ratio,
            "overall_assessment": f"통계 기반 분석: {style_type} 스타일로 추정됩니다."
        }
    }

