from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from src.utils.common import get_llm


_PATTERN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert in analyzing parent-child interaction patterns. "
            "Detect specific interaction patterns from labeled utterances. "
            "Common patterns include: '긍정기회놓치기' (missed positive opportunity), "
            "'명령과제시' (command without choice), '공감부족' (lack of empathy), "
            "'반영부족' (lack of reflection), '비판적반응' (critical response), etc. "
            "Return ONLY a JSON array of objects with: {pattern_name, description, utterance_indices, severity}. "
            "severity: 'low', 'medium', 'high'. No extra text."
        ),
    ),
    (
        "human",
        (
            "Labeled utterances:\n{utterances_labeled}\n\n"
            "Detect interaction patterns and return JSON array only."
        ),
    ),
])


def detect_patterns_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ④ detect_patterns: 규칙/LLM로 패턴 찾기
    utterances_labeled를 받아서 탐지된 패턴들 반환
    """
    utterances_labeled = state.get("utterances_labeled") or []
    
    if not utterances_labeled:
        return {"patterns": []}
    
    # 규칙 기반 패턴 탐지
    patterns: List[Dict[str, Any]] = []
    
    # 패턴 1: 긍정기회놓치기 - 아이의 긍정적 행동에 칭찬 없음
    for i, utt in enumerate(utterances_labeled):
        if utt.get("speaker") == "Child" and utt.get("label") in ["BD"]:
            # 다음 부모 발화가 칭찬이 아닌 경우
            if i + 1 < len(utterances_labeled):
                next_utt = utterances_labeled[i + 1]
                if next_utt.get("speaker") == "Parent" and next_utt.get("label") != "PR":
                    patterns.append({
                        "pattern_name": "긍정기회놓치기",
                        "description": f"Child's positive behavior at index {i} was not praised",
                        "utterance_indices": [i, i + 1],
                        "severity": "medium"
                    })
    
    # 패턴 2: 명령과제시 - 질문 없이 명령만
    for i, utt in enumerate(utterances_labeled):
        if utt.get("speaker") == "Parent" and utt.get("label") == "CMD":
            patterns.append({
                "pattern_name": "명령과제시",
                "description": f"Command given without offering choice at index {i}",
                "utterance_indices": [i],
                "severity": "low"
            })
    
    # 패턴 3: 비판적반응 - 부모의 부정적 반응
    for i, utt in enumerate(utterances_labeled):
        if utt.get("speaker") == "Parent" and utt.get("label") == "NEG":
            patterns.append({
                "pattern_name": "비판적반응",
                "description": f"Critical response at index {i}",
                "utterance_indices": [i],
                "severity": "high"
            })
    
    # LLM 기반 추가 패턴 탐지
    try:
        llm = get_llm(mini=True)
        
        # 발화를 포맷팅
        utterances_str = "\n".join([
            f"{i}. [{utt.get('speaker')}] [{utt.get('label')}] {utt.get('text')}"
            for i, utt in enumerate(utterances_labeled)
        ])
        
        res = (_PATTERN_PROMPT | llm).invoke({"utterances_labeled": utterances_str})
        content = getattr(res, "content", "") or str(res)
        
        # JSON 배열 파싱
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            llm_patterns = json.loads(json_match.group(0))
            if isinstance(llm_patterns, list):
                patterns.extend(llm_patterns)
    except Exception as e:
        print(f"LLM pattern detection error: {e}")
    
    # 중복 제거 (간단한 휴리스틱)
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = (p.get("pattern_name"), tuple(p.get("utterance_indices", [])))
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    return {"patterns": unique_patterns}

