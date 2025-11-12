from __future__ import annotations

import re
from typing import Dict, Any, List


def preprocess_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ① preprocess: 스피커 정규화
    utterances_ko를 받아서 스피커를 정규화한 utterances_normalized 반환
    반환 형식: [{speaker: "MOM" | "CHI", 발화내용_ko: str}, ...]
    """
    utterances_ko = state.get("utterances_ko") or []
    
    if not utterances_ko:
        # 기존 message/dialogue에서 파싱 시도
        dialogue = state.get("message") or state.get("dialogue") or ""
        if dialogue:
            utterances_ko = [ln.strip() for ln in str(dialogue).splitlines() if ln.strip()]
    
    if not utterances_ko:
        return {"utterances_normalized": []}
    
    normalized: List[Dict[str, str]] = []
    
    for utt in utterances_ko:
        utt_str = str(utt).strip()
        if not utt_str:
            continue
        
        # 스피커 추출 및 정규화
        speaker = None
        발화내용_ko = utt_str
        
        # 대괄호 패턴 매칭 (우선 처리)
        bracket_match = re.match(r'^\[(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father|아이|자녀|아들|딸|Child|Kid|Son|Daughter)\]\s*(.+)$', 
                                utt_str, flags=re.IGNORECASE)
        if bracket_match:
            speaker_label = bracket_match.group(1).lower()
            발화내용_ko = bracket_match.group(2).strip()
            # 부모 패턴
            if speaker_label in ['부모', '엄마', '아빠', '어머니', '아버지', 'parent', 'mom', 'dad', 'mother', 'father']:
                speaker = "MOM"
            # 아이 패턴
            elif speaker_label in ['아이', '자녀', '아들', '딸', 'child', 'kid', 'son', 'daughter']:
                speaker = "CHI"
        
        # 대괄호 패턴이 없으면 기존 패턴 매칭
        if speaker is None:
            # 부모 패턴 매칭
            parent_match = re.match(r'^(부모|엄마|아빠|어머니|아버지|Parent|Mom|Dad|Mother|Father)[:\s]+(.+)$', 
                                   utt_str, flags=re.IGNORECASE)
            if parent_match:
                speaker = "MOM"
                발화내용_ko = parent_match.group(2).strip()
            else:
                # 아이 패턴 매칭
                child_match = re.match(r'^(아이|자녀|아들|딸|Child|Kid|Son|Daughter)[:\s]+(.+)$', 
                                      utt_str, flags=re.IGNORECASE)
                if child_match:
                    speaker = "CHI"
                    발화내용_ko = child_match.group(2).strip()
        
        # 스피커가 없으면 이전 스피커 추론 (간단한 휴리스틱)
        if speaker is None:
            if normalized:
                last_speaker = normalized[-1].get("speaker")
                # 이전 발화와 다른 스피커로 가정 (대화는 주로 교대로 진행)
                if last_speaker == "MOM":
                    speaker = "CHI"
                elif last_speaker == "CHI":
                    speaker = "MOM"
                else:
                    speaker = "MOM"  # 기본값
            else:
                speaker = "MOM"  # 첫 발화는 기본적으로 부모
        
        normalized.append({
            "speaker": speaker,
            "발화내용_ko": 발화내용_ko
        })
    
    return {"utterances_normalized": normalized}

