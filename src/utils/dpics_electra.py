from __future__ import annotations

import os
from typing import List, Tuple, Optional
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import json
import re

from src.utils.dpics import _ALLOWED

# 모델의 전체 이름 라벨을 DPICS 코드로 매핑
_MODEL_LABEL_TO_DPICS = {
    "Behavior Description": "BD",
    "Command": "CMD",
    "Labeled Praise": "PR",
    "Unlabeled Praise": "PR",
    "Negative Talk": "NEG",
    "Neutral Talk": "NT",
    "Prosocial Talk": "PR",  # Prosocial Talk도 칭찬으로 분류
    "Question": "Q",
    "Reflective Statement": "RD",
}


def _normalize_text_for_model(text: str) -> str:
    """
    입력 텍스트를 모델 학습 시 사용한 형식으로 변환
    "Parent: ..." → "[MOM] ..."
    "Child: ..." → "[CHI] ..."
    
    Args:
        text: 원본 텍스트 (예: "Parent: How are you?")
        
    Returns:
        변환된 텍스트 (예: "[MOM] How are you?")
    """
    text = text.strip()
    
    # 이미 [MOM] 또는 [CHI] prefix가 있으면 그대로 반환
    if text.startswith("[MOM]") or text.startswith("[CHI]"):
        return text
    
    # Parent/Mom/Mother 등 → [MOM]
    parent_pattern = re.match(
        r'^(Parent|Mom|Dad|Mother|Father|부모|엄마|아빠|어머니|아버지)[:\s]+(.+)$',
        text,
        flags=re.IGNORECASE
    )
    if parent_pattern:
        return f"[MOM] {parent_pattern.group(2).strip()}"
    
    # Child/Kid/Son/Daughter 등 → [CHI]
    child_pattern = re.match(
        r'^(Child|Kid|Son|Daughter|아이|자녀|아들|딸)[:\s]+(.+)$',
        text,
        flags=re.IGNORECASE
    )
    if child_pattern:
        return f"[CHI] {child_pattern.group(2).strip()}"
    
    # 스피커 정보가 없으면 그대로 반환 (모델이 처리할 수 있도록)
    return text


class DPICSElectraModel:
    """DPICS 라벨링을 위한 ELECTRA 모델 래퍼"""
    
    def __init__(self, model_path: str = "/models/dpics-electra", device: Optional[str] = None):
        """
        Args:
            model_path: ELECTRA 모델이 저장된 경로
            device: 사용할 디바이스 ('cuda', 'cpu', None=자동 선택)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers 라이브러리가 필요합니다. "
                "pip install transformers torch 로 설치해주세요."
            )
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {model_path}")
        
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 모델과 토크나이저 로드
        print(f"DPICS ELECTRA 모델 로딩 중: {model_path} (device: {self.device})")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()  # 평가 모드
            
            # label_mapping.json 파일에서 라벨 매핑 로드 시도
            label_mapping_path = self.model_path / "label_mapping.json"
            if label_mapping_path.exists():
                with open(label_mapping_path, 'r', encoding='utf-8') as f:
                    label_mapping = json.load(f)
                # id2label을 정수 키로 변환
                id2label_raw = label_mapping.get("id2label", {})
                self.id2label = {int(k): v for k, v in id2label_raw.items()}
                self.label2id = {v: int(k) for k, v in id2label_raw.items()}
                print(f"label_mapping.json에서 라벨 매핑 로드: {self.id2label}")
            elif hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                # 모델 config에서 라벨 매핑 확인
                self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
                self.label2id = {v: int(k) for k, v in self.model.config.id2label.items()}
                print(f"모델 config에서 라벨 매핑 로드: {self.id2label}")
            else:
                raise RuntimeError("라벨 매핑을 찾을 수 없습니다. label_mapping.json 파일이 필요합니다.")
            
            # 모델 라벨을 DPICS 코드로 변환하는 매핑 생성
            self.id2dpics = {}
            for label_id, model_label in self.id2label.items():
                dpics_code = _MODEL_LABEL_TO_DPICS.get(model_label, "OTH")
                self.id2dpics[label_id] = dpics_code
            print(f"DPICS 코드 매핑: {self.id2dpics}")
            
            print("모델 로딩 완료")
        except Exception as e:
            raise RuntimeError(f"모델 로딩 실패: {e}")
    
    def predict(self, text: str, max_length: int = 512) -> str:
        """
        단일 텍스트에 대한 DPICS 라벨 예측
        
        Args:
            text: 예측할 텍스트 (예: "Parent: How are you?")
            max_length: 최대 토큰 길이
            
        Returns:
            DPICS 라벨 (PR, RD, BD, NT, Q, CMD, NEG, IGN, OTH 중 하나)
        """
        # 학습 시 사용한 형식으로 변환 ([MOM] 또는 [CHI] prefix 추가)
        normalized_text = _normalize_text_for_model(text)
        
        # 토큰화
        inputs = self.tokenizer(
            normalized_text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
        
        # 모델 라벨 ID를 DPICS 코드로 변환
        dpics_code = self.id2dpics.get(predicted_id, "OTH")
        
        # 허용된 라벨인지 확인
        if dpics_code not in _ALLOWED:
            dpics_code = "OTH"
        
        return dpics_code
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, max_length: int = 512) -> List[str]:
        """
        여러 텍스트에 대한 배치 예측
        
        Args:
            texts: 예측할 텍스트 리스트
            batch_size: 배치 크기
            max_length: 최대 토큰 길이
            
        Returns:
            DPICS 라벨 리스트
        """
        labels = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 학습 시 사용한 형식으로 변환 ([MOM] 또는 [CHI] prefix 추가)
            normalized_texts = [_normalize_text_for_model(text) for text in batch_texts]
            
            # 토큰화
            inputs = self.tokenizer(
                normalized_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 예측
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            
            # 모델 라벨 ID를 DPICS 코드로 변환
            batch_labels = []
            for pred_id in predicted_ids:
                dpics_code = self.id2dpics.get(pred_id, "OTH")
                if dpics_code not in _ALLOWED:
                    dpics_code = "OTH"
                batch_labels.append(dpics_code)
            
            labels.extend(batch_labels)
        
        return labels


# 전역 모델 인스턴스 (지연 로딩)
_model_instance: Optional[DPICSElectraModel] = None


def _get_project_root() -> Path:
    """프로젝트 루트 디렉토리 경로 반환"""
    # 현재 파일 위치: src/utils/dpics_electra.py
    # 프로젝트 루트: src/utils/../../ = 프로젝트 루트
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root


def _get_model() -> DPICSElectraModel:
    """전역 모델 인스턴스 가져오기 (싱글톤 패턴)"""
    global _model_instance
    if _model_instance is None:
        # 환경 변수로 경로 지정 가능, 없으면 프로젝트 루트 기준 상대 경로 사용
        env_path = os.getenv("DPICS_ELECTRA_MODEL_PATH")
        if env_path:
            model_path = env_path
        else:
            # 프로젝트 루트 기준 models/dpics-electra
            project_root = _get_project_root()
            model_path = str(project_root / "models" / "dpics-electra")
        _model_instance = DPICSElectraModel(model_path=model_path)
    return _model_instance


def label_lines_dpics_electra(text: str, use_batch: bool = True) -> List[Tuple[str, str]]:
    """
    ELECTRA 모델을 사용하여 DPICS 라벨링
    
    Args:
        text: 라벨링할 텍스트 (각 라인이 하나의 발화)
        use_batch: 배치 예측 사용 여부 (True면 더 빠름)
        
    Returns:
        (line, label) 튜플의 리스트
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers 라이브러리가 필요합니다. "
            "pip install transformers torch 로 설치해주세요."
        )
    
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    
    try:
        model = _get_model()
        
        if use_batch and len(lines) > 1:
            # 배치 예측
            labels = model.predict_batch(lines)
        else:
            # 개별 예측
            labels = [model.predict(line) for line in lines]
        
        # (line, label) 튜플 리스트 반환
        return list(zip(lines, labels))
    
    except Exception as e:
        print(f"ELECTRA 모델 예측 오류: {e}")
        # 폴백: 기본 라벨 반환
        return [(line, "OTH") for line in lines]


def reset_model_instance():
    """전역 모델 인스턴스 리셋 (테스트용)"""
    global _model_instance
    _model_instance = None

