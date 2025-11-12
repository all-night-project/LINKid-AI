from __future__ import annotations

from typing import Dict, Any


def aggregate_result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ⑩ aggregate_result: 최종 JSON 집계
    모든 분석 결과를 하나의 JSON으로 통합
    """
    result = {
        "summary": state.get("summary", ""),
        "key_moments": state.get("key_moments", []),
        "style_analysis": state.get("style_analysis", {}),
        "coaching_plan": state.get("coaching_plan", {}),
        "challenge_eval": state.get("challenge_eval", {}),
        "patterns": state.get("patterns", []),
        "meta": state.get("meta", {}),
    }
    
    return {"result": result}

