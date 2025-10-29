from __future__ import annotations

from typing import Dict, Any

from src.utils.dpics import annotate_dialogue_dpics


def sentiment_label_node(state: Dict[str, Any]) -> Dict[str, Any]:
    dialogue = state.get("message") or state.get("dialogue") or ""
    if not dialogue or not str(dialogue).strip():
        return {"annotated": ""}
    annotated = annotate_dialogue_dpics(str(dialogue))
    return {"annotated": annotated}


if __name__ == "__main__":
    sample = {
        "message": "부모: 숙제 했니?\n아이: 하기 싫어.",
    }
    print(sentiment_label_node(sample))
