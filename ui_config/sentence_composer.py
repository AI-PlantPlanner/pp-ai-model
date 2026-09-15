"""등급/재배가능여부/감점 사유를 실제 문장으로 조립하는 모듈.

sentence_templates.json에 미리 써둔 문장 문구에, 식물 이름을 조사(은/는, 이/가)까지
자연스럽게 맞춰서 끼워 넣는 역할만 합니다. "어느 문장을 쓸지 판단하는 로직"과
"문장 자체의 문구"를 분리해서, 판단 로직에 필요한 데이터(원예팀 등급 기준, 유리님 요소별
점수)가 없어도 문구는 미리 준비해둘 수 있게 했습니다.
"""

import json
from pathlib import Path
from typing import Optional

from ui_config.korean_particles import eun_neun, i_ga

TEMPLATES_PATH = Path(__file__).parent / "sentence_templates.json"

_templates_cache: Optional[dict] = None


def load_templates(path: Path = TEMPLATES_PATH) -> dict:
    global _templates_cache
    if _templates_cache is None:
        _templates_cache = json.loads(path.read_text(encoding="utf-8"))
    return _templates_cache


def _fill(template: str, name: str) -> str:
    return template.format(name=name, eun_neun=eun_neun(name), i_ga=i_ga(name))


def compose_unavailable_sentence(
    name: str, availability: str, templates: Optional[dict] = None
) -> Optional[str]:
    templates = templates or load_templates()
    template = templates["unavailable_templates"].get(availability)
    return _fill(template, name) if template else None


def compose_grade_sentence(
    name: str, grade: Optional[str], templates: Optional[dict] = None
) -> Optional[str]:
    """grade가 None이면(등급 기준 미정) None을 그대로 반환 — 문구를 지어내지 않습니다."""
    if grade is None:
        return None
    templates = templates or load_templates()
    template = templates["grade_templates"].get(grade)
    return _fill(template, name) if template else None


def compose_recommendation_sentence(card: dict, templates: Optional[dict] = None) -> Optional[str]:
    """카드 하나를 받아서, 지금 확정 가능한 만큼만 문장을 만든다.

    재배 불가(availability != "available")면 그 이유 문장을 바로 돌려주고, 그 외에는
    등급이 이미 정해져 있으면(grade_config.json에 점수 기준이 채워진 뒤) 등급 문장을,
    등급도 아직 없으면(원예팀 검증 대기) None을 돌려준다.
    """
    templates = templates or load_templates()
    if card["availability"] != "available":
        return compose_unavailable_sentence(card["name"], card["availability"], templates)
    return compose_grade_sentence(card["name"], card.get("grade"), templates)


def compose_factor_caution_sentence(
    name: str, factor_key: str, templates: Optional[dict] = None
) -> Optional[str]:
    """요소별 주의 문구 미리보기 — 아직 카드에는 연결 안 됨(유리님 데이터 대기 중).

    factor_key 예: "light_too_low" / "light_too_high" / "temp_too_low" / "temp_too_high" /
    "humidity_too_low" / "humidity_too_high". 유리님이 요소별 점수를 주셔서 "어느 요소가
    문제인지" 판단할 수 있게 되면, 이 함수를 card_builder.py에 그대로 연결하면 됩니다.
    """
    templates = templates or load_templates()
    template = templates["factor_caution_templates"].get(factor_key)
    return _fill(template, name) if template else None
