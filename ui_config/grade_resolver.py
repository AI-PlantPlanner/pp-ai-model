"""3번 작업: 점수 구간을 등급으로 해석.

등급 기준(grade_config.json)은 원예팀 최종 검증 대기 중이라 min_score를 비워뒀습니다(null).
이 모듈은 그 상태에서도 안전하게 동작하도록, 임계값이 없는 등급은 건너뛰고 "아직 등급을 정할
수 없음"(None)을 반환합니다 — 등급 이름을 지어내거나 임의 숫자로 채우지 않습니다.

반면 availability(재배 가능 여부)는 원예팀 검증과 무관하게 이미 확정 가능한 정보라, score가
몇 점이든 availability != "available"이면 "unavailable" 등급을 바로 부여합니다.

원예팀 검증이 끝나면 할 일: grade_config.json의 각 grade.min_score에 실제 값만 채우면 됩니다.
이 파일(grade_resolver.py)이나 card_builder.py 코드는 수정할 필요가 없습니다.
"""

import json
from pathlib import Path
from typing import Optional

GRADE_CONFIG_PATH = Path(__file__).parent / "grade_config.json"

_grade_config_cache: Optional[dict] = None


def load_grade_config(path: Path = GRADE_CONFIG_PATH) -> dict:
    global _grade_config_cache
    if _grade_config_cache is None:
        _grade_config_cache = json.loads(path.read_text(encoding="utf-8"))
    return _grade_config_cache


def resolve_grade(score: float, availability: str, grade_config: Optional[dict] = None) -> Optional[str]:
    """score(0~100)와 availability를 등급 id로 변환.

    - availability != "available" -> "unavailable" (원예팀 검증과 무관하게 지금 확정 가능)
    - availability == "available"인데 아직 min_score가 하나도 설정 안 됨 -> None
      ("이 종은 등급을 매길 수 있지만, 등급 기준표가 아직 없다"는 뜻. "poor"나 0 같은 값으로
      임의 대체하지 않는다 — 실제로 낮은 등급인지 아니면 기준이 없어서인지 섞이면 안 되므로.)
    - grades는 grade_config["grades"]에 나열된 순서대로 확인해서, min_score가 설정돼 있고
      score >= min_score인 첫 번째 등급을 채택한다 (grades는 높은 min_score부터 정렬돼 있어야 함).
    """
    if grade_config is None:
        grade_config = load_grade_config()

    if availability != "available":
        return grade_config["unavailable_grade"]["id"]

    for grade in grade_config["grades"]:
        if grade["min_score"] is None:
            continue
        if score >= grade["min_score"]:
            return grade["id"]

    return None


def grade_label(grade_id: Optional[str], grade_config: Optional[dict] = None) -> Optional[str]:
    """등급 id -> 화면에 보여줄 한글 라벨. grade_id가 None이면 None을 그대로 반환."""
    if grade_id is None:
        return None
    if grade_config is None:
        grade_config = load_grade_config()

    if grade_id == grade_config["unavailable_grade"]["id"]:
        return grade_config["unavailable_grade"]["label"]
    for grade in grade_config["grades"]:
        if grade["id"] == grade_id:
            return grade["label"]
    return None
