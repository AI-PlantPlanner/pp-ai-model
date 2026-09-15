"""2단계: 모델 출력값을 화면 추천 카드 구조로 변환.

predict.py의 predict_all()이 주는 건 점수 계산 결과(식물명/학명/점수) 3개 컬럼뿐이라, 화면
카드에 필요한 나머지 정보(과/생육형태/감상포인트/관리 주기/병해충 등)는 plants_normalized.csv에
이미 있는 컬럼을 그대로 가져와 합칩니다. 즉 새 데이터를 만드는 게 아니라 "이미 있는 것 중 뭘
카드에 넣고 어떤 키 이름으로 부를지"를 정하는 매핑 작업입니다.

predict_all()을 그대로 쓰지 않고 predict.predict_score()를 종별로 직접 호출하는 이유는,
predict_all()이 반환하는 3개 컬럼만으로는 카드에 필요한 나머지 속성(과/생육형태 등)을 다시
찾아 매칭해야 해서 번거롭기 때문입니다. predict_score()는 종 1개(plant_row) 단위로 동작해서
그 row의 나머지 컬럼에 바로 접근할 수 있습니다. 스킵 조건(결측 컬럼/재배구분 미상)은
predict_all()과 동일하게 predict_score()가 처리합니다.

## 아직 비워둔 필드 (원예팀 검증 대기 중이라 값을 넣지 않음)
- grade: 점수 구간 -> 등급 변환(3번 작업). 등급 기준표가 정해지면 채웁니다.
- caution_sentence: 감점 사유 문장(4/5번 작업). teacher_scoring.score_plant()가 지금은 광량/
  온도/습도 감점을 합산한 최종 점수만 반환해서, 어느 요소 때문에 감점됐는지는 알 수 없습니다.
  요소별 점수를 따로 받을 수 있는지 유리님과 협의 후 채울 예정입니다.
카드 구조 자체에는 이 두 필드의 자리를 미리 만들어둬서, 나중에 로직이 정해지면 이 필드 값만
채우면 되고 카드 스키마(Unity가 파싱하는 JSON 구조)는 안 바뀌도록 했습니다.

## availability (재배 가능 여부) - 점수와는 별도로 지금 바로 계산 가능
teacher_scoring.is_gated()는 "재배구분 불일치" 또는 "한계온도 미달" 중 하나라도 걸리면 점수를
무조건 0으로 덮어씁니다. 그런데 점수만 보면 "적합도가 그냥 매우 낮음(0점에 가까움)"과
"애초에 이 환경에서 키울 수 없음(게이트에 걸림)"을 구분할 수 없습니다. 이 둘은 사용자에게 보여줄
문구가 완전히 달라야 하므로(전자는 "주의가 필요해요" 계열, 후자는 "이 환경에서는 키울 수
없어요" 계열), is_gated()와 동일한 두 조건을 여기서 각각 따로 확인해 구체적인 사유를
availability 필드에 담습니다.

주의: 아래 두 조건은 teacher_scoring.is_gated()의 판정 로직을 그대로 옮겨온 것입니다(원본을
수정하지 않기 위해 미러링). is_gated()의 로직이 바뀌면 여기도 같이 바꿔야 합니다. 이상적으로는
is_gated()가 True/False 대신 사유까지 반환하도록 유리님께 요청하는 게 더 안전합니다 - 4번
작업(감점 사유) 협의할 때 같이 얘기하면 좋을 것 같습니다.
"""

from typing import Optional

import pandas as pd

from src import config
from src.predict import predict_score
from ui_config.grade_resolver import resolve_grade
from ui_config.sentence_composer import compose_recommendation_sentence


def _availability(plant_row: pd.Series, user_temp: float, user_cultivation_context: str) -> str:
    """teacher_scoring.is_gated()와 동일한 두 조건을 각각 확인해 구체적인 사유를 반환.

    반환값: "available" | "unavailable_context"(실내/실외 불일치) |
            "unavailable_temperature"(한계온도 미달)
    """
    cultivation_type = plant_row[config.NORM_CULTIVATION_TYPE]
    if cultivation_type != config.CULTIVATION_MIXED and cultivation_type != user_cultivation_context:
        return "unavailable_context"
    if user_temp < plant_row[config.NORM_TEMP_LIMIT_MAX]:
        return "unavailable_temperature"
    return "available"


def _range_or_none(plant_row: pd.Series, min_col: str, max_col: str) -> Optional[dict]:
    lo, hi = plant_row.get(min_col), plant_row.get(max_col)
    if pd.isna(lo) or pd.isna(hi):
        return None
    return {"min": lo, "max": hi}


def _care_guide(plant_row: pd.Series) -> dict:
    pruning_type = plant_row.get("pruning_freq_type")
    if pruning_type == "as_needed":
        pruning = {"as_needed": True}
    else:
        pruning_range = _range_or_none(plant_row, "pruning_freq_min", "pruning_freq_max")
        pruning = {"as_needed": False, **pruning_range} if pruning_range else None

    return {
        "watering_per_week": _range_or_none(plant_row, "watering_freq_min", "watering_freq_max"),
        "fertilizing_per_month": _range_or_none(plant_row, "fertilizing_freq_min", "fertilizing_freq_max"),
        "pruning_per_year": pruning,
    }


def _plant_key(scientific_name: str) -> str:
    """Unity에서 스프라이트/에셋을 찾을 때 쓸 수 있는 안전한 키(학명 기반 slug)."""
    return (
        scientific_name.strip()
        .lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace(".", "")
    )


def build_recommendation_card(
    plant_row: pd.Series,
    score: float,
    user_temp: float,
    user_cultivation_context: str,
) -> dict:
    """식물 1종 + 예측 점수 -> 화면 카드 하나의 데이터 구조."""
    # pest_disease_list는 plants_normalized.csv에서 "|"로 이미 정리된 컬럼(예: "응애|진딧물").
    # 원본 컬럼(병해충)은 쉼표 구분이라 형식이 달라 정규화된 컬럼을 우선 사용한다.
    pest_raw = plant_row.get("pest_disease_list")
    if pd.isna(pest_raw) or pest_raw is None:
        pest_raw = plant_row.get(config.COL_PEST_DISEASE)
        pest_list = [p.strip() for p in str(pest_raw).split(",") if p.strip()] if pd.notna(pest_raw) else []
    else:
        pest_list = [p.strip() for p in str(pest_raw).split("|") if p.strip()]

    appreciation_raw = plant_row.get(config.COL_APPRECIATION_POINT)
    appreciation_list = (
        [a.strip() for a in str(appreciation_raw).split(",") if a.strip()]
        if pd.notna(appreciation_raw)
        else []
    )

    availability = _availability(plant_row, user_temp, user_cultivation_context)
    rounded_score = round(float(score), 1)

    card = {
        "plant_key": _plant_key(plant_row[config.COL_SCIENTIFIC_NAME]),
        "name": plant_row[config.COL_NAME],
        "scientific_name": plant_row[config.COL_SCIENTIFIC_NAME],
        "family": plant_row.get(config.COL_FAMILY),
        "growth_form": plant_row.get(config.COL_GROWTH_FORM),
        "appreciation_point": appreciation_list,
        "cultivation_type": plant_row[config.NORM_CULTIVATION_TYPE],
        "score": rounded_score,
        # availability != "available"이면 점수와 무관하게 "unavailable"로 확정, 그 외에는
        # grade_config.json의 min_score가 아직 비어있어(원예팀 검증 대기) None.
        "grade": resolve_grade(rounded_score, availability),
        "availability": availability,
        "caution_sentence": None,  # TODO(4번 작업): 요소별 감점 사유(유리님 데이터) 협의 후 채움
        "care_guide": _care_guide(plant_row),
        "pest_disease": pest_list,
    }
    # 재배 불가 사유는 지금 바로 문장으로 만들 수 있고, 등급 문장은 grade_config.json에 점수
    # 기준이 채워지는 순간부터 자동으로 채워집니다(코드 수정 불필요).
    card["recommendation_sentence"] = compose_recommendation_sentence(card)
    return card


def build_recommendation_cards(
    plants_df: pd.DataFrame,
    user_light: float,
    user_temp: float,
    user_humidity: float,
    user_cultivation_context: str,
) -> list[dict]:
    """전체 종에 대해 카드를 만들어 점수 내림차순으로 반환.

    predict_score()가 None을 반환하는 종(결측 컬럼 또는 재배구분 '미상')은 predict_all()과
    동일하게 결과에서 제외합니다.
    """
    cards = []
    for _, row in plants_df.iterrows():
        score = predict_score(row, user_light, user_temp, user_humidity, user_cultivation_context)
        if score is None:
            continue
        cards.append(build_recommendation_card(row, score, user_temp, user_cultivation_context))

    return sorted(cards, key=lambda c: c["score"], reverse=True)
