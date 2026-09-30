"""2단계: 모델 출력값을 화면 추천 카드 구조로 변환.

predict_all()을 그대로 쓰지 않고 predict.predict_score()를 종별로 직접 호출하는 이유는,
predict_all()이 반환하는 3개 컬럼(식물명/학명/점수)만으로는 카드에 필요한 나머지 속성(과/
생육형태 등)을 다시 찾아 매칭해야 해서 번거롭기 때문입니다. predict_score()는 종 1개
(plant_row) 단위로 동작해서 그 row의 나머지 컬럼에 바로 접근할 수 있습니다. 스킵 조건(결측
컬럼/재배구분 미상)은 predict_all()과 동일하게 predict_score()가 처리합니다.

## 입력값: EnvironmentWindow
유리님이 만든 src.teacher_scoring.EnvironmentWindow를 그대로 받습니다(단일 시점 스냅샷이
아니라 기간 누적값). 선택지 기반 입력은 ui_config.input_resolver.resolve_environment_window(),
기록 누적 입력은 ui_config.environment_log.resolve_averaged_model_input()으로 만들 수 있습니다.

## 아직 비워둔 필드
- grade: 점수 구간 -> 등급 변환(3번 작업). 등급 기준표(grade_config.json의 min_score)가
  정해지면 채웁니다. 이건 원예팀 최종 검증이 필요한 부분이라 아직 비워둠.

## caution_sentence (4번 작업) - 이제 연결함
"몇 점부터 요인이 문제인지"는 원예팀이 이 용도로 직접 답해준 숫자는 아니지만, 이미 검증받은
①~④ 4단계 판정 구조에서 그대로 가져왔습니다. config.SCORE_LEVEL2_END(=40)가 "②생육 둔화 ->
③생육 정지"로 넘어가는 경계라서, 요인별 점수가 이 밑으로 떨어지면 "생육이 사실상 멈추는
수준"이라는 뜻이라 주의 문구 트리거로 재사용했습니다(원예팀이 이 정확한 목적으로 승인한 숫자는
아니라서, 게임팀/원예팀 논의 결과에 따라 바뀔 수 있는 잠정 기준입니다 - FACTOR_CAUTION_THRESHOLD
하나만 바꾸면 됩니다). "부족한지 초과한지" 방향은 원예팀 승인이 필요 없는 단순 비교라서
(사용자 환경값이 그 식물 적정범위보다 낮은지/높은지) 바로 계산했습니다.

## availability (재배 가능 여부) - teacher_scoring.score_plant_detail()의 gate_reason 그대로 사용
예전에는 teacher_scoring.is_gated()의 판정 로직(재배구분 불일치/한계온도 이탈)을 이 파일에
그대로 옮겨 적어(미러링) 썼는데, 원본이 바뀌면 여기도 같이 바꿔야 하는 문제가 있었습니다.
유리님이 score_plant_detail()을 만들면서 게이트 사유(gate_reason)까지 반환하게 됐으므로,
이제는 로직을 옮겨 적지 않고 그 반환값을 그대로 화면용 카테고리로 매핑만 합니다.
"""

from typing import Optional

import pandas as pd

from src import config
from src.predict import predict_score
from src.teacher_scoring import EnvironmentWindow, score_plant_detail
from ui_config.grade_resolver import resolve_grade
from ui_config.sentence_composer import compose_factor_caution_sentence, compose_recommendation_sentence

# "②생육 둔화 -> ③생육 정지" 경계(원예팀이 확정한 4단계 구조에서 그대로 가져옴)를 주의 문구
# 트리거로 재사용한다. 이 정확한 용도로 원예팀이 승인한 숫자는 아니므로, 게임팀/원예팀 논의
# 결과에 따라 이 상수만 바꾸면 된다.
FACTOR_CAUTION_THRESHOLD = config.SCORE_LEVEL2_END


def _availability(plant_row: pd.Series, detail: dict) -> str:
    """score_plant_detail()의 gate_reason -> 화면용 availability 문자열.

    반환값: "available" | "unavailable_context"(실내/실외 불일치) |
            "unavailable_temperature"(한계온도 상한 초과 또는 하한 미달)
    """
    reason = detail["gate_reason"]
    if reason is None:
        return "available"
    if reason == "재배구분 불일치":
        return "unavailable_context"
    return "unavailable_temperature"


def _factor_caution_key(plant_row: pd.Series, env: EnvironmentWindow, factor_scores: dict) -> Optional[str]:
    """factor_scores 중 임계값 밑으로 떨어진 요인이 있으면 "요인_방향" 키를 반환.

    여러 요인이 동시에 임계값 밑이어도 지금은 가장 점수가 낮은(가장 심각한) 요인 하나만
    고른다 — 여러 개를 한꺼번에 보여줄지는 게임팀 UI 논의에 달려있어서, 우선 "가장 심각한
    것 하나" 방식으로 구현해뒀다. 반환값 예: "light_too_low", "temp_too_high".
    """
    candidates = []  # (score, factor_caution_key)

    if factor_scores["light"] < FACTOR_CAUTION_THRESHOLD:
        too_low = env.light_mean < plant_row[config.NORM_LIGHT_LUX_MIN]
        key = "light_too_low" if too_low else "light_too_high"
        candidates.append((factor_scores["light"], key))

    if factor_scores["temp"] < FACTOR_CAUTION_THRESHOLD:
        too_low = env.temp_mean < plant_row[config.NORM_TEMP_OPTIMAL_MIN]
        key = "temp_too_low" if too_low else "temp_too_high"
        candidates.append((factor_scores["temp"], key))

    if factor_scores["humidity"] < FACTOR_CAUTION_THRESHOLD:
        too_low = env.humidity_mean < plant_row[config.NORM_HUMIDITY_MIN]
        key = "humidity_too_low" if too_low else "humidity_too_high"
        candidates.append((factor_scores["humidity"], key))

    if not candidates:
        return None
    _, worst_key = min(candidates, key=lambda c: c[0])
    return worst_key


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
    env: EnvironmentWindow,
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

    # predict_score()가 이미 점수를 반환했다면(None이 아니라면) 같은 필터링 조건(REQUIRED_COLUMNS,
    # 재배구분 미상)을 쓰는 score_plant_detail()도 반드시 dict를 반환해야 한다. None이 나오면
    # 두 필터링 조건이 어긋난 것이므로 조용히 넘기지 않고 바로 에러를 낸다.
    detail = score_plant_detail(plant_row, env)
    if detail is None:
        raise ValueError(
            f"{plant_row.get(config.COL_NAME)}: predict_score()는 점수를 반환했는데 "
            "score_plant_detail()은 None입니다 — 두 필터링 조건이 어긋났을 수 있습니다."
        )

    availability = _availability(plant_row, detail)
    rounded_score = round(float(score), 1)
    factor_scores = {
        "light": round(detail["light"], 1),
        "temp": round(detail["temp"], 1),
        "humidity": round(detail["humidity"], 1),
    }

    # 재배 자체가 불가능한 카드는 이미 recommendation_sentence로 사유를 알려주므로, 요인별
    # 주의 문구는 재배 가능한 카드에만 붙인다(게이트 걸리면 요인 점수가 전부 0으로 나와서
    # 그대로 적용하면 의미 없는 "전부 문제" 문구가 나오게 됨).
    caution_key = _factor_caution_key(plant_row, env, factor_scores) if availability == "available" else None
    caution_sentence = (
        compose_factor_caution_sentence(plant_row[config.COL_NAME], caution_key) if caution_key else None
    )

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
        # 요인별 점수(룰 기반, score_plant_detail() 산출). 최종 "score"는 ML모델(v2) 예측값이라
        # 서로 값이 정확히 일치하진 않지만, "어느 요인이 낮은지" 설명용으로는 이 값을 쓴다 —
        # teacher_scoring.py에도 이 함수가 "추천 이유 설명"용으로 분리돼 있다고 적혀있다.
        "factor_scores": factor_scores,
        "caution_sentence": caution_sentence,
        "care_guide": _care_guide(plant_row),
        "pest_disease": pest_list,
    }
    # 재배 불가 사유는 지금 바로 문장으로 만들 수 있고, 등급 문장은 grade_config.json에 점수
    # 기준이 채워지는 순간부터 자동으로 채워집니다(코드 수정 불필요).
    card["recommendation_sentence"] = compose_recommendation_sentence(card)
    return card


def build_recommendation_cards(plants_df: pd.DataFrame, env: EnvironmentWindow) -> list[dict]:
    """전체 종에 대해 카드를 만들어 점수 내림차순으로 반환.

    predict_score()가 None을 반환하는 종(결측 컬럼 또는 재배구분 '미상')은 predict_all()과
    동일하게 결과에서 제외합니다.
    """
    cards = []
    for _, row in plants_df.iterrows():
        score = predict_score(row, env)
        if score is None:
            continue
        cards.append(build_recommendation_card(row, score, env))

    return sorted(cards, key=lambda c: c["score"], reverse=True)
