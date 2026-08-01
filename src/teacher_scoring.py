"""룰 기반 티처 스코어링(v0).

사용자가 입력한 방 환경값(광량/온도/습도/실내·실외 구분)과 각 식물 종의 적정범위·재배구분을
비교해 0~100점 적합도를 계산한다.
3~4단계에서 합성 환경 데이터에 자동으로 레이블을 매길 때 이 점수를 "정답(티처)"으로 쓴다.

## 결측치 처리 방침
필요한 컬럼(광량/온도/한계온도/습도 하한) 중 하나라도 NaN이면 그 종의 점수를 중립값(예: 50점)이나
임의 추정치로 채우지 않고 None으로 반환한다. score_all_plants()는 이런 종을 결과에서 제외하고
콘솔에 경고로 남긴다 — 중립점수를 주면 실제로는 판단 불가능한 종이 "보통 수준으로 적합하다"는
잘못된 신호를 3~4단계 레이블링에 줄 수 있기 때문이다.

humidity_max만 예외로 필수 컬럼에서 뺐다 — "≥70"처럼 상한이 없는 개방형 범위 표기 때문에
정상적으로 NaN일 수 있어서다(파싱 실패와 구분됨). 이 경우는 _linear_range_score가
assumed_range_width로 대체 처리한다.
"""

import pandas as pd

from src import config


def _linear_range_score(
    value: float, lo: float, hi: float, tolerance_ratio: float, assumed_range_width: float | None = None
) -> float:
    """적정범위 [lo, hi] 안이면 100점. 벗어나면 범위 폭 * tolerance_ratio만큼
    벗어날 때 0점이 되도록 선형 감점한다.

    hi가 없는 개방형 범위("≥N")는 lo 이상이면 전부 100점, 미만이면 assumed_range_width
    * tolerance_ratio를 감점 기준으로 쓴다 (범위 폭이 없어 실제 폭을 쓸 수 없기 때문).
    """
    if pd.isna(hi):
        if assumed_range_width is None:
            raise ValueError("hi가 없는(개방형) 범위는 assumed_range_width가 필요합니다.")
        if value >= lo:
            return 100.0
        distance = lo - value
        tolerance = assumed_range_width * tolerance_ratio
        return max(0.0, 100.0 * (1 - distance / tolerance))

    if lo <= value <= hi:
        return 100.0

    distance = lo - value if value < lo else value - hi
    tolerance = (hi - lo) * tolerance_ratio
    if tolerance <= 0:
        # 적정범위 폭이 0인 종(예: 25-25)은 공식을 그대로 적용하면 정확히
        # 일치할 때만 100점이고 그 외엔 0점이 된다. 별도 임의 여유값은 두지 않는다.
        return 0.0
    return max(0.0, 100.0 * (1 - distance / tolerance))


def score_plant(
    plant_row: pd.Series,
    user_light: float,
    user_temp: float,
    user_humidity: float,
    user_cultivation_context: str,
) -> float | None:
    """식물 1종의 적합도 점수(0~100)를 계산. 필요한 컬럼에 결측이 있으면 None.

    user_cultivation_context: 사용자 환경의 실내/실외 구분(config.CULTIVATION_INDOOR 또는
    CULTIVATION_OUTDOOR). 세부 장소(베란다/마당 등)를 실내·실외 중 무엇으로 볼지는 이 함수의
    관심사가 아니고, 호출부(향후 UX/합성 데이터 생성 단계)에서 이 둘 중 하나로 정리해서 넘겨준다.
    """
    # humidity_max는 필수에서 뺐다 — "≥70"처럼 상한이 없는 개방형 범위는 humidity_max가
    # 정상적으로 NaN이며(파싱 실패가 아님), _linear_range_score가 별도로 처리한다.
    required_cols = (
        config.NORM_LIGHT_LUX_MIN,
        config.NORM_LIGHT_LUX_MAX,
        config.NORM_TEMP_OPTIMAL_MIN,
        config.NORM_TEMP_OPTIMAL_MAX,
        config.NORM_TEMP_LIMIT_MIN,
        config.NORM_TEMP_LIMIT_MAX,
        config.NORM_HUMIDITY_MIN,
    )
    if any(pd.isna(plant_row[col]) for col in required_cols):
        return None

    # 재배구분이 '미상'인 종은 실내/실외 재배 가능 여부 자체를 판단할 근거가 없으므로
    # (다른 결측 컬럼과 동일하게) 임의로 포함/제외하지 않고 None으로 제외한다.
    cultivation_type = plant_row[config.NORM_CULTIVATION_TYPE]
    if cultivation_type == config.CULTIVATION_UNKNOWN:
        return None

    """재배구분 게이트: 실외 전용 식물을 실내 환경으로(혹은 그 반대로) 매칭하는 건
    광량/온도/습도가 아무리 잘 맞아도 애초에 그 환경에서 재배 자체가 성립하지 않는
    가능/불가능의 문제다 — 정도 차이가 아니므로 온도 게이트와 동일하게 점수를 0으로 덮어쓴다.
    '혼합'(실내·실외 다 가능)인 종은 게이트를 적용하지 않는다."""
    if cultivation_type != config.CULTIVATION_MIXED and cultivation_type != user_cultivation_context:
        return 0.0

    """한계온도(temp_limit)는 "이 아래로 내려가면 생존 불가"인 하한값이다.
    원본이 "3-5"처럼 범위로 들어온 경우 값이 애매하므로, 더 높은 쪽(temp_limit_max)을 컷 기준으로 써서 실제로는 위험한데 안전하다고 잘못 점수를 주는 걸 피한다.
    생존 불가는 광량/습도 정도 차이(품질 저하)와 성격이 달라 정도 차이가 아니라 가능/불가능의 이진 문제
    — 광량/습도가 완벽해도 얼어 죽으면 추천이 안 되므로 온도 feature 점수만 0으로 두지 않고 최종 점수 자체를 0으로 덮어쓴다(게이트)."""
    if user_temp < plant_row[config.NORM_TEMP_LIMIT_MAX]:
        return 0.0

    light_score = _linear_range_score(
        user_light,
        plant_row[config.NORM_LIGHT_LUX_MIN],
        plant_row[config.NORM_LIGHT_LUX_MAX],
        config.SCORE_TOLERANCE_RATIO,
    )
    humidity_score = _linear_range_score(
        user_humidity,
        plant_row[config.NORM_HUMIDITY_MIN],
        plant_row[config.NORM_HUMIDITY_MAX],
        config.SCORE_TOLERANCE_RATIO,
        assumed_range_width=config.SCORE_HUMIDITY_ASSUMED_RANGE_WIDTH,
    )
    temp_score = _linear_range_score(
        user_temp,
        plant_row[config.NORM_TEMP_OPTIMAL_MIN],
        plant_row[config.NORM_TEMP_OPTIMAL_MAX],
        config.SCORE_TOLERANCE_RATIO,
    )

    return (
        light_score * config.SCORE_WEIGHT_LIGHT
        + temp_score * config.SCORE_WEIGHT_TEMP
        + humidity_score * config.SCORE_WEIGHT_HUMIDITY
    )


def score_all_plants(
    df: pd.DataFrame,
    user_light: float,
    user_temp: float,
    user_humidity: float,
    user_cultivation_context: str,
) -> pd.DataFrame:
    """전체 종의 점수를 계산해 (식물명, 학명, 점수) df를 점수 내림차순으로 반환.

    필요한 컬럼이 결측이거나(재배구분 '미상' 포함) 판단 불가한 종은 결과에서 제외하고 콘솔에 경고로 남긴다.
    """
    names, sci_names, scores = [], [], []
    skipped = []
    for _, row in df.iterrows():
        score = score_plant(row, user_light, user_temp, user_humidity, user_cultivation_context)
        if score is None:
            skipped.append(row[config.COL_NAME])
            continue
        names.append(row[config.COL_NAME])
        sci_names.append(row[config.COL_SCIENTIFIC_NAME])
        scores.append(round(score, 1))

    if skipped:
        print(f"점수 계산 제외(결측 컬럼 또는 재배구분 미상): {len(skipped)}종 -> {skipped}")

    result = pd.DataFrame({"식물명": names, "학명": sci_names, "점수": scores})
    return result.sort_values("점수", ascending=False).reset_index(drop=True)
