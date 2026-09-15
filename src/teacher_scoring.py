"""룰 기반 티처 스코어링.

사용자가 입력한 방 환경값(광량/온도/습도/실내·실외 구분)과 각 식물 종의 적정범위·한계값을
비교해 0~100점 적합도를 계산한다.
3~4단계에서 합성 환경 데이터에 자동으로 레이블을 매길 때 이 점수를 "정답(티처)"으로 쓴다.

## 채점 구조 (원예팀 '그룹별 판정 기준' 시트 기반)
광량/온도/습도 각각을 4단계로 판정한 뒤, 생육형 그룹별 가중치로 가중합한다.
    ① 거의 문제없음 (적정범위 안)   -> 100점
    ② 생육 둔화                     -> 100점에서 config.SCORE_LEVEL2_END까지 선형 감점
    ③ 생육 정지                     -> SCORE_LEVEL2_END에서 0점까지 선형 감점
    ④ 피해 발생                     -> 게이트(총점 0점)
④(총점 0점)를 적용하는 조건은 원예팀이 「0점 기준」으로 확정한 두 가지 —
한계온도 이탈, 재배구분 불일치 — 뿐이다. 판정 기준 시트에서 "③/④"로 병기되어
어느 단계인지 확정되지 않은 구간(광포화점 초과, 광보상점 미만, 습도 80% 초과)은
③으로 처리한다. 이 부분은 AI팀 잠정 판단이고 원예팀에 확인 요청을 보낸 상태다.

## 입력: 단일 시점이 아니라 기간 누적값
게임에서 일정 기간(3개월/6개월 등) 측정·계산한 환경값 배열을 받아 summarize_environment()로
요약한 뒤 채점한다. 감점은 기간 평균으로, 한계온도 게이트만 기간 극값으로 판정한다 —
게이트까지 평균으로 보면 겨울 한파가 평균에 묻혀 "적합"으로 추천된 식물이 겨울에 죽는다
(config.ENV_WINDOW_DAYS 주석의 실측 수치 참고).

## 광량만 배율로 채점하는 이유
종별 적정 광량이 300~70,000lux로 200배 넘게 차이나서, 절대 lux 차이로 감점하면
요구량이 큰 종(제라늄 30,000~70,000lux)이 어두운 방에서도 높은 점수를 받는다.
그래서 광량은 "적정범위의 몇 배를 벗어났는지"로 채점한다(로그 보간).
온도·습도는 종별 범위 폭이 비슷해서 절대값 기준을 그대로 쓴다.

## 광보상점/광포화점을 경계로 직접 쓰지 않는 이유
두 값은 4개 그룹당 하나씩만 부여되어(광보상점 2종류, 광포화점 4종류) 종별 변별력이 없다.
반면 적정 광량은 100종이 제각각인 종별 실측값이다. 그룹 공통값이 종별값을 덮어쓰면
같은 그룹이 한꺼번에 같은 점수가 되어 순위가 사라지므로, 광보상점/광포화점은
② 구간을 "넓히는" 방향으로만 반영한다(좁히지 않는다).
실제로 두 값이 적정 광량과 어긋나는 종이 88종 있다(data_normalization.detect_light_conflicts).

## 결측치 처리 방침
필요한 컬럼 중 하나라도 NaN이면 그 종의 점수를 중립값(예: 50점)이나 임의 추정치로 채우지 않고
None으로 반환한다. score_all_plants()는 이런 종을 결과에서 제외하고 콘솔에 경고로 남긴다 —
중립점수를 주면 실제로는 판단 불가능한 종이 "보통 수준으로 적합하다"는 잘못된 신호를
3~4단계 레이블링에 줄 수 있기 때문이다.

humidity_max만 예외로 필수 컬럼에서 뺐다 — "≥70"처럼 상한이 없는 개방형 범위 표기 때문에
정상적으로 NaN일 수 있어서다(파싱 실패와 구분됨). 이 경우 상한을 100%(물리적 상한)로 본다.
"""

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src import config

# 채점에 반드시 필요한 컬럼. humidity_max는 개방형 범위 때문에 정상적으로 NaN일 수 있어 제외한다.
REQUIRED_COLUMNS = (
    config.NORM_LIGHT_LUX_MIN,
    config.NORM_LIGHT_LUX_MAX,
    config.NORM_TEMP_OPTIMAL_MIN,
    config.NORM_TEMP_OPTIMAL_MAX,
    config.NORM_TEMP_LIMIT_LOWER,
    config.NORM_TEMP_LIMIT_UPPER,
    config.NORM_HUMIDITY_MIN,
)


@dataclass(frozen=True)
class EnvironmentWindow:
    """일정 기간 동안의 사용자 환경을 채점에 필요한 통계량으로 요약한 값.

    light/temp/humidity_mean: 감점 채점에 쓰는 기간 평균.
    temp_min/temp_max: 한계온도 게이트 판정에 쓰는 기간 극값. 평균이 아니라 극값을 쓰는 이유는
        원예팀 기준이 "한계온도를 벗어나면 0점"이고, 기간 중 한 번이라도 벗어나면 피해가
        발생하기 때문이다. 평균으로 판정하면 그 이탈이 묻힌다.
    n_samples: 요약에 쓰인 관측 수(1이면 단일 시점 입력).
    """

    light_mean: float
    temp_mean: float
    humidity_mean: float
    temp_min: float
    temp_max: float
    cultivation_context: str
    n_samples: int


def summarize_environment(
    light,
    temp,
    humidity,
    cultivation_context: str,
) -> EnvironmentWindow:
    """기간 누적 배열(또는 단일 값)을 EnvironmentWindow로 요약한다.

    각 인자는 배열/시리즈 또는 스칼라를 받는다. 스칼라를 주면 관측 1개짜리 기간으로 취급해
    기존의 단일 시점 채점과 동일하게 동작한다(기존 호출부 호환용).

    광량 평균은 산술평균을 쓴다 — 채점은 로그 스케일이지만, 게임에서 넘어오는 값의 의미가
    "기간 평균 광량"이라 여기서 기하평균으로 바꾸면 게임팀이 보낸 값과 다른 것을 채점하게 된다.
    """
    light_values = np.atleast_1d(np.asarray(light, dtype=float))
    temp_values = np.atleast_1d(np.asarray(temp, dtype=float))
    humidity_values = np.atleast_1d(np.asarray(humidity, dtype=float))

    if min(light_values.size, temp_values.size, humidity_values.size) == 0:
        raise ValueError("환경 값이 비어 있습니다 — 기간 배열에 관측이 최소 1개는 있어야 합니다.")

    return EnvironmentWindow(
        light_mean=float(light_values.mean()),
        temp_mean=float(temp_values.mean()),
        humidity_mean=float(humidity_values.mean()),
        temp_min=float(temp_values.min()),
        temp_max=float(temp_values.max()),
        cultivation_context=cultivation_context,
        n_samples=int(temp_values.size),
    )


def _score_by_ratio(ratio: float, ratio_level2: float, ratio_level3: float) -> float:
    """적정범위 대비 배율(ratio >= 1)을 점수로 변환한다.

    ratio=1(적정범위 경계)에서 100점, ratio_level2에서 SCORE_LEVEL2_END, ratio_level3에서 0점.
    배율은 곱셈 스케일이라 선형 보간하면 2배와 4배의 간격이 4배와 8배의 간격과 달라지므로
    로그 공간에서 보간한다.
    """
    if ratio <= 1:
        return 100.0
    log_ratio = math.log(ratio)
    log_level2 = math.log(ratio_level2)
    log_level3 = math.log(ratio_level3)
    if log_ratio <= log_level2:  # ② 생육 둔화
        return 100.0 - (100.0 - config.SCORE_LEVEL2_END) * log_ratio / log_level2
    if log_ratio >= log_level3:
        return 0.0
    # ③ 생육 정지
    return config.SCORE_LEVEL2_END * (1 - (log_ratio - log_level2) / (log_level3 - log_level2))


def _score_by_distance(distance: float, edge_level2: float, edge_level3: float) -> float:
    """적정범위 경계로부터의 거리(distance >= 0)를 점수로 변환한다.

    edge_level2/edge_level3도 경계로부터의 거리로 받는다(단위는 ℃ 또는 %p).
    구간 폭이 0인 경우(경계가 서로 붙은 경우)에도 0으로 나누지 않도록 방어한다.
    """
    if distance <= 0:
        return 100.0
    if distance >= edge_level3:
        return 0.0
    if distance <= edge_level2:  # ② 생육 둔화
        if edge_level2 <= 0:
            return config.SCORE_LEVEL2_END
        return 100.0 - (100.0 - config.SCORE_LEVEL2_END) * distance / edge_level2
    # ③ 생육 정지
    if edge_level3 <= edge_level2:
        return 0.0
    return config.SCORE_LEVEL2_END * (
        1 - (distance - edge_level2) / (edge_level3 - edge_level2)
    )


def light_score(plant_row: pd.Series, user_light: float) -> float:
    """광량 판정. 적정범위 대비 배율로 감점한다(모듈 docstring 참고)."""
    lux_min = plant_row[config.NORM_LIGHT_LUX_MIN]
    lux_max = plant_row[config.NORM_LIGHT_LUX_MAX]
    if lux_min <= user_light <= lux_max:
        return 100.0
    if user_light <= 0:
        return 0.0

    ratio_level2 = config.SCORE_LIGHT_RATIO_LEVEL2
    if user_light < lux_min:
        ratio = lux_min / user_light
        # 광보상점이 적정 하한보다 아래에 있으면 그 지점까지는 ②(생육 둔화)로 본다.
        compensation = plant_row[config.NORM_LIGHT_COMPENSATION_MIN]
        if pd.notna(compensation) and 0 < compensation < lux_min:
            ratio_level2 = max(ratio_level2, lux_min / compensation)
    else:
        ratio = user_light / lux_max
        # 광포화점이 적정 상한보다 위에 있으면 그 지점까지는 ②(생육 둔화)로 본다.
        saturation = plant_row[config.NORM_LIGHT_SATURATION_MAX]
        if pd.notna(saturation) and saturation > lux_max > 0:
            ratio_level2 = max(ratio_level2, saturation / lux_max)

    # ③ 구간 폭은 ② 구간과 같은 비례(기본 2배->5배)를 유지한다.
    ratio_level3 = ratio_level2 * (
        config.SCORE_LIGHT_RATIO_LEVEL3 / config.SCORE_LIGHT_RATIO_LEVEL2
    )
    return _score_by_ratio(ratio, ratio_level2, ratio_level3)


def temp_score(plant_row: pd.Series, user_temp: float) -> float:
    """온도 판정. 적정범위 밖은 한계온도까지 감점하고, 한계온도 밖은 게이트가 따로 처리한다.

    원예팀 기준은 "적정~한계온도 사이 = ② 생육 둔화"지만, 한계온도 상한(41~50℃)이
    생육 한계보다 치사 온도에 가까워 보여 그대로 쓰면 35℃에서도 감점이 거의 없다.
    그래서 ② 구간은 적정온도 범위 폭(최소 SCORE_TEMP_MIN_BAND)만큼만 잡고
    나머지 한계온도까지를 ③으로 둔다. 한계온도의 근거는 원예팀에 확인 요청한 상태다.
    """
    optimal_min = plant_row[config.NORM_TEMP_OPTIMAL_MIN]
    optimal_max = plant_row[config.NORM_TEMP_OPTIMAL_MAX]
    if optimal_min <= user_temp <= optimal_max:
        return 100.0

    band = max(config.SCORE_TEMP_MIN_BAND, optimal_max - optimal_min)
    if user_temp < optimal_min:
        distance = optimal_min - user_temp
        edge_level3 = optimal_min - plant_row[config.NORM_TEMP_LIMIT_LOWER]
    else:
        distance = user_temp - optimal_max
        edge_level3 = plant_row[config.NORM_TEMP_LIMIT_UPPER] - optimal_max

    # ②/③ 경계가 한계온도에 붙어 ③ 구간이 사라지면 한계온도 직전까지 높은 점수가 유지되다가
    # 게이트에서 통째로 0으로 떨어진다. 적정~한계 구간의 일정 비율은 ③으로 남겨 완만하게 잇는다.
    edge_level2 = min(band, edge_level3 * (1 - config.SCORE_TEMP_LEVEL3_MIN_RATIO))
    return _score_by_distance(distance, edge_level2, edge_level3)


def humidity_score(plant_row: pd.Series, user_humidity: float) -> float:
    """습도 판정.

    원예팀 기준은 "적정범위~80%가 ②, 80% 초과가 ③/④"이지만, 적정 습도 상한이 80% 이상인
    종이 9종(수국 65-85% 등) 있어 적정범위와 피해 구간이 겹친다. 그래서 80% 고정이 아니라
    "80%와 종별 적정 상한 중 높은 쪽"을 ② 구간 끝으로 쓴다 — AI팀 잠정 판단.
    """
    humidity_min = plant_row[config.NORM_HUMIDITY_MIN]
    humidity_max = plant_row[config.NORM_HUMIDITY_MAX]
    if pd.isna(humidity_max):
        # "≥70" 같은 개방형 범위 — 물리적 상한인 100%까지 전부 적정이라는 뜻.
        humidity_max = 100.0
    if humidity_min <= user_humidity <= humidity_max:
        return 100.0

    margin = config.SCORE_HUMIDITY_MIN_BAND
    if user_humidity < humidity_min:
        distance = humidity_min - user_humidity
        # 적정 하한이 이미 낮은 종은 ② 구간이 사라지지 않도록 최소 폭을 확보한다.
        level2_edge = min(config.SCORE_HUMIDITY_LEVEL2_LOWER, humidity_min - margin)
        edge_level2 = humidity_min - level2_edge
    else:
        distance = user_humidity - humidity_max
        level2_edge = max(config.SCORE_HUMIDITY_LEVEL2_UPPER, humidity_max + margin)
        edge_level2 = level2_edge - humidity_max
    return _score_by_distance(distance, edge_level2, edge_level2 + margin)


def group_weights(plant_row: pd.Series) -> tuple[float, float, float]:
    """생육형 그룹별 (광량, 온도, 습도) 가중치를 합이 1이 되도록 반환한다.

    그룹이 비어 있거나 알 수 없는 값이면 동일 비율로 처리한다(임의의 그룹으로 추측하지 않음).
    """
    group = plant_row.get(config.NORM_PLANT_GROUP)
    weights = config.GROUP_WEIGHTS.get(group)
    if weights is None:
        return config.DEFAULT_GROUP_WEIGHTS
    total = sum(weights)
    return tuple(w / total for w in weights)


def is_gated(plant_row: pd.Series, env: EnvironmentWindow) -> bool:
    """한계온도/재배구분 하드컷 게이트 중 하나라도 걸리면 True.

    원예팀이 「0점 기준」으로 확정한 두 조건이다. score_plant()와 predict.py(ML 예측의
    안전장치)가 공유해서 쓴다 — 생존/재배 가능 여부는 회귀모델의 근사치가 아니라
    이 규칙이 최종 결정권을 가져야 하기 때문이다.

    온도는 기간 평균이 아니라 기간 극값(temp_min/temp_max)으로 본다. 기간 중 한 번이라도
    한계온도를 벗어나면 피해가 발생하는데, 평균으로 판정하면 그 이탈이 묻히기 때문이다.

    한계온도가 범위로 기재된 종("41-50")은 과잉 제외를 피하려고 관대한 쪽을 쓴다:
    하한은 범위의 최솟값, 상한은 범위의 최댓값.
    """
    cultivation_type = plant_row[config.NORM_CULTIVATION_TYPE]
    if cultivation_type != config.CULTIVATION_MIXED and cultivation_type != env.cultivation_context:
        return True
    if env.temp_min < plant_row[config.NORM_TEMP_LIMIT_LOWER]:
        return True
    if env.temp_max > plant_row[config.NORM_TEMP_LIMIT_UPPER]:
        return True
    return False


def score_plant_detail(plant_row: pd.Series, env: EnvironmentWindow) -> dict | None:
    """총점과 함께 요인별 점수·게이트 사유를 반환한다. 필요한 컬럼에 결측이 있으면 None.

    추천 이유 설명(LLM 연동)이나 "광량 부족" 같은 경고 라벨에 쓰라고 분리해둔 함수다.
    가중합 구조상 한 요인이 0점이어도 총점은 절반 정도까지만 떨어지므로, 총점만으로는
    "이 방에서 이 식물이 왜 부적합한지"가 드러나지 않는다.
    """
    if any(pd.isna(plant_row[col]) for col in REQUIRED_COLUMNS):
        return None

    # 재배구분이 '미상'인 종은 실내/실외 재배 가능 여부 자체를 판단할 근거가 없으므로
    # (다른 결측 컬럼과 동일하게) 임의로 포함/제외하지 않고 None으로 제외한다.
    if plant_row[config.NORM_CULTIVATION_TYPE] == config.CULTIVATION_UNKNOWN:
        return None

    if is_gated(plant_row, env):
        cultivation_type = plant_row[config.NORM_CULTIVATION_TYPE]
        if cultivation_type != config.CULTIVATION_MIXED and cultivation_type != env.cultivation_context:
            reason = "재배구분 불일치"
        elif env.temp_min < plant_row[config.NORM_TEMP_LIMIT_LOWER]:
            reason = "기간 최저기온이 한계온도 하한 미만"
        else:
            reason = "기간 최고기온이 한계온도 상한 초과"
        return {"score": 0.0, "light": 0.0, "temp": 0.0, "humidity": 0.0, "gate_reason": reason}

    light = light_score(plant_row, env.light_mean)
    temp = temp_score(plant_row, env.temp_mean)
    humidity = humidity_score(plant_row, env.humidity_mean)
    weight_light, weight_temp, weight_humidity = group_weights(plant_row)
    total = light * weight_light + temp * weight_temp + humidity * weight_humidity
    return {
        "score": total,
        "light": light,
        "temp": temp,
        "humidity": humidity,
        "gate_reason": None,
    }


def score_plant(plant_row: pd.Series, env: EnvironmentWindow) -> float | None:
    """식물 1종의 적합도 점수(0~100)를 계산. 필요한 컬럼에 결측이 있으면 None."""
    detail = score_plant_detail(plant_row, env)
    return None if detail is None else detail["score"]


def score_all_plants(df: pd.DataFrame, env: EnvironmentWindow) -> pd.DataFrame:
    """전체 종의 점수를 계산해 (식물명, 학명, 점수) df를 점수 내림차순으로 반환.

    필요한 컬럼이 결측이거나(재배구분 '미상' 포함) 판단 불가한 종은 결과에서 제외하고
    콘솔에 경고로 남긴다.
    """
    names, sci_names, scores = [], [], []
    skipped = []
    for _, row in df.iterrows():
        score = score_plant(row, env)
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
