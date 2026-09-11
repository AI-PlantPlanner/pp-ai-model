"""6단계: 서빙용 예측 함수.

입력은 게임에서 넘어온 기간 누적 환경값이다 — teacher_scoring.summarize_environment()로
배열을 EnvironmentWindow로 요약해서 넘긴다(스칼라도 받는다).


ML모델(v2)은 학습 범위 안에서는 정확하지만(테스트셋 MAE는 train_model.py 실행 로그 참고), 학습 범위 밖 극단
조합이나 게이트 경계 근처에서는 근사 오차가 남을 수 있다. 생존/재배 가능 여부는 근사치로
어느 정도 맞으면 되는 문제가 아니므로, ML모델 예측과 무관하게 teacher_scoring.is_gated()로
한 번 더 강제 확인해 최종 점수를 덮어쓴다.
"""

import pandas as pd
from joblib import load

from src import config
from src.teacher_scoring import (
    REQUIRED_COLUMNS,
    EnvironmentWindow,
    is_gated,
    summarize_environment,
)
from src.train_model import build_features_v2

# 모델 입력을 만들 때 식물 쪽에서 가져와야 하는 컬럼.
# 결측 판정은 teacher_scoring.REQUIRED_COLUMNS를 그대로 쓴다 — 룰과 ML이 같은 종을
# 제외해야 두 경로의 결과가 어긋나지 않는다.
_PLANT_FEATURE_COLS = (
    config.NORM_LIGHT_LUX_MIN,
    config.NORM_LIGHT_LUX_MAX,
    config.NORM_LIGHT_SATURATION_MAX,
    config.NORM_LIGHT_COMPENSATION_MIN,
    config.NORM_TEMP_OPTIMAL_MIN,
    config.NORM_TEMP_OPTIMAL_MAX,
    config.NORM_TEMP_LIMIT_LOWER,
    config.NORM_TEMP_LIMIT_UPPER,
    config.NORM_HUMIDITY_MIN,
    config.NORM_HUMIDITY_MAX,
    config.NORM_CULTIVATION_TYPE,
    config.NORM_PLANT_GROUP,
)

_model_cache: dict | None = None


def load_model(path=config.MODEL_OUTPUT_PATH_V2) -> dict:
    """joblib 모델 번들({"model", "feature_columns"})을 로드하고 캐싱한다."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load(path)
    return _model_cache


def predict_score(
    plant_row: pd.Series,
    env: EnvironmentWindow,
    model_bundle: dict | None = None,
) -> float | None:
    """식물 1종의 적합도 점수(0~100)를 ML모델로 예측하되, 하드컷 게이트는 룰로 강제한다.

    재배구분이 '미상'이거나 필수 컬럼이 결측인 종은 teacher_scoring.score_plant()와
    동일하게 None을 반환한다(임의로 포함/제외하지 않음).
    """
    if pd.isna(plant_row[config.NORM_CULTIVATION_TYPE]) or (
        plant_row[config.NORM_CULTIVATION_TYPE] == config.CULTIVATION_UNKNOWN
    ):
        return None
    if any(pd.isna(plant_row[col]) for col in REQUIRED_COLUMNS):
        return None

    if is_gated(plant_row, env):
        return 0.0

    if model_bundle is None:
        model_bundle = load_model()
    model, feature_columns = model_bundle["model"], model_bundle["feature_columns"]

    row_df = pd.DataFrame([{
        "user_light_mean": env.light_mean,
        "user_temp_mean": env.temp_mean,
        "user_humidity_mean": env.humidity_mean,
        "user_temp_min": env.temp_min,
        "user_temp_max": env.temp_max,
        "user_cultivation_context": env.cultivation_context,
        **{col: plant_row[col] for col in _PLANT_FEATURE_COLS},
    }])
    features = build_features_v2(row_df).reindex(columns=feature_columns, fill_value=0)
    raw_score = model.predict(features)[0]
    # 학습 범위 밖 입력에서 트리 예측이 0~100을 살짝 벗어날 수 있어 안전하게 clip한다.
    return float(min(100.0, max(0.0, raw_score)))


def predict_all(plants_df: pd.DataFrame, env: EnvironmentWindow) -> pd.DataFrame:
    """전체 종의 점수를 계산해 (식물명, 학명, 점수) df를 점수 내림차순으로 반환.

    teacher_scoring.score_all_plants()와 동일한 인터페이스를 갖는다 — 다만 연속 점수는
    ML모델(v2)로 계산하고, 게이트만 룰로 강제한다는 점이 다르다.
    """
    model_bundle = load_model()
    names, sci_names, scores = [], [], []
    skipped = []
    for _, row in plants_df.iterrows():
        score = predict_score(row, env, model_bundle)
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


def main() -> None:
    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    # 게임에서 넘어오는 기간 누적 배열 대신, 예시로 단일 값을 넣어 호출한다.
    # summarize_environment()는 배열과 스칼라를 모두 받는다.
    env = summarize_environment(
        light=15000,
        temp=25,
        humidity=55,
        cultivation_context=config.CULTIVATION_INDOOR,
    )
    result = predict_all(plants_df, env)
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
