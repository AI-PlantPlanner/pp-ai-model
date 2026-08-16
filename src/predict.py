"""6단계: 서빙용 예측 함수.

ML모델(v1)은 synthetic_training_data.csv 범위 안에서는 매우 정확하지만(테스트셋 MAE
약 1.7점), 학습 범위 밖 극단 조합이나 게이트 경계 바로 근처에서는 근사 오차가 남을 수
있다(예: 한계온도 바로 아래인데도 모델이 생존 가능하다고 오판하는 경우 — 실측 확인 결과
오차 최대 16.5점까지 발생). 생존 가능/재배 가능 여부는 "근사치로 어느 정도 맞으면 되는"
문제가 아니라 정확히 지켜져야 하는 조건이므로, ML모델의 연속 예측값과 무관하게
teacher_scoring.is_gated()로 한 번 더 강제 확인해 최종 점수를 덮어쓴다.
"""

import pandas as pd
from joblib import load

from src import config
from src.teacher_scoring import is_gated
from src.train_model import build_features_v1

_PLANT_FEATURE_COLS = (
    config.NORM_LIGHT_LUX_MIN,
    config.NORM_LIGHT_LUX_MAX,
    config.NORM_TEMP_OPTIMAL_MIN,
    config.NORM_TEMP_OPTIMAL_MAX,
    config.NORM_TEMP_LIMIT_MIN,
    config.NORM_TEMP_LIMIT_MAX,
    config.NORM_HUMIDITY_MIN,
    config.NORM_HUMIDITY_MAX,
    config.NORM_CULTIVATION_TYPE,
)

_model_cache: dict | None = None


def load_model(path=config.MODEL_OUTPUT_PATH_V1) -> dict:
    """joblib 모델 번들({"model", "feature_columns"})을 로드하고 캐싱한다."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load(path)
    return _model_cache


def predict_score(
    plant_row: pd.Series,
    user_light: float,
    user_temp: float,
    user_humidity: float,
    user_cultivation_context: str,
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
    if any(pd.isna(plant_row[col]) for col in _PLANT_FEATURE_COLS if col != config.NORM_HUMIDITY_MAX):
        return None

    if is_gated(plant_row, user_temp, user_cultivation_context):
        return 0.0

    if model_bundle is None:
        model_bundle = load_model()
    model, feature_columns = model_bundle["model"], model_bundle["feature_columns"]

    row_df = pd.DataFrame([{
        "user_light": user_light,
        "user_temp": user_temp,
        "user_humidity": user_humidity,
        "user_cultivation_context": user_cultivation_context,
        **{col: plant_row[col] for col in _PLANT_FEATURE_COLS},
    }])
    features = build_features_v1(row_df).reindex(columns=feature_columns, fill_value=0)
    raw_score = model.predict(features)[0]
    # 학습 범위 밖 입력에서 트리 예측이 0~100을 살짝 벗어날 수 있어 안전하게 clip한다.
    return float(min(100.0, max(0.0, raw_score)))


def predict_all(
    plants_df: pd.DataFrame,
    user_light: float,
    user_temp: float,
    user_humidity: float,
    user_cultivation_context: str,
) -> pd.DataFrame:
    """전체 종의 점수를 계산해 (식물명, 학명, 점수) df를 점수 내림차순으로 반환.

    teacher_scoring.score_all_plants()와 동일한 인터페이스를 갖는다 — 다만 연속 점수는
    ML모델(v1)로 계산하고, 게이트만 룰로 강제한다는 점이 다르다.
    """
    model_bundle = load_model()
    names, sci_names, scores = [], [], []
    skipped = []
    for _, row in plants_df.iterrows():
        score = predict_score(
            row, user_light, user_temp, user_humidity, user_cultivation_context, model_bundle
        )
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
    result = predict_all(
        plants_df,
        user_light=15000,
        user_temp=25,
        user_humidity=55,
        user_cultivation_context=config.CULTIVATION_INDOOR,
    )
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
