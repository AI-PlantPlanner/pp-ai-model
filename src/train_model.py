"""5단계: 환경 + 식물 피처 -> 적합도 점수를 예측하는 공유 회귀모델 학습.

4단계에서 만든 synthetic_training_data.csv(룰 기반 티처 점수로 라벨링됨)를 학습 데이터로 쓴다.
목표는 "이미 정확한 규칙(teacher_scoring)을 ONNX로 배포 가능한 형태로 근사/압축"하는 것

트리 기반 모델을 쓰는 이유: 한계온도/재배구분 게이트 같은 하드컷 조건은 트리 분기 구조로 표현
서빙 단계에서는 이 게이트를 모델 예측에만 맡기지 않고 teacher_scoring.is_gated()로 한 번 더 강제(근사 오차가 남을 수 있어서)

v0 -> v1 피처
v0은 원본 값(환경값 + 식물 적정범위)만 피처로 사용(MAE 6.88).
v1은 적정범위까지의 마진과 게이트 근접도를 파생 피처로 추가해, 트리가 규칙의 힌지 구조를 값 하나만 보고 바로 분기할 수 있게함

## 알고리즘: GBM으로 최종 확정
RandomizedSearchCV + 15-fold CV로 비교했을 때 RandomForest가 MAE는 더 낮았지만 (0.48 vs GBM 0.65),
joblib 파일이 575~687MB까지 커져 ONNX 변환 후에도 게임에 내장하기 비현실적이었다(GBM은 0.7MB).
그 정도 MAE 차이는 0~100점 추천 점수에서 체감 불가능한 수준이라 배포 크기를 우선해 GBM(튜닝된 하이퍼파라미터, GBR_TUNED_PARAMS)으로 확정
"""

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src import config

CULTIVATION_CATEGORIES = (
    config.CULTIVATION_INDOOR,
    config.CULTIVATION_OUTDOOR,
    config.CULTIVATION_MIXED,
)

NUMERIC_FEATURE_COLS = (
    "user_light",
    "user_temp",
    "user_humidity",
    config.NORM_LIGHT_LUX_MIN,
    config.NORM_LIGHT_LUX_MAX,
    config.NORM_TEMP_OPTIMAL_MIN,
    config.NORM_TEMP_OPTIMAL_MAX,
    config.NORM_TEMP_LIMIT_MIN,
    config.NORM_TEMP_LIMIT_MAX,
    config.NORM_HUMIDITY_MIN,
    config.NORM_HUMIDITY_MAX,
)

# v0(비교용 베이스라인) — 탐색 없이 수동 선택한 값 그대로 유지.
GBR_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=config.MODEL_RANDOM_SEED)

# v1(프로덕션) — RandomizedSearchCV + CV 검증으로 찾은 하이퍼파라미터(모듈 docstring 참고).
GBR_TUNED_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=10,
    subsample=0.85,
    random_state=config.MODEL_RANDOM_SEED,
)


def build_features_v0(df: pd.DataFrame) -> pd.DataFrame:
    """1차 피처: 원본 값(사용자 환경값 + 식물 적정범위)만 그대로 넘긴다.

    - humidity_max NaN(개방형 범위)은 물리적 상한(100%)으로 채운다.
    - 재배구분(식물의 cultivation_type, 사용자의 user_cultivation_context)은 원-핫 인코딩한다
    """
    features = df[list(NUMERIC_FEATURE_COLS)].copy()
    features[config.NORM_HUMIDITY_MAX] = features[config.NORM_HUMIDITY_MAX].fillna(
        config.MODEL_HUMIDITY_MAX_FILL
    )

    for category in CULTIVATION_CATEGORIES:
        features[f"cultivation_type_{category}"] = (
            df[config.NORM_CULTIVATION_TYPE] == category
        ).astype(int)
    for category in (config.CULTIVATION_INDOOR, config.CULTIVATION_OUTDOOR):
        features[f"user_context_{category}"] = (
            df["user_cultivation_context"] == category
        ).astype(int)

    return features


def _below_above(value: pd.Series, lo: pd.Series, hi: pd.Series) -> tuple[pd.Series, pd.Series]:
    """적정범위 [lo, hi] 대비 value의 마진. 범위 안이면 둘 다 0."""
    below = (lo - value).clip(lower=0)
    above = (value - hi).clip(lower=0)
    return below, above


def build_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """2차 피처: v0 원본 값에 마진/게이트 근접도 파생 피처를 추가한다.

    teacher_scoring의 _linear_range_score()가 실제로 계산에 쓰는 값(하한/상한까지의 거리)과 두 게이트(한계온도, 재배구분)의 판정 경계값을 피처로 직접 노출
    """
    features = build_features_v0(df)

    light_below, light_above = _below_above(
        df["user_light"], df[config.NORM_LIGHT_LUX_MIN], df[config.NORM_LIGHT_LUX_MAX]
    )
    features["light_margin_below"] = light_below
    features["light_margin_above"] = light_above

    temp_below, temp_above = _below_above(
        df["user_temp"], df[config.NORM_TEMP_OPTIMAL_MIN], df[config.NORM_TEMP_OPTIMAL_MAX]
    )
    features["temp_margin_below"] = temp_below
    features["temp_margin_above"] = temp_above

    humidity_max_filled = df[config.NORM_HUMIDITY_MAX].fillna(config.MODEL_HUMIDITY_MAX_FILL)
    humidity_below, humidity_above = _below_above(
        df["user_humidity"], df[config.NORM_HUMIDITY_MIN], humidity_max_filled
    )
    features["humidity_margin_below"] = humidity_below
    features["humidity_margin_above"] = humidity_above

    # 한계온도 게이트 판정 경계: 음수면 게이트가 발동(생존 불가)하는 지점.
    features["temp_limit_margin"] = df["user_temp"] - df[config.NORM_TEMP_LIMIT_MAX]

    # 재배구분 게이트 판정 결과: 혼합이거나 사용자 환경과 일치하면 1, 아니면 0.
    features["cultivation_match"] = (
        (df[config.NORM_CULTIVATION_TYPE] == config.CULTIVATION_MIXED)
        | (df[config.NORM_CULTIVATION_TYPE] == df["user_cultivation_context"])
    ).astype(int)

    return features


def _train_and_evaluate(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, label: str, model_factory) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.MODEL_TEST_SIZE, random_state=config.MODEL_RANDOM_SEED
    )

    model = model_factory()
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = mean_squared_error(y_test, pred_test) ** 0.5
    r2 = r2_score(y_test, pred_test)
    print(f"[{label}] 테스트셋({len(X_test)}행) 평가: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

    test_df = df.loc[X_test.index].copy()
    test_df["pred"] = pred_test
    test_df["abs_error"] = (test_df["pred"] - test_df["score"]).abs()
    per_species = test_df.groupby(config.COL_NAME)["abs_error"].mean().sort_values(ascending=False)
    print(f"[{label}] 종별 평균 절대오차 상위 3개: {per_species.head(3).round(2).to_dict()}")
    print(f"[{label}] 종별 평균 절대오차 하위 3개: {per_species.tail(3).round(2).to_dict()}")

    # 테스트셋으로 검증이 끝났으니, 배포용 최종 모델은 전체 데이터로 다시 학습한다.
    final_model = model_factory()
    final_model.fit(X, y)

    return final_model, {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    df = pd.read_csv(config.SYNTHETIC_OUTPUT_PATH)
    y = df["score"]

    X_v0 = build_features_v0(df)
    model_v0, metrics_v0 = _train_and_evaluate(
        X_v0, y, df, "v0", lambda: GradientBoostingRegressor(**GBR_PARAMS)
    )

    X_v1 = build_features_v1(df)
    model_v1, metrics_v1 = _train_and_evaluate(
        X_v1, y, df, "v1", lambda: GradientBoostingRegressor(**GBR_TUNED_PARAMS)
    )

    print(
        f"\nv0 -> v1: MAE {metrics_v0['mae']:.2f} -> {metrics_v1['mae']:.2f}"
        f" ({(metrics_v1['mae'] - metrics_v0['mae']) / metrics_v0['mae']:+.1%})"
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump({"model": model_v0, "feature_columns": list(X_v0.columns)}, config.MODEL_OUTPUT_PATH)
    dump({"model": model_v1, "feature_columns": list(X_v1.columns)}, config.MODEL_OUTPUT_PATH_V1)
    print(f"\n모델 저장 완료 -> {config.MODEL_OUTPUT_PATH}, {config.MODEL_OUTPUT_PATH_V1}")


if __name__ == "__main__":
    main()
