"""5단계: 환경 + 식물 피처 -> 적합도 점수를 예측하는 공유 회귀모델 학습.

4단계에서 만든 synthetic_training_data.csv(룰 기반 티처 점수로 라벨링된 6만 행)를 학습
데이터로 쓴다. 새 종이 추가될 계획이 거의 없다는 전제 하에, 목표는 "미지의 패턴 학습"이
아니라 "이미 정확한 규칙(teacher_scoring)을 ONNX로 배포 가능한 형태로 근사/압축"하는
것이다 — 그래서 검증도 종 홀드아웃보다 100종 전체에 대한 예측 정확도를 우선한다.

트리 기반(GradientBoostingRegressor)을 쓰는 이유: 한계온도/재배구분 게이트처럼 "정도
차이가 아니라 가능/불가능"인 하드컷 조건은 선형/단순 모델보다 트리 분기 구조로 훨씬 잘
표현된다.

## v0 -> v1
v0(1차)은 원본 값(사용자 환경값, 식물 적정범위)만 피처로 썼다. MAE 6.88점으로 완전히
나쁘진 않지만, teacher_scoring 공식 자체가 "적정범위에서 얼마나 벗어났는지(마진)"로 감점을
계산하는데 그 뺄셈을 트리가 원본 값들로부터 스스로 근사해야 하는 구조라 오차가 남았다.
v1(2차)은 그 마진(적정범위 하한/상한까지 거리)과 게이트 근접도를 파생 피처로 직접 추가해서,
트리가 규칙의 힌지(꺾이는 지점) 구조를 값 하나만 보고 바로 분기할 수 있게 했다.
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

GBR_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=config.MODEL_RANDOM_SEED)


def build_features_v0(df: pd.DataFrame) -> pd.DataFrame:
    """1차 피처: 원본 값(사용자 환경값 + 식물 적정범위)만 그대로 넘긴다.

    - humidity_max NaN(개방형 범위)은 물리적 상한(100%)으로 채운다.
    - 재배구분(식물의 cultivation_type, 사용자의 user_cultivation_context)은 원-핫
      인코딩한다 — 트리 모델이 재배구분 게이트를 스스로 분기로 학습하도록 값 자체는
      가공하지 않고 그대로 넘긴다.
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

    teacher_scoring의 _linear_range_score()가 실제로 계산에 쓰는 값(하한/상한까지의
    거리)과 두 게이트(한계온도, 재배구분)의 판정 경계값을 피처로 직접 노출한다.
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


def _train_and_evaluate(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, label: str) -> tuple[GradientBoostingRegressor, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.MODEL_TEST_SIZE, random_state=config.MODEL_RANDOM_SEED
    )

    model = GradientBoostingRegressor(**GBR_PARAMS)
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
    final_model = GradientBoostingRegressor(**GBR_PARAMS)
    final_model.fit(X, y)

    return final_model, {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    df = pd.read_csv(config.SYNTHETIC_OUTPUT_PATH)
    y = df["score"]

    X_v0 = build_features_v0(df)
    model_v0, metrics_v0 = _train_and_evaluate(X_v0, y, df, "v0")

    X_v1 = build_features_v1(df)
    model_v1, metrics_v1 = _train_and_evaluate(X_v1, y, df, "v1")

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
