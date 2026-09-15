"""5단계: 환경 + 식물 피처 -> 적합도 점수를 예측하는 공유 회귀모델 학습.

4단계에서 만든 synthetic_training_data.csv(룰 기반 티처 점수로 라벨링됨)를 학습 데이터로 쓴다.
목표는 "이미 정확한 규칙(teacher_scoring)을 ONNX로 배포 가능한 형태로 근사/압축"하는 것

트리 기반 모델을 쓰는 이유: 한계온도/재배구분 게이트 같은 하드컷 조건은 트리 분기 구조로 표현
서빙 단계에서는 이 게이트를 모델 예측에만 맡기지 않고 teacher_scoring.is_gated()로 한 번 더 강제(근사 오차가 남을 수 있어서)

v0 -> v2 피처
v0은 원본 값(환경값 + 식물 적정범위 + 생육형 그룹)만 피처로 사용.
v2는 적정범위까지의 마진/배율과 게이트 근접도를 파생 피처로 추가해, 트리가 규칙의 힌지 구조를
값 하나만 보고 바로 분기할 수 있게 한다.
(v1은 2차 데이터 + 구버전 채점 규칙으로 학습한 모델이라 3차 반영 시점에 v2로 대체됐다.
 models/recommendation_model_v1.* 는 비교용 기록으로만 남겨둔다.)

## 알고리즘: GBM으로 최종 확정
RandomizedSearchCV + 15-fold CV로 비교했을 때 RandomForest가 MAE는 더 낮았지만 (0.48 vs GBM 0.65),
joblib 파일이 575~687MB까지 커져 ONNX 변환 후에도 게임에 내장하기 비현실적이었다(GBM은 0.7MB).
그 정도 MAE 차이는 0~100점 추천 점수에서 체감 불가능한 수준이라 배포 크기를 우선해 GBM(튜닝된 하이퍼파라미터, GBR_TUNED_PARAMS)으로 확정
"""

import numpy as np
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
    # 기간 누적 입력 — 감점 채점에 쓰는 평균값
    "user_light_mean",
    "user_temp_mean",
    "user_humidity_mean",
    # 한계온도 게이트 판정에 쓰는 기간 극값. 평균과 별개로 넣어야 게이트가 학습된다
    # (같은 평균이라도 기간 내 변동폭에 따라 게이트 발동 여부가 갈린다).
    "user_temp_min",
    "user_temp_max",
    config.NORM_LIGHT_LUX_MIN,
    config.NORM_LIGHT_LUX_MAX,
    config.NORM_TEMP_OPTIMAL_MIN,
    config.NORM_TEMP_OPTIMAL_MAX,
    config.NORM_TEMP_LIMIT_LOWER,
    config.NORM_TEMP_LIMIT_UPPER,
    config.NORM_HUMIDITY_MIN,
    config.NORM_HUMIDITY_MAX,
    config.NORM_LIGHT_SATURATION_MAX,
    config.NORM_LIGHT_COMPENSATION_MIN,
)

# 생육형 그룹은 광량/온도/습도 가중치를 결정하므로 모델도 종의 그룹을 알아야 한다.
PLANT_GROUPS = tuple(config.GROUP_WEIGHTS.keys())

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
    for group in PLANT_GROUPS:
        features[f"plant_group_{group}"] = (df[config.NORM_PLANT_GROUP] == group).astype(int)

    return features


def _below_above(value: pd.Series, lo: pd.Series, hi: pd.Series) -> tuple[pd.Series, pd.Series]:
    """적정범위 [lo, hi] 대비 value의 마진. 범위 안이면 둘 다 0."""
    below = (lo - value).clip(lower=0)
    above = (value - hi).clip(lower=0)
    return below, above


def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """2차 피처: v0 원본 값에 마진/배율/게이트 근접도 파생 피처를 추가한다.

    teacher_scoring이 실제로 계산에 쓰는 값을 트리에 그대로 노출해, 규칙의 힌지 구조를
    값 하나만 보고 바로 분기할 수 있게 한다.

    광량은 절대 마진이 아니라 "적정범위 대비 배율"로 채점하므로(teacher_scoring 참고)
    배율 피처를 따로 만든다. 절대 마진만 주면 트리가 종별로 다른 임계값을 일일이
    학습해야 해서 근사 오차가 커진다.
    """
    features = build_features_v0(df)

    light_below, light_above = _below_above(
        df["user_light_mean"], df[config.NORM_LIGHT_LUX_MIN], df[config.NORM_LIGHT_LUX_MAX]
    )
    features["light_margin_below"] = light_below
    features["light_margin_above"] = light_above

    # 적정범위 대비 배율(범위 안이면 1). 0 나눗셈과 log(0)을 피하려고 하한을 둔다.
    user_light = df["user_light_mean"].clip(lower=1)
    ratio_below = (df[config.NORM_LIGHT_LUX_MIN] / user_light).clip(lower=1)
    ratio_above = (user_light / df[config.NORM_LIGHT_LUX_MAX].clip(lower=1)).clip(lower=1)
    features["light_ratio_below"] = ratio_below
    features["light_ratio_above"] = ratio_above
    # teacher_scoring은 배율을 로그 공간에서 보간하므로 로그값도 함께 준다.
    # 배율 원본만 주면 트리가 로그 곡선을 계단으로 흉내내야 해서, 어두운 환경처럼
    # 배율이 크게 벌어지는 구간(300lux에서 최대 100배)에서 근사 오차가 커진다.
    features["light_log_ratio_below"] = np.log(ratio_below)
    features["light_log_ratio_above"] = np.log(ratio_above)

    temp_below, temp_above = _below_above(
        df["user_temp_mean"], df[config.NORM_TEMP_OPTIMAL_MIN], df[config.NORM_TEMP_OPTIMAL_MAX]
    )
    features["temp_margin_below"] = temp_below
    features["temp_margin_above"] = temp_above

    humidity_max_filled = df[config.NORM_HUMIDITY_MAX].fillna(config.MODEL_HUMIDITY_MAX_FILL)
    humidity_below, humidity_above = _below_above(
        df["user_humidity_mean"], df[config.NORM_HUMIDITY_MIN], humidity_max_filled
    )
    features["humidity_margin_below"] = humidity_below
    features["humidity_margin_above"] = humidity_above

    # 적정범위 경계에서 한계값까지를 1로 놓은 상대 위치. teacher_scoring이 감점을 계산할 때
    # 실제로 쓰는 양이 이 비율이다(종마다 적정~한계 폭이 달라 절대 마진만으로는 표현이 안 된다).
    temp_lower_span = (df[config.NORM_TEMP_OPTIMAL_MIN] - df[config.NORM_TEMP_LIMIT_LOWER]).clip(lower=1e-6)
    temp_upper_span = (df[config.NORM_TEMP_LIMIT_UPPER] - df[config.NORM_TEMP_OPTIMAL_MAX]).clip(lower=1e-6)
    features["temp_below_position"] = (temp_below / temp_lower_span).clip(upper=2)
    features["temp_above_position"] = (temp_above / temp_upper_span).clip(upper=2)

    humidity_lower_span = (df[config.NORM_HUMIDITY_MIN] - config.SCORE_HUMIDITY_LEVEL2_LOWER).clip(lower=1e-6)
    humidity_upper_span = (
        humidity_max_filled.combine(
            pd.Series(config.SCORE_HUMIDITY_LEVEL2_UPPER, index=df.index), max
        )
        + config.SCORE_HUMIDITY_MIN_BAND
        - humidity_max_filled
    ).clip(lower=1e-6)
    features["humidity_below_position"] = (humidity_below / humidity_lower_span).clip(upper=2)
    features["humidity_above_position"] = (humidity_above / humidity_upper_span).clip(upper=2)

    # 한계온도 게이트 판정 경계: 음수면 게이트가 발동(생존 불가)하는 지점.
    # 3차 데이터부터 한계온도가 상·하한으로 분리돼 게이트도 양쪽 모두 확인한다.
    features["temp_limit_lower_margin"] = df["user_temp_min"] - df[config.NORM_TEMP_LIMIT_LOWER]
    features["temp_limit_upper_margin"] = df[config.NORM_TEMP_LIMIT_UPPER] - df["user_temp_max"]
    # 기간 내 온도 변동폭 — 평균이 같아도 변동이 크면 게이트에 걸릴 가능성이 높다.
    features["temp_window_spread"] = df["user_temp_max"] - df["user_temp_min"]

    # 재배구분 게이트 판정 결과: 혼합이거나 사용자 환경과 일치하면 1, 아니면 0.
    features["cultivation_match"] = (
        (df[config.NORM_CULTIVATION_TYPE] == config.CULTIVATION_MIXED)
        | (df[config.NORM_CULTIVATION_TYPE] == df["user_cultivation_context"])
    ).astype(int)

    # 그룹별 가중치 자체를 숫자로 노출한다. 원-핫만 주면 트리가 "그룹 -> 가중치" 대응을
    # 분기로 다시 학습해야 하는데, 이 값은 config에 이미 정해져 있는 상수다.
    weights = df[config.NORM_PLANT_GROUP].map(
        lambda g: config.GROUP_WEIGHTS.get(g, config.DEFAULT_GROUP_WEIGHTS)
    )
    features["weight_light"] = [w[0] for w in weights]
    features["weight_temp"] = [w[1] for w in weights]
    features["weight_humidity"] = [w[2] for w in weights]

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

    X_v2 = build_features_v2(df)
    model_v2, metrics_v2 = _train_and_evaluate(
        X_v2, y, df, "v2", lambda: GradientBoostingRegressor(**GBR_TUNED_PARAMS)
    )

    print(
        f"\nv0 -> v2: MAE {metrics_v0['mae']:.2f} -> {metrics_v2['mae']:.2f}"
        f" ({(metrics_v2['mae'] - metrics_v0['mae']) / metrics_v0['mae']:+.1%})"
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump({"model": model_v0, "feature_columns": list(X_v0.columns)}, config.MODEL_OUTPUT_PATH)
    dump({"model": model_v2, "feature_columns": list(X_v2.columns)}, config.MODEL_OUTPUT_PATH_V2)
    print(f"\n모델 저장 완료 -> {config.MODEL_OUTPUT_PATH}, {config.MODEL_OUTPUT_PATH_V2}")


if __name__ == "__main__":
    main()
