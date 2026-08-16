"""5단계: 환경 + 식물 피처 -> 적합도 점수를 예측하는 공유 회귀모델 학습.

4단계에서 만든 synthetic_training_data.csv(룰 기반 티처 점수로 라벨링됨)를 학습 데이터로
쓴다. 새 종이 추가될 계획이 거의 없다는 전제 하에, 목표는 "미지의 패턴 학습"이 아니라
"이미 정확한 규칙(teacher_scoring)을 ONNX로 배포 가능한 형태로 근사/압축"하는 것이다.

트리 기반 모델을 쓰는 이유: 한계온도/재배구분 게이트처럼 "정도 차이가 아니라
가능/불가능"인 하드컷 조건은 선형/단순 모델보다 트리 분기 구조로 훨씬 잘 표현된다.
(다만 실제 서빙에서는 이 게이트를 모델 예측에만 맡기지 않고 predict.py에서
teacher_scoring.is_gated()로 한 번 더 강제한다 — 근사 오차가 남을 수 있어서다.)

## v0 -> v1 피처
v0(1차)은 원본 값(사용자 환경값, 식물 적정범위)만 피처로 썼다. MAE 6.88점으로 완전히
나쁘진 않지만, teacher_scoring 공식 자체가 "적정범위에서 얼마나 벗어났는지(마진)"로 감점을
계산하는데 그 뺄셈을 트리가 원본 값들로부터 스스로 근사해야 하는 구조라 오차가 남았다.
v1(2차)은 그 마진(적정범위 하한/상한까지 거리)과 게이트 근접도를 파생 피처로 직접 추가해서,
트리가 규칙의 힌지(꺾이는 지점) 구조를 값 하나만 보고 바로 분기할 수 있게 했다.

## v1 알고리즘: GradientBoosting -> RandomForest -> GradientBoosting(최종)
처음엔 v0/v1 둘 다 GradientBoostingRegressor(하이퍼파라미터는 탐색 없이 수동 선택,
n_estimators=300/max_depth=4/learning_rate=0.05)를 썼다. 이후 RandomizedSearchCV로
GBM/RF 각각 하이퍼파라미터를 탐색하고(15개 조합 x cv=3), 최적 조합끼리 다음 세 가지로
교차검증했다:
  1) 일반 5-fold CV(행 단위, 서로 다른 시드 2개로 fold 구성을 바꿔 재확인) - 10개 fold
  2) 종 단위 GroupKFold 5-fold(신규 종 추가 시나리오 시뮬레이션) - 5개 fold
총 15개 fold 전부에서 RF가 GBM보다 MAE가 낮아(표준편차도 작아 우연한 split 효과가 아님)
한때 v1을 RF로 교체했었다.

그런데 RF(max_depth=None, min_samples_leaf=1, n_estimators=400)로 저장한 joblib
파일이 **575~687MB**까지 커졌다 — 트리를 끝까지 다 키우다 보니(리프 1개 샘플까지 분기)
트리 400개의 노드 수가 폭증한 것. 같은 조건의 GBM은 트리 깊이를 6으로 제한해서
**0.7MB**(800배 이상 작음)였다. min_samples_leaf를 올려 크기를 줄여봐도(leaf=10 ->
82MB) GBM보다 여전히 100배 이상 크면서 정확도는 오히려 GBM보다 나빴다(MAE 0.74 > GBM
0.65~0.79 근처). ONNX로 변환해 게임/비-Python 런타임에 배포하는 게 목표인데, 이 정도
크기의 트리 앙상블은 ONNX로 변환해도 비슷하게 커서 게임에 내장하기 비현실적이다.
반면 RF가 이긴 정확도 차이(MAE 0.48 vs GBM 0.65~0.79)는 0~100점 추천 점수에서 사용자가
체감할 수 없는 수준이라, **최종적으로 v1은 다시 GBM(튜닝된 하이퍼파라미터)으로 되돌렸다.**
RF 실험 자체(교차검증 결과, 크기 문제)는 이후 재검토할 때 참고할 수 있도록 이 문단에
기록만 남겨둔다. v0은 "왜 마진 피처가 필요한지" 보여주는 비교용 베이스라인이라 튜닝하지
않고 원래 GBM(수동 설정) 그대로 둔다.

## 과적합 갭(train/test MAE 차이) — RF로 교체했을 때 발견했던 이슈, GBM으로 되돌리며 자연 해소
RF는 GBM보다 이 갭이 컸다(GBM 0.08 vs RF 0.40, 비율로는 5배). max_depth/min_samples_leaf를
낮춰 갭을 줄일 수 있는지 스캔해봤는데, 갭을 GBM 수준까지 줄이려면 정확도가 튜닝 전
GBM보다도 나빠져서 "갭이 작으면서 정확도도 높은" RF 조합은 없었다. 최종적으로 GBM으로
되돌리면서 이 갭 자체가 원래도 작았던 GBM 특성으로 회귀했으므로 별도 조치가 필요 없다.
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
# RF도 같은 방식으로 찾은 조합이 CV 성능은 더 좋았지만 joblib/ONNX 크기가 배포 불가능한
# 수준이라 최종적으로 GBM으로 되돌렸다.
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
