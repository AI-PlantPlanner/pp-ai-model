"""4단계: 합성 방 환경 데이터 생성 + 룰 기반 티처 스코어링으로 자동 라벨링.

실사용자 데이터가 없는 지금 단계에서, 광량/온도/습도/실내·실외 조합을 현실적인 범위 안에서
대량으로 샘플링하고 teacher_scoring으로 정답 점수를 매겨 학습용 테이블을 만든다.
5단계(모델 학습)에서 이 테이블로 "환경 + 식물 피처 -> 점수"를 예측하는 공유 회귀모델을 학습한다.

샘플링 범위 근거는 config.py의 SYNTHETIC_* 상수 주석 참고 — 진단 모델(사진/센서 -> 환경값추정)
스펙이 확정되면 실제 입력 분포에 맞춰 다시 조정해야 하는 잠정치다.
"""

import numpy as np
import pandas as pd

from src import config
from src.teacher_scoring import score_plant

PLANT_FEATURE_COLS = (
    config.COL_NAME,
    config.COL_SCIENTIFIC_NAME,
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


def _sample_environments(
    n: int,
    light_range: tuple[float, float],
    temp_range: tuple[float, float],
    humidity_range: tuple[float, float],
    cultivation_context: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_light": rng.uniform(*light_range, size=n),
            "user_temp": rng.uniform(*temp_range, size=n),
            "user_humidity": rng.uniform(*humidity_range, size=n),
            "user_cultivation_context": cultivation_context,
        }
    )


def generate_environments(rng: np.random.Generator) -> pd.DataFrame:
    """실내/실외 각각의 현실적 범위 안에서 환경 조합을 균일 샘플링한다."""
    indoor = _sample_environments(
        config.SYNTHETIC_N_SAMPLES_INDOOR,
        config.SYNTHETIC_INDOOR_LIGHT_LUX_RANGE,
        config.SYNTHETIC_INDOOR_TEMP_RANGE,
        config.SYNTHETIC_INDOOR_HUMIDITY_RANGE,
        config.CULTIVATION_INDOOR,
        rng,
    )
    outdoor = _sample_environments(
        config.SYNTHETIC_N_SAMPLES_OUTDOOR,
        config.SYNTHETIC_OUTDOOR_LIGHT_LUX_RANGE,
        config.SYNTHETIC_OUTDOOR_TEMP_RANGE,
        config.SYNTHETIC_OUTDOOR_HUMIDITY_RANGE,
        config.CULTIVATION_OUTDOOR,
        rng,
    )
    return pd.concat([indoor, outdoor], ignore_index=True)


def build_training_table(plants_df: pd.DataFrame, environments_df: pd.DataFrame) -> pd.DataFrame:
    """환경 샘플 x 전체 종 조합마다 티처 점수를 매겨 학습 테이블을 만든다.

    score_plant()가 None을 반환하는 조합(결측 컬럼 또는 재배구분 미상)은 행 자체를 만들지
    않는다 — 임의 추정 라벨을 학습 데이터에 섞지 않기 위함(teacher_scoring.py와 동일한 방침).
    """
    plant_records = plants_df.to_dict("records")
    env_records = environments_df.to_dict("records")

    rows = []
    for env in env_records:
        for plant in plant_records:
            score = score_plant(
                plant,
                env["user_light"],
                env["user_temp"],
                env["user_humidity"],
                env["user_cultivation_context"],
            )
            if score is None:
                continue
            row = {col: plant[col] for col in PLANT_FEATURE_COLS}
            row.update(env)
            row["score"] = round(score, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(config.SYNTHETIC_RANDOM_SEED)
    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    environments_df = generate_environments(rng)
    training_df = build_training_table(plants_df, environments_df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(config.SYNTHETIC_OUTPUT_PATH, index=False)

    total_possible = len(environments_df) * len(plants_df)
    excluded = total_possible - len(training_df)
    zero_ratio = (training_df["score"] == 0).mean()
    print(f"합성 데이터 생성 완료: {len(training_df)}행 -> {config.SYNTHETIC_OUTPUT_PATH}")
    print(
        f"(환경 {len(environments_df)}개 x 식물 {len(plants_df)}종 = 최대 {total_possible}행 중 "
        f"{excluded}행은 결측/재배구분 미상으로 제외)"
    )
    print(f"0점 비율: {zero_ratio:.1%}")
    print(training_df["score"].describe())


if __name__ == "__main__":
    main()
