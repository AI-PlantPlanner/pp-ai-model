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
from src.env_profiles import load_profiles
from src.teacher_scoring import EnvironmentWindow, score_plant


def _window_from_row(env: dict) -> EnvironmentWindow:
    """generate_environments()가 만든 요약 행을 EnvironmentWindow로 되돌린다."""
    return EnvironmentWindow(
        light_mean=env["user_light_mean"],
        temp_mean=env["user_temp_mean"],
        humidity_mean=env["user_humidity_mean"],
        temp_min=env["user_temp_min"],
        temp_max=env["user_temp_max"],
        cultivation_context=env["user_cultivation_context"],
        n_samples=config.ENV_WINDOW_DAYS,
    )

PLANT_FEATURE_COLS = (
    config.COL_NAME,
    config.COL_SCIENTIFIC_NAME,
    config.NORM_LIGHT_LUX_MIN,
    config.NORM_LIGHT_LUX_MAX,
    config.NORM_TEMP_OPTIMAL_MIN,
    config.NORM_TEMP_OPTIMAL_MAX,
    config.NORM_TEMP_LIMIT_LOWER,
    config.NORM_TEMP_LIMIT_UPPER,
    config.NORM_HUMIDITY_MIN,
    config.NORM_HUMIDITY_MAX,
    config.NORM_CULTIVATION_TYPE,
    # 3차 데이터 추가분 — 광보상점/광포화점은 광량 ② 구간을 넓히는 데,
    # 생육형 그룹은 광량/온도/습도 가중치를 고르는 데 쓰인다(teacher_scoring 참고).
    config.NORM_LIGHT_SATURATION_MAX,
    config.NORM_LIGHT_COMPENSATION_MIN,
    config.NORM_PLANT_GROUP,
)


def _sample_environment_windows(
    n: int,
    light_range: tuple[float, float],
    temp_range: tuple[float, float],
    humidity_range: tuple[float, float],
    temp_spread_range: tuple[float, float],
    humidity_spread_range: tuple[float, float],
    light_log_spread_range: tuple[float, float],
    cultivation_context: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """기간 누적 환경(config.ENV_WINDOW_DAYS일치)을 n개 만들어 요약 통계량으로 돌려준다.

    서빙 입력이 기간 누적 배열이므로 학습 데이터도 같은 형태여야 한다. 기간 평균만 흔들면
    한계온도 게이트가 학습되지 않는다 — 같은 평균이라도 기간 내 변동폭에 따라 최저기온이
    한계온도를 넘느냐가 갈리기 때문이다. 그래서 평균과 변동폭을 각각 샘플링해
    일별 값을 만든 뒤 teacher_scoring과 동일한 방식으로 요약한다.

    광량만 로그 균등으로 뽑는다. 이유가 두 가지다.
    (1) 채점이 로그 스케일이다 — teacher_scoring은 광량을 적정범위 대비 "배율"로 채점한다.
    (2) 실제 분포가 로그에 가깝다 — 선형 균등으로 뽑으면 100~500lux(창에서 떨어진 거실
        안쪽, 실사용에서 가장 흔한 환경)가 전체 샘플의 1%밖에 안 잡혀서 그 구간의
        모델 근사 오차가 커진다.
    """
    days = config.ENV_WINDOW_DAYS
    log_low, log_high = np.log(light_range[0]), np.log(light_range[1])

    light_center = rng.uniform(log_low, log_high, size=n)
    temp_center = rng.uniform(*temp_range, size=n)
    humidity_center = rng.uniform(*humidity_range, size=n)
    light_spread = rng.uniform(*light_log_spread_range, size=n)
    temp_spread = rng.uniform(*temp_spread_range, size=n)
    humidity_spread = rng.uniform(*humidity_spread_range, size=n)

    # (n, days) 일별 값. 광량은 로그 공간에서 흔든 뒤 되돌린다.
    light_daily = np.exp(rng.normal(light_center[:, None], light_spread[:, None], size=(n, days)))
    temp_daily = rng.normal(temp_center[:, None], temp_spread[:, None], size=(n, days))
    humidity_daily = rng.normal(humidity_center[:, None], humidity_spread[:, None], size=(n, days))

    # 일별 값을 각 범위 안으로 자른다. SYNTHETIC_*_RANGE는 "그 환경에서 하루 값이 가질 수
    # 있는 범위"라는 뜻이다. 자르지 않으면 평균 주변으로 흔든 값이 범위를 크게 벗어나
    # 실외 최고 66℃, 최저 -35℃ 같은 한국에 존재하지 않는 창이 만들어지고,
    # 그런 환경에서는 대부분의 종이 게이트에 걸려 학습 데이터가 0점으로 쏠린다.
    light_daily = light_daily.clip(*light_range)
    temp_daily = temp_daily.clip(*temp_range)
    humidity_daily = humidity_daily.clip(*humidity_range)

    return pd.DataFrame(
        {
            "user_light_mean": light_daily.mean(axis=1),
            "user_temp_mean": temp_daily.mean(axis=1),
            "user_humidity_mean": humidity_daily.mean(axis=1),
            "user_temp_min": temp_daily.min(axis=1),
            "user_temp_max": temp_daily.max(axis=1),
            "user_cultivation_context": cultivation_context,
        }
    )


def _summarize_daily(
    light_daily: np.ndarray,
    temp_daily: np.ndarray,
    humidity_daily: np.ndarray,
    cultivation_context: str,
) -> dict:
    """일별 값 배열 (n, days)을 teacher_scoring과 동일한 통계량으로 요약한다."""
    return {
        "user_light_mean": light_daily.mean(axis=1),
        "user_temp_mean": temp_daily.mean(axis=1),
        "user_humidity_mean": humidity_daily.mean(axis=1),
        "user_temp_min": temp_daily.min(axis=1),
        "user_temp_max": temp_daily.max(axis=1),
        "user_cultivation_context": cultivation_context,
    }


def _sample_windows_from_profiles(
    profiles: list[dict],
    n: int,
    cultivation_context: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """LLM이 만든 환경 프로파일에서 기간 누적 환경을 n개 샘플링한다.

    균등 샘플링과의 차이는 "광량/온도/습도가 함께 움직인다"는 점이다. 남향 창가 여름은
    광량도 높고 온도도 높은데, 각 축을 독립적으로 균등 샘플링하면 그런 상관이 생기지 않아
    실재하지 않는 조합(어두운데 무더운 실내 등)에 학습 데이터가 낭비된다.
    프로파일을 먼저 고르고 그 안에서 흔들면 축 간 상관이 유지된다.

    프로파일은 weight(실사용 빈도)에 비례해 뽑는다 — 흔한 환경에 샘플이 더 가도록.
    """
    pool = [p for p in profiles if p["cultivation_context"] == cultivation_context]
    if not pool:
        raise ValueError(f"{cultivation_context} 프로파일이 없습니다.")

    weights = np.array([p["weight"] for p in pool], dtype=float)
    weights = weights / weights.sum()
    picked = rng.choice(len(pool), size=n, p=weights)

    days = config.ENV_WINDOW_DAYS
    light_center = np.log([pool[i]["light_lux_mean"] for i in picked])
    light_spread = np.array([pool[i]["light_log_spread"] for i in picked])
    temp_center = np.array([pool[i]["temp_mean"] for i in picked])
    temp_spread = np.array([pool[i]["temp_spread"] for i in picked])
    humidity_center = np.array([pool[i]["humidity_mean"] for i in picked])
    humidity_spread = np.array([pool[i]["humidity_spread"] for i in picked])

    light_daily = np.exp(rng.normal(light_center[:, None], light_spread[:, None], size=(n, days)))
    temp_daily = rng.normal(temp_center[:, None], temp_spread[:, None], size=(n, days))
    humidity_daily = rng.normal(humidity_center[:, None], humidity_spread[:, None], size=(n, days))

    # 물리적으로 불가능한 값만 잘라낸다. 프로파일 자체가 현실 범위를 담고 있으므로
    # 균등 샘플링 때처럼 config의 좁은 범위로 자르지 않는다(자르면 프로파일 의미가 사라진다).
    light_daily = light_daily.clip(1.0, 130_000.0)
    temp_daily = temp_daily.clip(-30.0, 45.0)
    humidity_daily = humidity_daily.clip(0.0, 100.0)

    frame = pd.DataFrame(
        _summarize_daily(light_daily, temp_daily, humidity_daily, cultivation_context)
    )
    frame["profile_name"] = [pool[i]["name"] for i in picked]
    return frame


def generate_environments(rng: np.random.Generator) -> pd.DataFrame:
    """실내/실외 각각의 현실적 범위 안에서 기간 누적 환경을 샘플링한다.

    LLM으로 만든 환경 프로파일(config.ENV_PROFILES_PATH)이 있으면 그것을 쓰고,
    없으면 config.SYNTHETIC_*_RANGE 기반 균등 샘플링으로 폴백한다.
    폴백 경로는 프로파일 도입 전후를 비교하는 베이스라인 역할도 한다.
    """
    profiles = load_profiles()
    if profiles:
        print(f"환경 프로파일 {len(profiles)}개 사용 -> {config.ENV_PROFILES_PATH.name}")
        return pd.concat(
            [
                _sample_windows_from_profiles(
                    profiles, config.SYNTHETIC_N_SAMPLES_INDOOR, config.CULTIVATION_INDOOR, rng
                ),
                _sample_windows_from_profiles(
                    profiles, config.SYNTHETIC_N_SAMPLES_OUTDOOR, config.CULTIVATION_OUTDOOR, rng
                ),
            ],
            ignore_index=True,
        )

    print(f"환경 프로파일이 없어 균등 샘플링으로 폴백합니다 ({config.ENV_PROFILES_PATH.name} 없음)")
    return _generate_environments_uniform(rng)


def _generate_environments_uniform(rng: np.random.Generator) -> pd.DataFrame:
    """프로파일 없이 config 범위에서 균등 샘플링하는 기존 경로(베이스라인)."""
    indoor = _sample_environment_windows(
        config.SYNTHETIC_N_SAMPLES_INDOOR,
        config.SYNTHETIC_INDOOR_LIGHT_LUX_RANGE,
        config.SYNTHETIC_INDOOR_TEMP_RANGE,
        config.SYNTHETIC_INDOOR_HUMIDITY_RANGE,
        config.SYNTHETIC_INDOOR_TEMP_SPREAD_RANGE,
        config.SYNTHETIC_INDOOR_HUMIDITY_SPREAD_RANGE,
        config.SYNTHETIC_INDOOR_LIGHT_LOG_SPREAD_RANGE,
        config.CULTIVATION_INDOOR,
        rng,
    )
    outdoor = _sample_environment_windows(
        config.SYNTHETIC_N_SAMPLES_OUTDOOR,
        config.SYNTHETIC_OUTDOOR_LIGHT_LUX_RANGE,
        config.SYNTHETIC_OUTDOOR_TEMP_RANGE,
        config.SYNTHETIC_OUTDOOR_HUMIDITY_RANGE,
        config.SYNTHETIC_OUTDOOR_TEMP_SPREAD_RANGE,
        config.SYNTHETIC_OUTDOOR_HUMIDITY_SPREAD_RANGE,
        config.SYNTHETIC_OUTDOOR_LIGHT_LOG_SPREAD_RANGE,
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
            score = score_plant(plant, _window_from_row(env))
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
