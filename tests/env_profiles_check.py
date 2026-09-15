"""env_profiles의 검증 로직과 프로파일 기반 샘플링을 확인하는 스크립트.

pytest 없이 `python -m tests.env_profiles_check` 로 실행한다.

이 검사가 중요한 이유: 환경 프로파일은 LLM이 만든다. LLM 생성 데이터는 모델 편향 때문에
실제 분포에서 체계적으로 벗어나는 것이 보고되어 있어서, 그대로 학습에 넣으면 안 된다.
env_profiles.validate_profiles()가 그 방어선인데, 방어선 자체가 동작하는지 확인하지 않으면
"검증했다"는 말이 의미가 없다. 그래서 일부러 틀린 프로파일을 넣어 걸리는지 본다.

아래 프로파일은 전부 검증 로직을 시험하기 위한 테스트 픽스처다 — LLM이 생성한 값이 아니고,
학습에 쓰이지도 않는다(실제 프로파일은 config.ENV_PROFILES_PATH에 저장된다).
"""

import numpy as np
import pandas as pd

from src import config
from src.env_profiles import validate_profiles
from src.synthetic_data import _sample_windows_from_profiles
from src.teacher_scoring import score_all_plants, summarize_environment

_failures: list[str] = []


def _check(passed: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}{f': {detail}' if detail else ''}")
    if not passed:
        _failures.append(label)


def _profile(name, context, season, placement, lux, temp, humidity, **overrides):
    base = {
        "name": name,
        "cultivation_context": context,
        "season": season,
        "placement": placement,
        "light_lux_mean": lux,
        "light_log_spread": 0.3,
        "temp_mean": temp,
        "temp_spread": 2.0 if context == config.CULTIVATION_INDOOR else 5.0,
        "humidity_mean": humidity,
        "humidity_spread": 5.0,
        "weight": 1.0,
        "rationale": "테스트 픽스처",
    }
    base.update(overrides)
    return base


def _valid_set() -> list[dict]:
    """검증을 통과해야 하는 최소 구성(실내/실외 + 4계절 + 광량 서열 정상)."""
    return [
        _profile("남향 창가 밀착, 봄", config.CULTIVATION_INDOOR, "봄", "남향 창가", 12_000, 20, 50),
        _profile("동향 창가 1m, 여름", config.CULTIVATION_INDOOR, "여름", "동향 창가", 6_000, 27, 65),
        _profile("북향 창가, 가을", config.CULTIVATION_INDOOR, "가을", "북향 창가", 1_500, 20, 55),
        _profile("창 없는 구석, 겨울", config.CULTIVATION_INDOOR, "겨울", "창 없는 구석", 250, 19, 40),
        _profile("마당 노지, 봄", config.CULTIVATION_OUTDOOR, "봄", "마당", 45_000, 13, 60),
        _profile("마당 노지, 여름", config.CULTIVATION_OUTDOOR, "여름", "마당", 80_000, 26, 80),
        _profile("마당 노지, 가을", config.CULTIVATION_OUTDOOR, "가을", "마당", 40_000, 14, 65),
        _profile("마당 노지, 겨울", config.CULTIVATION_OUTDOOR, "겨울", "마당", 25_000, -1, 55),
    ]


def check_valid_set_passes() -> None:
    print("\n[1] 정상 프로파일은 통과해야 한다")
    valid, issues = validate_profiles(_valid_set())
    _check(len(valid) == 8 and not issues, "정상 8개 통과", f"{len(valid)}개 통과 / 문제 {len(issues)}건 {issues}")


def check_catches_violations() -> None:
    print("\n[2] 일부러 틀린 프로파일을 잡아내야 한다")

    cases = [
        (
            "필드 누락",
            lambda ps: [{k: v for k, v in ps[0].items() if k != "temp_mean"}] + ps[1:],
            "필드 누락",
        ),
        (
            "물리 범위 이탈(광량 50만 lux)",
            lambda ps: [dict(ps[0], light_lux_mean=500_000)] + ps[1:],
            "범위 이탈",
        ),
        (
            "물리 범위 이탈(습도 -5%)",
            lambda ps: [dict(ps[0], humidity_mean=-5)] + ps[1:],
            "범위 이탈",
        ),
        (
            "광량 서열 모순(북향이 남향보다 밝음)",
            lambda ps: [dict(ps[2], light_lux_mean=99_000)] + ps[:2] + ps[3:],
            "광량 서열 모순",
        ),
        (
            "계절 기온 모순(실외 겨울이 여름보다 더움)",
            lambda ps: ps[:7] + [dict(ps[7], temp_mean=35)],
            "계절 기온 모순",
        ),
        (
            "기온 변동 모순(실내 변동이 실외보다 큼)",
            lambda ps: [dict(p, temp_spread=12.0) for p in ps[:4]] + ps[4:],
            "기온 변동 모순",
        ),
        (
            "커버리지 부족(실외 없음)",
            lambda ps: ps[:4],
            "커버리지 부족",
        ),
        (
            "커버리지 부족(겨울 없음)",
            lambda ps: [p for p in ps if p["season"] != "겨울"],
            "커버리지 부족",
        ),
    ]

    for label, mutate, expected in cases:
        _, issues = validate_profiles(mutate(_valid_set()))
        caught = any(expected in issue for issue in issues)
        _check(caught, label, f"검출된 문제: {issues if issues else '없음'}"[:110])


def check_profile_sampling() -> None:
    print("\n[3] 프로파일 기반 샘플링이 정상 범위의 창을 만들어야 한다")
    rng = np.random.default_rng(0)
    profiles = _valid_set()

    for context in (config.CULTIVATION_INDOOR, config.CULTIVATION_OUTDOOR):
        frame = _sample_windows_from_profiles(profiles, 200, context, rng)
        _check(
            len(frame) == 200 and frame["user_cultivation_context"].eq(context).all(),
            f"{context} 창 200개 생성",
        )
        _check(
            (frame["user_temp_min"] <= frame["user_temp_mean"]).all()
            and (frame["user_temp_mean"] <= frame["user_temp_max"]).all(),
            f"{context} 최저 <= 평균 <= 최고",
        )
        _check(
            frame["user_light_mean"].gt(0).all() and frame["user_humidity_mean"].between(0, 100).all(),
            f"{context} 광량 양수 / 습도 0~100%",
            f"광량 {frame.user_light_mean.min():.0f}~{frame.user_light_mean.max():,.0f}lux",
        )

    # 프로파일을 쓰면 축 간 상관이 생겨야 한다(균등 샘플링의 핵심 차이).
    outdoor = _sample_windows_from_profiles(profiles, 500, config.CULTIVATION_OUTDOOR, rng)
    corr = outdoor["user_light_mean"].corr(outdoor["user_temp_mean"])
    _check(
        corr > 0.3,
        "실외 광량-온도 상관이 생김(균등 샘플링은 0에 가까움)",
        f"상관계수 {corr:.2f}",
    )


def check_scoring_runs() -> None:
    print("\n[4] 프로파일에서 만든 창으로 채점이 동작해야 한다")
    plants = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    rng = np.random.default_rng(1)
    frame = _sample_windows_from_profiles(_valid_set(), 5, config.CULTIVATION_INDOOR, rng)
    row = frame.iloc[0]
    env = summarize_environment(
        [row.user_light_mean], [row.user_temp_min, row.user_temp_mean, row.user_temp_max],
        [row.user_humidity_mean], row.user_cultivation_context,
    )
    ranked = score_all_plants(plants, env)
    alive = ranked[ranked["점수"] > 0]
    _check(
        len(alive) > 0,
        "채점 결과가 나옴",
        f"프로파일 '{row.profile_name}' -> 추천 {len(alive)}종 / 1위 {alive.iloc[0]['식물명']}({alive.iloc[0]['점수']:.0f})",
    )


def main() -> None:
    check_valid_set_passes()
    check_catches_violations()
    check_profile_sampling()
    check_scoring_runs()

    print()
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        raise SystemExit(1)
    print("전체 통과")


if __name__ == "__main__":
    main()
