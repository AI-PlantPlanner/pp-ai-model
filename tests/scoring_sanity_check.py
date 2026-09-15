"""채점 규칙이 상식에 맞는지 자동 검증하는 회귀 스크립트.

pytest 없이 `python -m tests.scoring_sanity_check` 로 실행한다.

3차 데이터 반영 과정에서 실제로 발생했던 결함들을 그대로 검사 항목으로 만들었다.
config.py의 SCORE_* 상수를 바꾸면(원예팀 답변 반영 등) 이 스크립트를 다시 돌려
아래 성질이 깨지지 않았는지 확인한다.

검사 항목
  1. 순위 방향   — 강한 빛에서는 양지식물이, 어두운 곳에서는 음지식물이 위로 와야 한다.
                  (광량 점수가 전 종 0이 되면 그룹 가중치만 남아 순위가 거꾸로 뒤집혔던 적이 있다)
  2. 절벽 없음   — 입력값을 조금 움직였을 때 점수가 급락하면 안 된다.
                  (게이트는 예외 — 한계온도/재배구분은 원예팀이 0점으로 확정한 하드컷이다)
  3. 배율 감점   — 요구 광량이 큰 종은 어두운 환경에서 광량 점수가 0에 가까워야 한다.
  4. 습도 예외   — 적정 습도 상한이 80% 이상인 종(수국 등)이 적정범위 바로 밖에서 급락하면 안 된다.
  5. 변별력      — 한 시나리오에서 종끼리 점수가 충분히 갈려야 순위가 의미를 갖는다.
                  (전 종이 같은 점수로 뭉쳐 순위가 사라지는 붕괴를 잡는 것이 목적이다)
  6. 누적 게이트   — 기간 누적 입력에서 한계온도 게이트가 평균이 아니라 극값으로 걸려야 한다.
                  (평균으로 판정하면 겨울 한파가 묻혀 겨울에 죽을 식물이 추천된다)
  7. 룰-모델 일치 — 서빙용 ML 모델(v2)이 룰 점수를 충분히 근사해야 한다.
"""

import numpy as np
import pandas as pd

from src import config
from src.predict import predict_all
from src.teacher_scoring import (
    humidity_score,
    light_score,
    score_all_plants,
    summarize_environment,
    temp_score,
)

def _env(name, light, temp, humidity, context, temp_low=None, temp_high=None):
    """기간 누적 환경을 만든다. temp_low/high를 주면 기간 내 온도 변동이 있는 창을 흉내낸다.

    평균은 그대로 두고 극값만 벌린 창을 만들기 위해, 평균/최저/최고를 갖는 3개짜리
    관측 배열을 구성한다(평균이 유지되도록 최저·최고를 대칭으로 잡는다).
    """
    if temp_low is None and temp_high is None:
        temps = temp
    else:
        low = temp if temp_low is None else temp_low
        high = temp if temp_high is None else temp_high
        temps = [low, temp, high]
    return name, summarize_environment(light, temps, humidity, context)


# 실사용에서 자주 나오는 환경 + 과거에 문제를 드러냈던 조건.
#
# 주의: 입력이 단일 시점에서 기간 누적으로 바뀌면서 시나리오 값의 의미도 바뀌었다.
# 여기 적힌 광량/온도/습도는 "그 순간의 값"이 아니라 "기간(config.ENV_WINDOW_DAYS일) 평균"이다.
# 그래서 예전에 쓰던 "한여름 100,000lux" 같은 값은 더 이상 쓰지 않는다 —
# 3개월 평균이 100,000lux라는 건 그 기간 내내 정오 직사광이었다는 뜻이라 실재하지 않고,
# 실제로 학습 데이터 16만 행 중 그런 창이 0건이라 모델 오차만 크게 나온다.
SCENARIOS = [
    _env("실내 거실 안쪽", 300, 22, 50, config.CULTIVATION_INDOOR),
    _env("실내 밝은 창가", 3_000, 22, 50, config.CULTIVATION_INDOOR),
    _env("실내 매우 밝은 창가", 10_000, 24, 55, config.CULTIVATION_INDOOR),
    _env("실외 봄가을", 8_000, 15, 60, config.CULTIVATION_OUTDOOR, temp_low=2, temp_high=26),
    _env("실외 여름 강광", 60_000, 25, 70, config.CULTIVATION_OUTDOOR, temp_low=17, temp_high=33),
    _env("실외 한여름 고온다습", 30_000, 28, 82, config.CULTIVATION_OUTDOOR, temp_low=21, temp_high=35),
    _env("실외 한겨울", 20_000, -5, 45, config.CULTIVATION_OUTDOOR, temp_low=-10, temp_high=3),
    # 누적 입력의 핵심 케이스: 평균은 온화한데 기간 중 한파가 있었던 창.
    # 평균만 보면 대부분 통과하지만 최저기온으로 게이트를 걸면 크게 줄어야 한다.
    _env("실외 6개월(한파 포함)", 20_000, 6, 55, config.CULTIVATION_OUTDOOR, temp_low=-12, temp_high=24),
]

# 절벽으로 볼 낙차 기준(점). 게이트가 아닌 구간에서 이만큼 떨어지면 실패로 본다.
MAX_ADJACENT_DROP = 15.0
# 이 검사가 실제로 잡아야 하는 실패는 "전 종이 같은 점수를 받아 순위가 사라지는" 붕괴다.
# (실제로 있었던 일이다 — 광량 점수가 전 종 0이 되자 그룹 가중치만 남아 67종이 몇 개 값으로 뭉쳤다.)
# 서로 다른 점수 개수만 보면 적정범위가 똑같은 종들이 동점이 되는 정상 상황까지 걸리고,
# 추천 가능 종 수 대비 비율로 보면 게이트로 후보가 줄어든 시나리오가 걸린다.
# 그래서 "고유 점수 개수"와 "점수가 실제로 벌어진 폭"을 함께 본다.
#
# 상위권이 좁게 몰리는 것 자체는 실패로 보지 않는다. 게이트가 많이 걸리는 시나리오(한겨울 등)에서
# 살아남은 종은 전부 내한성 종이라 실제로 적합도가 비슷하다 — 상위 10종이 3점 안에 몰리는 건
# "이 열 종은 비슷하게 적합하다"는 정직한 표현이지 채점 오류가 아니다.
MIN_DISTINCT_SCORES = 10
MIN_SCORE_SPREAD = 10.0

_failures: list[str] = []


def _check(passed: bool, label: str, detail: str) -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}: {detail}")
    if not passed:
        _failures.append(label)


def _row(plants: pd.DataFrame, name: str) -> pd.Series:
    return plants[plants[config.COL_NAME] == name].iloc[0]


def check_ranking_direction(plants: pd.DataFrame) -> None:
    """강한 빛에서 양지식물이, 어두운 곳에서 음지식물이 위로 오는지."""
    print("\n[1] 순위 방향")

    ranked = score_all_plants(
        plants, summarize_environment(80_000, 25, 60, config.CULTIVATION_OUTDOOR)
    )
    ranked = ranked[ranked["점수"] > 0]
    lux_max = plants.set_index(config.COL_NAME)[config.NORM_LIGHT_LUX_MAX]
    top_lux = lux_max.reindex(ranked.head(10)["식물명"]).mean()
    bottom_lux = lux_max.reindex(ranked.tail(10)["식물명"]).mean()
    _check(
        top_lux > bottom_lux,
        "강한 빛(80,000lux)에서 요구 광량이 큰 종이 상위",
        f"상위10 평균 적정광량 상한 {top_lux:,.0f}lux vs 하위10 {bottom_lux:,.0f}lux",
    )

    ranked = score_all_plants(
        plants, summarize_environment(300, 22, 50, config.CULTIVATION_INDOOR)
    )
    ranked = ranked[ranked["점수"] > 0]
    lux_min = plants.set_index(config.COL_NAME)[config.NORM_LIGHT_LUX_MIN]
    top_lux_min = lux_min.reindex(ranked.head(10)["식물명"]).mean()
    bottom_lux_min = lux_min.reindex(ranked.tail(10)["식물명"]).mean()
    _check(
        top_lux_min < bottom_lux_min,
        "어두운 실내(300lux)에서 요구 광량이 작은 종이 상위",
        f"상위10 평균 적정광량 하한 {top_lux_min:,.0f}lux vs 하위10 {bottom_lux_min:,.0f}lux",
    )


def check_no_cliffs(plants: pd.DataFrame) -> None:
    """게이트가 아닌 구간에서 점수가 급락하지 않는지."""
    print("\n[2] 절벽 없음")

    light_grid = np.geomspace(50, 120_000, 400)
    humidity_grid = np.arange(5, 100.5, 0.5)

    for label, grid, fn in [
        ("광량", light_grid, light_score),
        ("습도", humidity_grid, humidity_score),
    ]:
        worst_drop, worst_name = 0.0, ""
        for _, row in plants.iterrows():
            values = [fn(row, v) for v in grid]
            drop = float(np.abs(np.diff(values)).max())
            if drop > worst_drop:
                worst_drop, worst_name = drop, row[config.COL_NAME]
        _check(
            worst_drop < MAX_ADJACENT_DROP,
            f"{label} 점수 연속성",
            f"최대 낙차 {worst_drop:.1f}점 ({worst_name})",
        )

    # 온도는 한계온도(게이트)에서 떨어지는 것이 정상이므로, 한계온도 안쪽만 검사한다.
    worst_drop, worst_name = 0.0, ""
    for _, row in plants.iterrows():
        lower = row[config.NORM_TEMP_LIMIT_LOWER]
        upper = row[config.NORM_TEMP_LIMIT_UPPER]
        grid = np.arange(lower, upper + 0.25, 0.25)
        values = [temp_score(row, v) for v in grid]
        drop = float(np.abs(np.diff(values)).max())
        if drop > worst_drop:
            worst_drop, worst_name = drop, row[config.COL_NAME]
    _check(
        worst_drop < MAX_ADJACENT_DROP,
        "온도 점수 연속성(한계온도 안쪽)",
        f"최대 낙차 {worst_drop:.1f}점 ({worst_name})",
    )


def check_ratio_penalty(plants: pd.DataFrame) -> None:
    """요구 광량이 큰 종이 어두운 환경에서 0점에 가까운지."""
    print("\n[3] 배율 감점")
    for name in ("제라늄", "프리지아"):
        row = _row(plants, name)
        score = light_score(row, 300)
        _check(
            score < 5,
            f"{name}(적정 {row[config.COL_LIGHT_LUX]}) 300lux 광량 점수",
            f"{score:.1f}점",
        )


def check_humidity_exception(plants: pd.DataFrame) -> None:
    """적정 습도 상한이 80% 이상인 종이 적정범위 바로 밖에서 급락하지 않는지."""
    print("\n[4] 습도 예외 처리")
    row = _row(plants, "수국")
    just_outside = humidity_score(row, row[config.NORM_HUMIDITY_MAX] + 1)
    _check(
        just_outside > 80,
        f"수국(적정 {row[config.COL_HUMIDITY]}) 적정 상한 +1%p 습도 점수",
        f"{just_outside:.1f}점",
    )


def check_discrimination(plants: pd.DataFrame) -> None:
    """시나리오마다 종별 점수가 충분히 갈리는지."""
    print("\n[5] 변별력")
    for name, env in SCENARIOS:
        ranked = score_all_plants(plants, env)
        alive = ranked[ranked["점수"] > 0]
        distinct = alive["점수"].nunique()
        spread = alive["점수"].max() - alive["점수"].min()
        _check(
            distinct >= MIN_DISTINCT_SCORES and spread >= MIN_SCORE_SPREAD,
            f"{name}",
            f"추천 가능 {len(alive)}종 / 서로 다른 점수 {distinct}개 / 점수 폭 {spread:.0f}점 / "
            f"1위 {alive.iloc[0]['식물명']}({alive.iloc[0]['점수']:.0f})",
        )


def check_window_gate(plants: pd.DataFrame) -> None:
    """기간 누적 입력에서 게이트가 평균이 아니라 극값으로 걸리는지.

    이 검사가 이 구조의 존재 이유다. 평균으로 게이트를 판정하면 겨울 한파가 평균에 묻혀
    겨울에 죽을 식물이 "적합"으로 추천된다.
    """
    print("\n[6] 누적 입력 게이트(평균이 아닌 극값 기준)")
    mean_only = summarize_environment(20_000, 6, 55, config.CULTIVATION_OUTDOOR)
    with_cold = summarize_environment(20_000, [-12, 6, 24], 55, config.CULTIVATION_OUTDOOR)

    alive_mean = (score_all_plants(plants, mean_only)["점수"] > 0).sum()
    alive_cold = (score_all_plants(plants, with_cold)["점수"] > 0).sum()
    _check(
        alive_cold < alive_mean,
        "한파를 포함한 기간은 평균만 볼 때보다 추천 종이 줄어야 함",
        f"평균 6℃만 볼 때 {alive_mean}종 -> 기간 최저 -12℃ 반영 시 {alive_cold}종",
    )

    # 평균이 같아도 변동폭이 크면 더 줄어야 한다(변동폭이 게이트에 반영되는지).
    narrow = summarize_environment(20_000, [2, 6, 10], 55, config.CULTIVATION_OUTDOOR)
    alive_narrow = (score_all_plants(plants, narrow)["점수"] > 0).sum()
    _check(
        alive_cold < alive_narrow,
        "평균이 같아도 변동폭이 크면 추천 종이 더 줄어야 함",
        f"변동 작은 창 {alive_narrow}종 vs 변동 큰 창 {alive_cold}종 (둘 다 평균 6℃)",
    )


def check_model_agreement(plants: pd.DataFrame) -> None:
    """서빙용 ML 모델(v2)이 룰 점수를 충분히 근사하는지.

    ML 모델은 룰을 ONNX로 배포 가능한 형태로 압축한 것이라, 룰과의 차이는 전부 손실이다.

    단, 사용자 환경 온도가 어떤 종의 한계온도와 정확히 같은 지점에서는 룰 자체가 불연속이다
    (한계온도 미만은 게이트로 0점, 이상은 감점 점수). 연속 함수인 회귀모델은 이 계단을
    한 점에서 재현할 수 없어 구조적으로 큰 오차가 남는다. 이 경계점은 실패로 세지 않고
    따로 보고한다 — 게이트 자체는 predict.py가 룰로 강제하므로 "추천/제외" 판정은 틀리지 않는다.
    """
    print("\n[7] 룰 vs ML 모델(v2) 일치도")
    for name, env in SCENARIOS:
        rule = score_all_plants(plants, env).set_index("식물명")["점수"]
        pred = predict_all(plants, env).set_index("식물명")["점수"]
        joined = pd.concat([rule.rename("rule"), pred.rename("ml")], axis=1).dropna()
        joined = joined[joined["rule"] > 0]

        # 환경 온도가 한계온도 하한과 사실상 같은 종 = 룰의 불연속점.
        limit_lower = plants.set_index(config.COL_NAME)[config.NORM_TEMP_LIMIT_LOWER]
        at_boundary = (limit_lower.reindex(joined.index) - env.temp_min).abs() < 0.5
        interior = joined[~at_boundary]

        error = (interior["rule"] - interior["ml"]).abs()
        boundary_note = ""
        if at_boundary.any():
            boundary_error = (joined[at_boundary]["rule"] - joined[at_boundary]["ml"]).abs()
            boundary_note = f" / 한계온도 경계 {int(at_boundary.sum())}종 최대 {boundary_error.max():.1f}점(구조적)"
        _check(
            error.mean() < 2.0 and error.max() < 8.0,
            f"{name}",
            f"MAE {error.mean():.2f}점 / 최대 {error.max():.2f}점{boundary_note}",
        )


def main() -> None:
    plants = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    print(f"검증 대상: {len(plants)}종 ({config.NORMALIZED_OUTPUT_PATH.name})")

    check_ranking_direction(plants)
    check_no_cliffs(plants)
    check_ratio_penalty(plants)
    check_humidity_exception(plants)
    check_discrimination(plants)
    check_window_gate(plants)
    check_model_agreement(plants)

    print()
    if _failures:
        print(f"실패 {len(_failures)}건: {_failures}")
        raise SystemExit(1)
    print("전체 통과")


if __name__ == "__main__":
    main()
