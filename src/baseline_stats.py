"""백로그 5번: 단순 통계 베이스라인 — 자생지 기후 통계만으로 원예 기준을 얼마나 맞히는가.

딥러닝 모델(6번)이 넘어야 할 기준선. 10-01 파일럿(WorldClim BIO6 관측지 5퍼센타일 vs 한계온도 하한 ρ 0.85, 86종)을
새 데이터(정제·10년 ERA5-Land·76식물)로 다시 구현하고, 파일럿에서 못 본 더위·습도까지 같은 방식으로 본다.

종 통계(관측 칸들의 분포에서, 칸은 종마다 1번):
  추위 쪽 = 하위 5퍼센타일, 더위 쪽 = 상위 95퍼센타일(파일럿과 같음), 연평균기온 = 중앙값,
  습도 = 하위 5퍼센타일(건조한 쪽 한계 — 추위와 같은 논리).
비교 대상(원예 자료): 한계온도 하한·상한(최소값), 적정온도 하한·중앙, 적정습도 하한.
  같은 GBIF 종을 두 식물이 가리키면(디펜바키아·트로픽스노우) 원예값 평균.
평가:
  ① Spearman 순위 상관 + 부트스트랩 95% CI(2000회, seed 0)
  ② 한계온도 하한 예측 오차: 종 통계 → 원예값 1차 회귀를 한 종씩 빼고 맞춰(leave-one-out) MAE(℃).
     비교용 무정보 기준 = 나머지 종 평균으로 예측. 6번 모델은 같은 종 단위 평가로 이 MAE와 비교한다.
  ③ 기후형(열대·아열대 / 온대) 안에서의 상관 — 파일럿에서 집단 안 상관이 0.58·0.67로 약했음.
  ④ 감도 세트(build_dataset의 8개)에서 주 지표(cold_ext)의 상관.
지표는 미리 정한 것을 모두 보고한다(결과를 보고 고르지 않음). 습도는 선행 근거가 없어 탐색 분석으로 서술.

입력: data/dataset/plants.csv, species_cells.parquet, data/weather/cell_climate_10y.parquet,
      data/processed/plants_normalized.csv
출력: results/baseline/species_stats.csv, correlations.csv, loo_mae.csv, sensitivity.csv
실행: python -m src.baseline_stats
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.build_dataset import DS, VERSIONS, eligible
from src.gee_annual_extremes import OUT, ROOT

RES = ROOT / "results" / "baseline"
N_BOOT = 2000
SEED = 0
STATS = {  # 지표: (칸 분포에서 쓰는 분위, 비교할 원예값들)
    "cold_ext": (5, ["temp_limit_lower_min"]),
    "bio6": (5, ["temp_limit_lower_min"]),
    "hot_ext": (95, ["temp_limit_upper_min"]),
    "bio5": (95, ["temp_limit_upper_min"]),
    "t_annual": (50, ["temp_optimal_min", "temp_optimal_mid"]),
    "rh_annual": (5, ["humidity_min"]),
    "rh_dry_month": (5, ["humidity_min"]),
    "rh_p05": (5, ["humidity_min"]),
}
CLIMATE_COL = "기후형(열대/ 열대·아열대/아열대/온대)"


def horticulture() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data" / "processed" / "plants_normalized.csv").rename(columns={"식물명 (국명)": "식물명"})
    p["temp_optimal_mid"] = (p["temp_optimal_min"] + p["temp_optimal_max"]) / 2
    p["climate"] = np.where(p[CLIMATE_COL].eq("온대"), "온대", "열대·아열대")
    return p[["식물명", "climate", "temp_limit_lower_min", "temp_limit_upper_min", "temp_optimal_min",
              "temp_optimal_mid", "humidity_min"]]


def species_stats(plants: pd.DataFrame, sc: pd.DataFrame, clim: pd.DataFrame, version: str) -> pd.DataFrame:
    keys = set(plants["count_key"])
    s = sc[sc["count_key"].isin(keys) & (sc[f"n_{version}"] > 0)][["count_key", "src_id"]].merge(clim, on="src_id")
    rows = []
    for k, g in s.groupby("count_key"):
        r = {"count_key": k, "n_cells": len(g)}
        for st, (q, _) in STATS.items():
            r[st] = float(np.percentile(g[st], q))
        rows.append(r)
    sp = pd.DataFrame(rows)
    hort = plants[["식물명", "count_key"]].merge(horticulture(), on="식물명")
    num = [c for c in hort.columns if c not in ("식물명", "count_key", "climate")]
    hort = hort.groupby("count_key").agg({"식물명": " / ".join, "climate": "first", **{c: "mean" for c in num}})
    return sp.merge(hort.reset_index(), on="count_key")


def spearman_ci(x, y) -> tuple:
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(SEED)
    boots = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(N_BOOT))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return round(rho, 3), round(lo, 3), round(hi, 3), int(ok.sum())


def loo_mae(x: np.ndarray, y: np.ndarray) -> tuple:
    err, err0 = [], []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        a, b = np.polyfit(x[m], y[m], 1)
        err.append(abs(a * x[i] + b - y[i]))
        err0.append(abs(y[m].mean() - y[i]))
    return round(float(np.mean(err)), 2), round(float(np.median(err)), 2), round(float(np.mean(err0)), 2)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    plants = pd.read_csv(DS / "plants.csv")
    sc = pd.read_parquet(DS / "species_cells.parquet")
    clim = pd.read_parquet(OUT / "cell_climate_10y.parquet")

    sp = species_stats(eligible(plants), sc, clim, "v_main")
    sp.to_csv(RES / "species_stats.csv", index=False, encoding="utf-8-sig")

    rows = []
    for st, (_, targets) in STATS.items():
        for t in targets:
            for grp, s in [("전체", sp), ("열대·아열대", sp[sp.climate == "열대·아열대"]), ("온대", sp[sp.climate == "온대"])]:
                rho, lo, hi, n = spearman_ci(s[st].to_numpy(), s[t].to_numpy())
                rows.append({"stat": st, "target": t, "group": grp, "n": n, "spearman": rho, "ci_lo": lo, "ci_hi": hi,
                             "target_unique_values": int(s[t].nunique())})
    corr = pd.DataFrame(rows)
    corr.to_csv(RES / "correlations.csv", index=False, encoding="utf-8-sig")

    mae = []
    for st in ["cold_ext", "bio6"]:
        s = sp.dropna(subset=[st, "temp_limit_lower_min"])
        m, med, m0 = loo_mae(s[st].to_numpy(), s["temp_limit_lower_min"].to_numpy())
        mae.append({"stat": st, "n": len(s), "loo_mae_C": m, "loo_median_ae_C": med, "mean_only_mae_C": m0})
    mae = pd.DataFrame(mae)
    mae.to_csv(RES / "loo_mae.csv", index=False, encoding="utf-8-sig")

    sens = []
    for name, (v, d, cv) in {"본(v_main,D25)": ("v_main", 25, True), "D14": ("v_main", 14, True),
                             "품종 제외": ("v_main", 25, False),
                             **{v: (v, 25, True) for v in VERSIONS[1:]}}.items():
        s = species_stats(eligible(plants, v, d, cv), sc, clim, v)
        rho, lo, hi, n = spearman_ci(s["cold_ext"].to_numpy(), s["temp_limit_lower_min"].to_numpy())
        sens.append({"set": name, "n_species": n, "spearman": rho, "ci_lo": lo, "ci_hi": hi})
    sens = pd.DataFrame(sens)
    sens.to_csv(RES / "sensitivity.csv", index=False, encoding="utf-8-sig")

    print(corr.to_string(index=False))
    print(mae.to_string(index=False))
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
