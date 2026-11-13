"""D(최소 기록 수) 시험: 종마다 관측 칸 수를 N개로 줄였을 때, 원예 한계온도 하한과의 일치도가 언제 전체 수준에 이르는가.

근거: van Proosdij et al. 2016(Ecography 39:542–552)은 표본 수를 줄여 가며 성능이 무너지지 않는 최소 수를
연구지역마다 직접 구하라고 권한다(그들의 아프리카 값 14~25는 지역 고유값). 같은 생각을 우리 지표에 적용한다.

방법: 칸이 100개 이상인 종만 대상으로(전체 값을 '정답'으로 쓸 수 있도록), N ∈ {5,10,14,20,25,30,50,100}마다
종별로 칸 N개를 무작위로 뽑아(반복 200회) 추위 지표(10년 연 극한 최저기온 평균의 하위 5퍼센타일)를 구하고
① 전체 칸으로 구한 값과의 절대오차 ② 원예 한계온도 하한과의 Spearman 상관을 본다.

입력: year_window_experiment와 같음. 출력: data/weather/min_records_test.csv
실행: python -m src.min_records_test
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.gee_annual_extremes import GRID, OUT, ROOT
from src.year_window_experiment import EXCLUDE_KEYS, window_means

NS = [5, 10, 14, 20, 25, 30, 50, 100]
REPS = 200
COL = "cold_10y_2016_2025"


def main() -> None:
    cv = window_means(pd.read_csv(OUT / "annual_extremes_1991_2025.csv"))[["cell_id", COL]]
    o = pd.read_parquet(ROOT / "data/gbif/occurrences_clean.parquet",
                        columns=["count_key", "decimalLatitude", "decimalLongitude", "v_main"])
    o = o[o["v_main"]]
    ilat = np.round(o["decimalLatitude"] / GRID).astype(int)
    ilon = ((np.round(o["decimalLongitude"] / GRID).astype(int) + 1800) % 3600) - 1800
    o = pd.DataFrame({"k": o["count_key"].astype(int), "cell_id": ilat.astype(str) + "_" + ilon.astype(str)})
    o = o.drop_duplicates().merge(cv, on="cell_id").dropna()
    n = o.groupby("k").size()
    o = o[o["k"].isin(n[n >= 100].index.difference(list(EXCLUDE_KEYS)))]

    match = pd.read_csv(ROOT / "data/gbif/taxon_match.csv")[["식물명", "count_key"]]
    plants = pd.read_csv(ROOT / "data/processed/plants_normalized.csv").rename(columns={"식물명 (국명)": "식물명"})
    hort = match.merge(plants[["식물명", "temp_limit_lower_min"]], on="식물명").groupby("count_key")[
        "temp_limit_lower_min"].mean()

    groups = {k: g[COL].to_numpy() for k, g in o.groupby("k")}
    keys = sorted(groups)
    full = pd.Series({k: np.percentile(v, 5) for k, v in groups.items()})
    y = hort.reindex(keys).to_numpy()
    rho_full = spearmanr(full.reindex(keys), y).statistic
    rng = np.random.default_rng(0)
    rows = []
    for N in NS:
        errs, rhos = [], []
        for _ in range(REPS):
            est = np.array([np.percentile(rng.choice(groups[k], N, replace=False), 5) for k in keys])
            errs.append(np.abs(est - full.reindex(keys).to_numpy()))
            rhos.append(spearmanr(est, y).statistic)
        e = np.concatenate(errs)
        rows.append({"N": N, "err_median_C": round(np.median(e), 2), "err_p90_C": round(np.percentile(e, 90), 2),
                     "rho_mean": round(np.mean(rhos), 3), "rho_p2.5": round(np.percentile(rhos, 2.5), 3),
                     "rho_p97.5": round(np.percentile(rhos, 97.5), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "min_records_test.csv", index=False)
    print(f"대상 {len(keys)}종(칸 ≥100), 전체 칸 기준 상관 {rho_full:.3f}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
