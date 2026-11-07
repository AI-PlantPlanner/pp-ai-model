"""연도 실험: 기상 기간(1년 / 10년 / 30년)에 따라 GBIF 관측지 기후와 원예 한계온도의 일치도가 어떻게 달라지는가. (결정 E)

설계 (근거는 src/gee_annual_extremes.py 머리말과 메모 참조):
  - 칸 지표: 연 극한 최저기온(연 최저 일최저기온)의 기간 평균 — USDA 내한성 지도 정의.
    더위 쪽은 연 극한 최고기온(연 최고 일최고기온)의 기간 평균(보조).
  - 종 지표: 그 종이 관측된 칸들의 하위 5퍼센타일(추위)·상위 95퍼센타일(더위) — 10-01 파일럿과 같은 방식.
  - 일치도: 원예 한계온도 하한(추위)·상한(더위)과의 Spearman 순위 상관.
  - 평가 순환 방지: 종을 무작위로 반씩(선택용/확인용) 나눠, 선택용에서 고른 기간이 확인용에서도 유지되는지 본다.
  - 대상: v_main(원산 + 도시 제외), 0.1° 칸 기준 25칸 이상인 종, 섬초롱꽃 제외(heterotypic 이명).
    두 식물이 같은 GBIF 종을 가리키면(디펜바키아·트로픽스노우) 원예값은 평균을 쓴다.
  - 칸 수: 비용이 작아 표본 추출 없이 모든 칸을 쓴다.

입력: data/gbif/occurrences_clean.parquet, data/weather/annual_extremes_1991_2025.csv,
      data/gbif/taxon_match.csv, data/processed/plants_normalized.csv
출력: data/weather/year_window_species.csv(종별 지표), data/weather/year_window_result.csv(상관 요약)
실행: python -m src.year_window_experiment
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.gee_annual_extremes import GRID, OUT, ROOT

MIN_CELLS = 25
EXCLUDE_KEYS = {7222087}  # 섬초롱꽃: WCVP heterotypic 이명 → SAGE 규칙상 병합 불가
WINDOWS = {"1y_2025": (2025, 2025), "10y_2016_2025": (2016, 2025), "30y_1991_2020": (1991, 2020)}
K = 273.15
N_BOOT = 2000
SEED = 0


def window_means(ext: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"cell_id": ext["cell_id"]})
    for w, (a, b) in WINDOWS.items():
        yrs = range(a, b + 1)
        out[f"cold_{w}"] = ext[[f"tmin_{y}" for y in yrs]].mean(axis=1) - K
        out[f"hot_{w}"] = ext[[f"tmax_{y}" for y in yrs]].mean(axis=1) - K
    return out


def species_table(cellv: pd.DataFrame) -> pd.DataFrame:
    o = pd.read_parquet(ROOT / "data" / "gbif" / "occurrences_clean.parquet",
                        columns=["count_key", "decimalLatitude", "decimalLongitude", "v_main"])
    o = o[o["v_main"]]
    ilat = np.round(o["decimalLatitude"] / GRID).astype(int)
    ilon = ((np.round(o["decimalLongitude"] / GRID).astype(int) + 1800) % 3600) - 1800
    o = pd.DataFrame({"count_key": o["count_key"].astype(int),
                      "cell_id": ilat.astype(str) + "_" + ilon.astype(str)}).drop_duplicates()
    n_cells = o.groupby("count_key").size()
    keep = n_cells[(n_cells >= MIN_CELLS) & ~n_cells.index.isin(EXCLUDE_KEYS)].index
    o = o[o["count_key"].isin(keep)].merge(cellv, on="cell_id", how="left")
    cols = [c for c in cellv.columns if c != "cell_id"]
    rows = []
    for k, g in o.groupby("count_key"):
        r = {"count_key": k, "n_cells": len(g), "n_cells_with_weather": int(g[cols[0]].notna().sum())}
        for c in cols:
            q = 5 if c.startswith("cold_") else 95
            r[c] = np.nanpercentile(g[c], q)
        rows.append(r)
    sp = pd.DataFrame(rows)

    match = pd.read_csv(ROOT / "data" / "gbif" / "taxon_match.csv")[["식물명", "count_key"]]
    plants = pd.read_csv(ROOT / "data" / "processed" / "plants_normalized.csv")
    plants = plants.rename(columns={"식물명 (국명)": "식물명"})
    lim = ["temp_limit_lower_min", "temp_limit_lower_max", "temp_limit_upper_min", "temp_limit_upper_max"]
    hort = match.merge(plants[["식물명"] + lim], on="식물명", how="left")
    hort = hort.groupby("count_key").agg({"식물명": " / ".join, **{c: "mean" for c in lim}}).reset_index()
    return sp.merge(hort, on="count_key", how="left")


def corr_with_ci(x: pd.Series, y: pd.Series, rng) -> tuple:
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(), y[ok].to_numpy()
    rho = spearmanr(x, y).statistic
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi, int(ok.sum())


def main() -> None:
    ext = pd.read_csv(OUT / "annual_extremes_1991_2025.csv")
    sp = species_table(window_means(ext))
    rng = np.random.default_rng(SEED)
    keys = sp["count_key"].to_numpy().copy()
    rng.shuffle(keys)
    half = set(keys[: len(keys) // 2])
    sp["split"] = np.where(sp["count_key"].isin(half), "select", "confirm")
    sp.to_csv(OUT / "year_window_species.csv", index=False, encoding="utf-8-sig")

    rows = []
    for split in ["select", "confirm", "all"]:
        s = sp if split == "all" else sp[sp["split"] == split]
        for w in WINDOWS:
            for side, col, targets in [("cold", f"cold_{w}", ["temp_limit_lower_min", "temp_limit_lower_max"]),
                                       ("hot", f"hot_{w}", ["temp_limit_upper_min", "temp_limit_upper_max"])]:
                for t in targets:
                    rho, lo, hi, n = corr_with_ci(s[col], s[t], np.random.default_rng(SEED))
                    rows.append({"split": split, "window": w, "side": side, "target": t,
                                 "n_species": n, "spearman": round(rho, 3),
                                 "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "year_window_result.csv", index=False)
    print(res[res["side"] == "cold"].to_string(index=False))
    print(res[res["side"] == "hot"].to_string(index=False))


if __name__ == "__main__":
    main()
