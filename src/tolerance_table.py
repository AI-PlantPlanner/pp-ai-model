"""1번 ①: 모델 → 식물별 견딤 범위 표 (docs/model_design.md 3-8). 새 학습 없음.

설정 2 모델(겹 −1, 시드 5개)의 전체 칸 점수 + 식물별 max SSS 문턱으로 '적합 칸'을 고르고,
그 칸들의 기후 지표에서 한계를 뽑는다. 지표 정의는 평가·베이스라인과 같음(src/cell_climate.py):
  추위 cold_lim = 적합 칸 cold_ext(연 최저 일최저기온, 10년 평균)의 5퍼센타일   ← 평가에서 검증한 값 그대로
  더위 hot_lim  = 적합 칸 hot_ext(연 최고 일최고기온)의 95퍼센타일              ← 베이스라인과 같은 방향
  건조 dry_lim  = 적합 칸 rh_p05(일평균 습도의 연 5퍼센타일)의 5퍼센타일
시드마다 계산 후 평균(± 표준편차). 방 그래프도 같은 연산(연 최저 일최저기온 등)으로 요약해 비교한다.
추위는 원예 한계온도 하한에 맞춘 1차 보정값(cold_cal)도 함께 저장 — 보정 오차는 평가의 LOO MAE(5.02℃)가 이미 보고한 값.
출력: results/tolerance/plant_tolerance.csv, summary.json
실행: python -m src.tolerance_table
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.baseline_stats import loo_mae
from src.evaluate import EVAL, SEEDS, probs_f
from src.gee_annual_extremes import OUT as WOUT, ROOT
from src.model_data import CLIMATE_COL, plant_table

RES = ROOT / "results" / "tolerance"


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    t = plant_table()
    cc = pd.read_parquet(WOUT / "cell_climate_10y.parquet").sort_values("src_id")
    cold, hot, dry = (cc[c].to_numpy() for c in ["cold_ext", "hot_ext", "rh_p05"])
    thr = pd.read_csv(EVAL / "setting2_per_plant.csv").pivot(index="seed", columns="row", values="thr")
    out = np.full((len(SEEDS), len(t), 4), np.nan)  # cold, hot, dry, n_cells
    for s in SEEDS:
        p = np.load(probs_f(-1, s)).astype(np.float32)
        for r in range(len(t)):
            m = p[:, r] >= thr.at[s, r]
            if m.any():
                out[s, r] = [np.percentile(cold[m], 5), np.percentile(hot[m], 95), np.percentile(dry[m], 5), m.sum()]
    mean, sd = np.nanmean(out, 0), np.nanstd(out, 0, ddof=1)
    tab = pd.DataFrame({"식물명": t["식물명"], "count_key": t["count_key"], "climate": t[CLIMATE_COL],
                        "cultivation": t["cultivation_type"],
                        "cold_lim": mean[:, 0], "cold_lim_sd": sd[:, 0], "hot_lim": mean[:, 1], "hot_lim_sd": sd[:, 1],
                        "dry_lim": mean[:, 2], "dry_lim_sd": sd[:, 2], "n_cells": mean[:, 3],
                        "hort_cold": t["temp_limit_lower_min"], "hort_hot": t["temp_limit_upper_min"],
                        "hort_hum_min": t["humidity_min"]})
    ok = tab["cold_lim"].notna()
    a, b = np.polyfit(tab.loc[ok, "cold_lim"], tab.loc[ok, "hort_cold"], 1)
    tab["cold_cal"] = a * tab["cold_lim"] + b
    tab.round(2).to_csv(RES / "plant_tolerance.csv", index=False, encoding="utf-8-sig")

    def rho(x, y):
        k = tab[x].notna() & tab[y].notna()
        return round(float(spearmanr(tab.loc[k, x], tab.loc[k, y]).statistic), 3)

    m_loo, med_loo, _ = loo_mae(tab.loc[ok, "cold_lim"].to_numpy(), tab.loc[ok, "hort_cold"].to_numpy())
    summ = {"n_plants": int(ok.sum()), "cold_cal_fit": {"a": round(a, 3), "b": round(b, 3)},
            "rho_cold_vs_hort": rho("cold_lim", "hort_cold"), "cold_loo_mae": m_loo, "cold_loo_median": med_loo,
            "rho_hot_vs_hort": rho("hot_lim", "hort_hot"), "rho_dry_vs_hort_hum_min": rho("dry_lim", "hort_hum_min"),
            "seed_sd_median": {"cold": round(float(np.nanmedian(sd[:, 0])), 2), "hot": round(float(np.nanmedian(sd[:, 1])), 2),
                               "dry": round(float(np.nanmedian(sd[:, 2])), 2)},
            "note": "식물 76행 단위(평가의 종 75 단위와 약간 다름). cold_cal은 76식물 전체로 맞춤 — 제품용 값, 오차는 LOO로 보고."}
    (RES / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(tab[["식물명", "cold_lim", "cold_cal", "hort_cold", "hot_lim", "hort_hot", "dry_lim", "hort_hum_min"]].round(1).to_string())


if __name__ == "__main__":
    main()
