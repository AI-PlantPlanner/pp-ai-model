"""유의성 검정 (docs/model_design.md 3-6, 규칙은 실행 전 고정).

종 짝 부트스트랩(2,000회, seed 0) + 윌콕슨 부호 순위(종별 절대 오차) + Holm 보정, 비교 10개.
입력: results/eval/setting{1,2}_per_plant.csv, results/baseline/species_stats.csv
출력: results/eval/significance.csv
실행: python -m src.significance
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.baseline_stats import RES as BASE_RES
from src.evaluate import EVAL, by_key

N_BOOT, SEED = 2000, 0
GROUPS = ["전체", "열대·아열대", "온대"]


def loo_err(x, y) -> np.ndarray:
    """베이스라인 loo_mae와 같은 계산, 종별 절대 오차를 돌려줌."""
    e = np.empty(len(x))
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        a, b = np.polyfit(x[m], y[m], 1)
        e[i] = abs(a * x[i] + b - y[i])
    return e


def rho(x, y):
    return spearmanr(x, y).statistic


def boot_idx(climate: np.ndarray, group: str, rng) -> list:
    """전체는 그대로 복원 추출, 집단은 집단 안에서 복원 추출(크기 고정)."""
    pool = np.arange(len(climate)) if group == "전체" else np.flatnonzero(climate == group)
    return [pool[rng.integers(0, len(pool), len(pool))] for _ in range(N_BOOT)]


def boot_test(stat_a, stat_b, idx_list, obs) -> dict:
    """stat_*: idx → 시드 평균 지표. 차이 = a − b."""
    d = np.array([stat_a(i) - stat_b(i) for i in idx_list])
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
    return {"diff": obs, "ci_lo": lo, "ci_hi": hi, "p_boot": p}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    rows = []

    # ---- 설정 2: 모델 vs 베이스라인 ----
    s2 = pd.read_csv(EVAL / "setting2_per_plant.csv")
    base = pd.read_csv(BASE_RES / "species_stats.csv")[["count_key", "cold_ext"]]
    seeds = sorted(s2["seed"].unique())
    per = {s: by_key(s2[s2.seed == s], ["model_cold", "target"]).dropna() for s in seeds}
    keys = per[seeds[0]]["count_key"].to_numpy()
    assert all((per[s]["count_key"].to_numpy() == keys).all() for s in seeds)
    b = base.set_index("count_key").loc[keys, "cold_ext"].to_numpy()
    y = per[seeds[0]]["target"].to_numpy()
    clim = per[seeds[0]]["climate"].to_numpy()
    M = np.stack([per[s]["model_cold"].to_numpy() for s in seeds])  # 시드 × 종
    rng = np.random.default_rng(SEED)
    for g in GROUPS:
        idx = boot_idx(clim, g, rng)
        m = np.ones(len(keys), bool) if g == "전체" else clim == g
        obs = np.mean([rho(M[k][m], y[m]) for k in range(len(seeds))]) - rho(b[m], y[m])
        r = boot_test(lambda i: np.mean([rho(M[k][i], y[i]) for k in range(len(seeds))]),
                      lambda i: rho(b[i], y[i]), idx, obs)
        rows.append({"setting": 2, "metric": f"rho_{g}", "vs": "baseline", "model": obs + rho(b[m], y[m]),
                     "other": rho(b[m], y[m]), **r, "p_wilcoxon": np.nan})
    E = np.stack([loo_err(M[k], y) for k in range(len(seeds))])  # 시드 × 종
    eb = loo_err(b, y)
    idx = boot_idx(clim, "전체", rng)
    obs = E.mean() - eb.mean()
    r = boot_test(lambda i: E[:, i].mean(), lambda i: eb[i].mean(), idx, obs)
    rows.append({"setting": 2, "metric": "mae", "vs": "baseline", "model": E.mean(), "other": eb.mean(), **r,
                 "p_wilcoxon": wilcoxon(E.mean(0), eb).pvalue})

    # ---- 설정 1: 모델 vs RF·릿지·기후형 평균 ----
    s1 = pd.read_csv(EVAL / "setting1_per_plant.csv")
    cols = ["model_cold", "model_pred", "ridge_pred", "rf_pred", "ctype_pred", "target"]
    per = {s: by_key(s1[s1.seed == s], cols) for s in seeds}
    keys = per[seeds[0]]["count_key"].to_numpy()
    assert all((per[s]["count_key"].to_numpy() == keys).all() for s in seeds)
    y = per[seeds[0]]["target"].to_numpy()
    clim = per[seeds[0]]["climate"].to_numpy()
    A = {c: np.stack([per[s][c].to_numpy() for s in seeds]) for c in cols[:-1]}

    def rho_nan(x, yy):
        ok = ~np.isnan(x)
        return rho(x[ok], yy[ok])

    def mrho(c, i):
        return np.mean([rho_nan(A[c][k][i], y[i]) for k in range(len(seeds))])

    def merr(c):
        return np.abs(A[c] - y)  # 시드 × 종 (적합 칸 없는 종은 nan)

    for other in ["rf", "ridge", "ctype"]:
        idx = boot_idx(clim, "전체", rng)
        allidx = np.arange(len(keys))
        obs = mrho("model_cold", allidx) - mrho(f"{other}_pred", allidx)
        r = boot_test(lambda i: mrho("model_cold", i), lambda i: mrho(f"{other}_pred", i), idx, obs)
        rows.append({"setting": 1, "metric": "rho_전체", "vs": other, "model": mrho("model_cold", allidx),
                     "other": mrho(f"{other}_pred", allidx), **r, "p_wilcoxon": np.nan})
        em, eo = merr("model_pred"), merr(f"{other}_pred")
        ok = ~np.isnan(em)  # 모델 예측이 없는 (시드, 종)은 두 방법 모두 제외 — 같은 종 집합으로 비교
        eo_ = np.where(ok, eo, np.nan)
        mm = lambda i, e: np.nanmean(e[:, i])
        obs = mm(allidx, em) - mm(allidx, eo_)
        r = boot_test(lambda i: mm(i, em), lambda i: mm(i, eo_), idx, obs)
        both = ~np.isnan(np.nanmean(em, 0))
        rows.append({"setting": 1, "metric": "mae", "vs": other, "model": mm(allidx, em), "other": mm(allidx, eo_), **r,
                     "p_wilcoxon": wilcoxon(np.nanmean(em, 0)[both], np.nanmean(eo_, 0)[both]).pvalue})

    out = pd.DataFrame(rows)
    out["p_boot_holm"] = holm(out["p_boot"].to_numpy())
    w = out["p_wilcoxon"].notna()
    out.loc[w, "p_wilcoxon_holm"] = holm(out.loc[w, "p_wilcoxon"].to_numpy())
    out["sig"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    out = out.round(4)
    out.to_csv(EVAL / "significance.csv", index=False, encoding="utf-8-sig")
    print(out.to_string())


if __name__ == "__main__":
    main()
