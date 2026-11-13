"""백로그 6번 4·5단계: 최종 모델 평가 — 설정 1(학습에 없던 식물)·설정 2(학습에 있던 식물). 설계 docs/model_design.md 2-6.

1) score: 모델 하나(겹 k, 시드 s)로 전체 육지 칸 84,340 × 10년을 모두 임베딩 → 76식물 점수
   ŷ = sigmoid(cos × 배율 + b)(손실 B의 출현 확률, SINR 식 5와 같은 출력) → 칸마다 10년 평균.
   저장: results/eval/probs/fold{k}_s{s}.npy (칸, 식물) float16. 겹 -1 = 설정 2(전 종 학습) 모델.
2) report: 저장된 점수로 지표 계산(모델 다시 안 돌림).

'적합 칸' 정하기(문턱): max SSS = 민감도+특이도 최대(Liu, White & Newell 2013, J. Biogeogr. 40:778 — 원문 확인:
  "not affected by pseudo-absences", "When there are no independent presence data for threshold selection, the training
  data can be used for this purpose"). 출현 = 그 종 관측 칸, 배경 = 전체 육지 칸(무작위 점 대신 전수).
  - 설정 2: 종마다 자기 관측 칸으로 문턱(원문 그대로, 학습 자료 사용).
  - 설정 1: 평가 종은 관측 칸을 쓰면 안 되므로, 같은 모델의 학습 종 문턱들의 중앙값을 평가 종에 적용(우리 적용 — 원문은 종별 문턱).
모델의 추위 지표 = 적합 칸들의 cold_ext 하위 5퍼센타일(베이스라인 src/baseline_stats.py와 같은 연산).

지표(미리 정함, 모두 보고)
  설정 1 — 겹 5개 예측을 모아 종(count_key) 단위로:
    ① 자생지 AUC: 평가 종 관측 칸 vs 전체 칸(종별 평균)
    ② 추위 지표 vs 원예 한계온도 하한: Spearman(전체·열대·아열대·온대), 오차 MAE(℃) — 학습 종으로 맞춘 1차 회귀로 ℃ 환산
    비교: (a) 원예 입력만으로 한계 하한 회귀(같은 23개 입력·같은 겹): 릿지(규제 세기 학습 겹 안 LOO, ISLR 6.2)와
          랜덤 포레스트(Breiman 2001; 표 자료에서 나무 모델이 강함 — Grinsztajn 2022, 원문 확인), sklearn 기본값
          (b) 같은 기후형(4분류) 학습 식물 평균  (c) 베이스라인 ρ 0.889 = 평가 종 GBIF를 직접 쓴 값(상한 참고, 설정 다름)
  설정 2 — 전 종 학습 모델, 종별 문턱: 추위 지표 Spearman(전체·집단 안) + LOO MAE — 베이스라인 cold_ext와 같은 종·같은 계산.
  시드 5개 각각 계산 → 평균 ± 표준편차.
실행: python -m src.evaluate score --fold 0 --seed 0 | python -m src.evaluate score-all | python -m src.evaluate report
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, roc_curve

from src.align import RES as ALIGN_RES, build_model, make_inputs
from src.baseline_stats import loo_mae
from src.build_dataset import DS
from src.gee_annual_extremes import OUT, ROOT
from src.model_data import CLIMATE_COL, FOLDS_F, PlantFeatures, plant_table

EVAL = ROOT / "results" / "eval"
PROBS = EVAL / "probs"
TAG, PATCH = "B_p8_lam1", 8
SEEDS = range(5)
FOLDS = range(5)
GROUPS = ["전체", "열대·아열대", "온대"]


def sfx(ablate: str) -> str:
    return f"_abl-{ablate}" if ablate != "none" else ""


def model_dir(fold: int, seed: int, ablate: str = "none"):
    return ALIGN_RES / "final" / f"fold{fold}" / (TAG + (f"_s{seed}" if seed else "") + sfx(ablate))


def probs_f(fold: int, seed: int, ablate: str = "none"):
    return PROBS / f"fold{fold}_s{seed}{sfx(ablate)}.npy"


def train_rows(fold: int) -> np.ndarray:
    f = pd.read_csv(FOLDS_F)["fold"].to_numpy()
    return np.flatnonzero(f != fold)  # 겹 -1이면 전부


# ---------- 1) 점수 ----------
@torch.no_grad()
def score(fold: int, seed: int, ablate: str = "none", bs: int = 2048) -> None:
    out = probs_f(fold, seed, ablate)
    if out.exists():
        print("있음", out); return
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    table = plant_table()
    pf = PlantFeatures(table, use_climate_type=ablate != "no_ctype").fit_scaler(train_rows(fold))  # 학습 때와 같은 표준화
    C = make_inputs(ablate)
    model = build_model(ablate, PATCH, pf.dim, C)
    model.load_state_dict(torch.load(model_dir(fold, seed, ablate) / "best.pt", map_location="cpu")["model"])
    model.to(dev).eval()
    zp = model.plant(pf.tensor().to(dev))
    scale, b = model.scale(), model.b
    full = getattr(C, "clim", None)
    acc = torch.zeros(C.n_src, len(table))
    for y in range(C.n_years):
        if full is not None:
            C.clim = np.ascontiguousarray(full[y:y + 1])  # 한 해(약 246 MB)를 통째로 메모리에 — 메모리 맵 조각 읽기보다 빠름
        for i in range(0, C.n_src, bs):
            src = np.arange(i, min(i + bs, C.n_src))
            zc = model.climate(C.batch(src, np.zeros(len(src), int)).to(dev))
            acc[src] += torch.sigmoid(zc @ zp.T * scale + b).cpu()
        print(f"fold{fold} s{seed} 연도 {y + 1}/{C.n_years}", flush=True)
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(out, (acc / C.n_years).numpy().astype(np.float16))


# ---------- 2) 지표 ----------
def presence_cells(table: pd.DataFrame) -> dict:
    sc = pd.read_parquet(DS / "species_cells.parquet", columns=["count_key", "src_id", "n_v_main"])
    sc = sc[sc["n_v_main"] > 0].drop_duplicates(["count_key", "src_id"])
    return {k: v.to_numpy() for k, v in sc[sc["count_key"].isin(set(table["count_key"]))].groupby("count_key")["src_id"]}


def max_sss(p: np.ndarray, pres: np.ndarray) -> float:
    """출현 칸 vs 전체 칸 ROC에서 민감도+특이도(= tpr − fpr + 1) 최대 문턱."""
    y = np.zeros(len(p), bool); y[pres] = True
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def cold_of(p: np.ndarray, thr: float, cold: np.ndarray) -> tuple:
    m = p >= thr
    return (float(np.percentile(cold[m], 5)) if m.any() else np.nan), int(m.sum())


def by_key(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """같은 종 두 식물(디펜바키아·트로픽스노우) → 종 하나로 평균(베이스라인과 같은 단위)."""
    return df.groupby("count_key").agg({"climate": "first", **{c: "mean" for c in cols}}).reset_index()


def rho_by_group(df, x, y="target") -> dict:
    out = {}
    for g in GROUPS:
        s = df if g == "전체" else df[df["climate"] == g]
        s = s.dropna(subset=[x, y])
        out[g] = float(spearmanr(s[x], s[y]).statistic) if len(s) > 2 else np.nan
    return out


def setting1(seed, table, pres, cold, ablate="none") -> tuple:
    folds = pd.read_csv(FOLDS_F)["fold"].to_numpy()
    rows = []
    for k in FOLDS:
        probs = np.load(probs_f(k, seed, ablate)).astype(np.float32)
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        thr_tr = {r: max_sss(probs[:, r], pres[table.at[r, "count_key"]]) for r in tr}
        tau = float(np.median(list(thr_tr.values())))
        cold_tr = np.array([cold_of(probs[:, r], tau, cold)[0] for r in tr])
        y_tr = table.loc[tr, "temp_limit_lower_min"].to_numpy()
        ok = ~np.isnan(cold_tr)
        a, b = np.polyfit(cold_tr[ok], y_tr[ok], 1)  # 추위 지표 → 원예 ℃ (학습 종으로만)
        # 비교 (a) 원예 입력 회귀 (b) 기후형 평균
        pf = PlantFeatures(table).fit_scaler(tr); X = pf.tensor().numpy()
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], y_tr)
        rf = RandomForestRegressor(random_state=seed).fit(X[tr], y_tr)
        ct = table[CLIMATE_COL]
        ct_mean = pd.Series(y_tr).groupby(ct.iloc[tr].to_numpy()).mean()
        for r in te:
            p = probs[:, r]; key = table.at[r, "count_key"]
            yb = np.zeros(len(p), bool); yb[pres[key]] = True
            c, n = cold_of(p, tau, cold)
            rows.append({"fold": k, "row": r, "count_key": key, "auc": roc_auc_score(yb, p), "tau": tau, "n_suitable": n,
                         "model_cold": c, "model_pred": a * c + b if not np.isnan(c) else np.nan,
                         "ridge_pred": float(ridge.predict(X[[r]])[0]), "rf_pred": float(rf.predict(X[[r]])[0]),
                         "ctype_pred": float(ct_mean.get(ct.iat[r], y_tr.mean()))})
    d = pd.DataFrame(rows)
    d["target"] = table.loc[d["row"], "temp_limit_lower_min"].to_numpy()
    d["climate"] = np.where(table.loc[d["row"], CLIMATE_COL].eq("온대"), "온대", "열대·아열대")
    s = by_key(d, ["auc", "model_cold", "model_pred", "ridge_pred", "rf_pred", "ctype_pred", "target", "n_suitable"])
    res = {"auc_mean": float(s["auc"].mean()), "n_species": len(s), "no_suitable": int(s["model_cold"].isna().sum())}
    for name, col in [("model", "model_cold"), ("ridge", "ridge_pred"), ("rf", "rf_pred"), ("ctype", "ctype_pred")]:
        res[f"rho_{name}"] = rho_by_group(s, col)
    for name, col in [("model", "model_pred"), ("ridge", "ridge_pred"), ("rf", "rf_pred"), ("ctype", "ctype_pred")]:
        e = (s[col] - s["target"]).abs()
        res[f"mae_{name}"] = float(e.mean())
    return res, d


def setting2(seed, table, pres, cold, ablate="none") -> tuple:
    probs = np.load(probs_f(-1, seed, ablate)).astype(np.float32)
    rows = []
    for r in range(len(table)):
        key = table.at[r, "count_key"]
        thr = max_sss(probs[:, r], pres[key])
        c, n = cold_of(probs[:, r], thr, cold)
        rows.append({"row": r, "count_key": key, "thr": thr, "n_suitable": n, "model_cold": c,
                     "target": table.at[r, "temp_limit_lower_min"],
                     "climate": "온대" if table.at[r, CLIMATE_COL] == "온대" else "열대·아열대"})
    d = pd.DataFrame(rows)
    s = by_key(d, ["model_cold", "target", "n_suitable"]).dropna()
    m, med, m0 = loo_mae(s["model_cold"].to_numpy(), s["target"].to_numpy())
    return {"rho_model": rho_by_group(s, "model_cold"), "loo_mae": m, "loo_median_ae": med, "mean_only_mae": m0,
            "n_species": len(s)}, d


def summarize(per_seed: list) -> dict:
    """시드별 결과 dict 목록 → 같은 구조에 [평균, 표준편차]."""
    def walk(vals):
        if isinstance(vals[0], dict):
            return {k: walk([v[k] for v in vals]) for k in vals[0]}
        a = np.asarray(vals, float)
        return [round(float(np.nanmean(a)), 3), round(float(np.nanstd(a, ddof=1)), 3) if len(a) > 1 else None]
    return walk(per_seed)


def report(ablate: str = "none") -> None:
    table = plant_table()
    pres = presence_cells(table)
    cold = pd.read_parquet(OUT / "cell_climate_10y.parquet", columns=["src_id", "cold_ext"]).sort_values("src_id")["cold_ext"].to_numpy()
    out = {}
    s1 = [setting1(s, table, pres, cold, ablate) for s in SEEDS if all(probs_f(k, s, ablate).exists() for k in FOLDS)]
    if s1:
        out["setting1"] = {"per_seed": [r for r, _ in s1], "summary": summarize([r for r, _ in s1])}
        pd.concat([d.assign(seed=i) for i, (_, d) in enumerate(s1)]).to_csv(EVAL / f"setting1_per_plant{sfx(ablate)}.csv", index=False, encoding="utf-8-sig")
    s2 = [setting2(s, table, pres, cold, ablate) for s in SEEDS if probs_f(-1, s, ablate).exists()]
    if s2:
        out["setting2"] = {"per_seed": [r for r, _ in s2], "summary": summarize([r for r, _ in s2])}
        pd.concat([d.assign(seed=i) for i, (_, d) in enumerate(s2)]).to_csv(EVAL / f"setting2_per_plant{sfx(ablate)}.csv", index=False, encoding="utf-8-sig")
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / f"summary{sfx(ablate)}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    for k, v in out.items():
        print(k, f"(시드 {len(v['per_seed'])}개)"); print(json.dumps(v["summary"], ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "score-all", "report"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="none")
    a = ap.parse_args()
    if a.cmd == "score":
        score(a.fold, a.seed, a.ablate)
    elif a.cmd == "score-all":
        for s in SEEDS:
            for k in [*FOLDS, -1]:
                if (model_dir(k, s, a.ablate) / "summary.json").exists():
                    score(k, s, a.ablate)
    else:
        report(a.ablate)


if __name__ == "__main__":
    main()
