"""실내 입력 점검 (docs/model_design.md 3-7, 규칙은 실행 전 고정). 새 학습 없음.

점검 1: 실내 1년 그래프가 학습 날씨 분포 안인가 (원자료 최근접 거리 / 모델 임베딩 최근접 유사도 / 사전학습 복원 오차)
점검 2: 실내 시나리오 점수가 원예 상식과 맞는가 (설정 2 모델 5시드)
출력: results/indoor/check1.csv, check2.json, scenario_scores.csv
실행: python -m src.indoor_check
"""

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.align import build_model
from src.evaluate import EVAL, PATCH, SEEDS, model_dir, train_rows
from src.gee_annual_extremes import ROOT
from src.model_data import CLIMATE_COL, Climate, N_DAYS, PlantFeatures, plant_table
from src.model_patchtst import PatchTSTPretrain
from src.pretrain import RES as PRE_RES, split

OUT = ROOT / "results" / "indoor"
SEOUL = 32825
N_REF, SEED, N_BOOT = 2000, 0, 2000
T_HEAT, T_HEAT_LOW, T_COOL = 21.5, 18.5, 28.0
RH_MEASURED, RH_HUMID = 35.4, 50.0
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------- 시나리오 ----------
def es(t):  # 포화 수증기압 hPa (Magnus)
    return 6.112 * np.exp(17.62 * t / (243.12 + t))


def scenarios(raw: np.ndarray) -> dict:
    """raw: (연도, 365, 4) 서울 바깥 [t_min, t_mean, t_max, rh]. 반환: 이름 → 같은 모양."""
    def indoor(lo, rh_floor):
        x = raw.copy()
        x[..., :3] = np.clip(raw[..., :3], lo, T_COOL)
        e_out = raw[..., 3] / 100 * es(raw[..., 1])
        rh_vc = np.clip(100 * e_out / es(x[..., 1]), 0, 100)
        x[..., 3] = np.maximum(rh_vc, rh_floor)
        return x
    return {"S0_실외": raw.copy(), "S1_난방_건조": indoor(T_HEAT, 0), "S2_난방_측정습도": indoor(T_HEAT, RH_MEASURED),
            "S3_난방_가습": indoor(T_HEAT, RH_HUMID), "S4_절약난방": indoor(T_HEAT_LOW, RH_MEASURED)}


def to_input(C: Climate, raw: np.ndarray, src: int) -> torch.Tensor:
    x = (raw.astype(np.float32) - C.mean) / C.std
    dl = (np.broadcast_to(C.daylen[src].astype(np.float32), raw.shape[:2])[..., None] - C.dl_mean) / C.dl_std
    se = np.broadcast_to(C.season, (len(raw), N_DAYS, 2))
    return torch.from_numpy(np.concatenate([x, dl, se], -1).astype(np.float32))


def weekly(xn: np.ndarray) -> np.ndarray:  # (..., 365, 4) 표준화 → (..., 208)
    return xn[..., :364, :].reshape(*xn.shape[:-2], 52, 7, 4).mean(-2).reshape(*xn.shape[:-2], -1)


# ---------- 점검 1 ----------
@torch.no_grad()
def check1(C, model0, queries: dict, ref_pairs: np.ndarray, pool_cells: np.ndarray) -> pd.DataFrame:
    """queries: 이름 → (n, 365, 7) 입력 텐서. 최근접은 pool_cells × 10년에서."""
    names = list(queries)
    Q = torch.cat([queries[k] for k in names])
    Qw = torch.from_numpy(weekly(Q[..., :4].numpy()))
    Qe = torch.cat([model0.climate(Q[i:i + 512].to(DEV)).cpu() for i in range(0, len(Q), 512)])
    best_d = torch.full((len(Q),), float("inf"))
    best_s = torch.full((len(Q),), -1.0)
    for y in range(C.n_years):
        full = np.ascontiguousarray(C.clim[y, pool_cells]).astype(np.float32)  # (P, 365, 4)
        Pw = torch.from_numpy(weekly((full - C.mean) / C.std))
        for i in range(0, len(Pw), 50000):
            best_d = torch.minimum(best_d, torch.cdist(Qw, Pw[i:i + 50000]).min(1).values)
        for i in range(0, len(pool_cells), 2048):
            src = pool_cells[i:i + 2048]
            ze = model0.climate(C.batch(src, np.full(len(src), y)).to(DEV)).cpu()
            best_s = torch.maximum(best_s, (Qe @ ze.T).max(1).values)
        print(f"점검1 연도 {y + 1}/{C.n_years}", flush=True)
    pre = PatchTSTPretrain(patch_len=PATCH)
    pre.load_state_dict(torch.load(PRE_RES / f"p{PATCH}" / "best.pt", map_location="cpu")["model"])
    pre.to(DEV).eval()
    rec = []
    for i in range(len(Q)):  # 표본마다 같은 고정 가림
        _, (t, p, m) = pre(Q[i:i + 1].to(DEV), torch.Generator().manual_seed(SEED))
        mm = m[..., None].expand_as(t)
        rec.append(float((((p - t) ** 2) * mm).sum() / mm.sum()))
    rec = np.array(rec)
    df, k0 = [], 0
    for k in names:
        n = len(queries[k])
        df.append(pd.DataFrame({"set": k, "nn_dist": best_d[k0:k0 + n].numpy(), "nn_cos": best_s[k0:k0 + n].numpy(),
                                "recon_mse": rec[k0:k0 + n]}))
        k0 += n
    return pd.concat(df, ignore_index=True)


def judge(d: pd.DataFrame) -> pd.DataFrame:
    ref = d[d.set == "참조"]
    rows = []
    for k, g in d[d.set != "참조"].groupby("set", sort=False):
        r = {"set": k}
        for col, worse_high in [("nn_dist", True), ("recon_mse", True), ("nn_cos", False)]:
            v = g[col].median()
            pct = (ref[col] < v).mean() * 100 if worse_high else (ref[col] > v).mean() * 100  # 참조 중 이보다 나은 비율
            r[col] = round(v, 4)
            r[f"{col}_pct"] = round(pct, 1)
            r[f"{col}_판정"] = "분포 밖" if pct > 99 else ("경계" if pct > 95 else "분포 안")
        rows.append(r)
    ref_row = {"set": "참조(중앙값)", **{c: round(ref[c].median(), 4) for c in ["nn_dist", "nn_cos", "recon_mse"]}}
    return pd.DataFrame([ref_row, *rows])


# ---------- 점검 2 ----------
def boot_auc(y, s, rng):
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C = Climate()
    table = plant_table()
    pf = PlantFeatures(table).fit_scaler(train_rows(-1))
    raw = np.asarray(C.clim[:, SEOUL]).astype(np.float32)  # (10, 365, 4)
    sc = scenarios(raw)
    inputs = {k: to_input(C, v, SEOUL) for k, v in sc.items()}
    for k, v in sc.items():
        print(k, "1월 평균 기온 %.1f℃ 습도 %.0f%% · 7월 기온 %.1f℃ 습도 %.0f%%" % (
            v[:, :31, 1].mean(), v[:, :31, 3].mean(), v[:, 181:212, 1].mean(), v[:, 181:212, 3].mean()))

    # 참조: 사전학습 검증 칸 × 연도 무작위 2,000
    _, val = split(C.n_src, C.n_years)
    rng = np.random.default_rng(SEED)
    ref = val[rng.choice(len(val), N_REF, replace=False)]
    val_cells = np.unique(val[:, 0])
    pool = np.setdiff1d(np.arange(C.n_src), np.append(val_cells, SEOUL))
    ref_in = torch.cat([C.batch(ref[i:i + 512, 0], ref[i:i + 512, 1]) for i in range(0, len(ref), 512)])

    models = []
    for s in SEEDS:
        m = build_model("none", PATCH, pf.dim, C)
        m.load_state_dict(torch.load(model_dir(-1, s) / "best.pt", map_location="cpu")["model"])
        models.append(m.to(DEV).eval())

    d1 = check1(C, models[0], {"참조": ref_in, **inputs}, ref, pool)
    d1.to_csv(OUT / "check1_raw.csv", index=False, encoding="utf-8-sig")
    j1 = judge(d1)
    j1.to_csv(OUT / "check1.csv", index=False, encoding="utf-8-sig")
    print(j1.to_string())

    # 점검 2: 설정 2 모델 점수 (시드 × 시나리오 × 식물), 10개 연도 평균
    with torch.no_grad():
        P = {k: np.stack([torch.sigmoid(m.climate(x.to(DEV)) @ m.plant(pf.tensor().to(DEV)).T * m.scale() + m.b)
                          .mean(0).cpu().numpy() for m in models]) for k, x in inputs.items()}
    thr = pd.read_csv(EVAL / "setting2_per_plant.csv").pivot(index="seed", columns="row", values="thr").to_numpy()
    rows = []
    for k, p in P.items():
        rows.append(pd.DataFrame({"scenario": k, "식물명": table["식물명"], "score": p.mean(0),
                                  "suitable_frac_seeds": (p >= thr).mean(0)}))
    pd.concat(rows).to_csv(OUT / "scenario_scores.csv", index=False, encoding="utf-8-sig")

    res = {}
    s0 = P["S0_실외"].mean(0)
    seoul_min = float(raw[..., 0].min())
    yA = (seoul_min >= table["temp_limit_lower_min"].to_numpy()).astype(int)
    res["A_실외_겨울견딤_AUC"] = {"auc": round(roc_auc_score(yA, s0), 3), "ci": boot_auc(yA, s0, np.random.default_rng(SEED)),
                              "서울_10년_최저": round(seoul_min, 1), "견딤_식물수": int(yA.sum())}
    s2 = P["S2_난방_측정습도"].mean(0)
    yB = table["cultivation_type"].isin(["실내", "실내·실외"]).to_numpy().astype(int)
    res["B_난방거실_실내식물_AUC"] = {"auc": round(roc_auc_score(yB, s2), 3), "ci": boot_auc(yB, s2, np.random.default_rng(SEED)),
                                "실내형_식물수": int(yB.sum())}
    gain = P["S3_난방_가습"].mean(0) - P["S1_난방_건조"].mean(0)
    r = spearmanr(gain, table["humidity_min"])
    res["C_가습이득_vs_적정습도"] = {"rho": round(float(r.statistic), 3), "p": round(float(r.pvalue), 4),
                               "가습이득_평균": round(float(gain.mean()), 4)}
    res["D_적합식물수"] = {k: round(float((p >= thr).sum(1).mean()), 1) for k, p in P.items()}
    res["D_평균점수"] = {k: round(float(p.mean()), 4) for k, p in P.items()}
    (OUT / "check2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
