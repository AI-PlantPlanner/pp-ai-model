"""백로그 6번 3단계: 기후 ↔ 식물 정렬 학습 — 손실 A(CLIP InfoNCE) vs B(SINR L_AN-full) 비교 (설계 docs/model_design.md 2-1, 3-3).

모델
  기후 인코더 = 사전학습 PatchTST 인코더(results/pretrain/p{P}/best.pt) + 머리: 펼치기(채널×패치×D) → 드롭아웃 0.2 → 선형 → E.
    펼치기+선형 머리는 PatchTST 원문 지도학습 머리("a flatten layer with linear head", 본문 3.1), 드롭아웃 0.2는 원문 부록 A.1.4.
    공식 분류 머리는 마지막 패치만 쓰지만 우리는 1년 전체(어느 계절이 추운지)가 정보라 원문 지도학습 머리를 따른다.
  식물 인코더 = MLP 23 → 256 → 256 → E (ReLU, 드롭아웃 0.2). 폭 256은 SINR 위치 인코더 폭(원문 부록 C.1) 준용. E = 128(= D).
  점수 = cos(기후, 식물) × exp(t), t 학습형, 시작 1/0.07, 배율 상한 100 (CLIP 원문 2.5절·의사코드).
손실
  A. CLIP 대칭 교차 엔트로피(원문 의사코드). 한 묶음 = 학습 종마다 1개(종 중복 없음 → 같은 종이 서로 음성이 되는 일 없음).
     묶음 G개를 한 단계에 평균해 단계당 기후 수를 B와 맞춤.
  B. SINR L_AN-full(원문 식 5): 양성 λ·log ŷ + 같은 칸의 다른 식물 log(1−ŷ) + 무작위 칸의 모든 식물 log(1−ŷ), 식물 수로 평균.
     ŷ = sigmoid(점수 + b), b 학습형. 무작위 칸 = 전체 육지 칸 균등(원문 r ~ Unif). 같은 종의 다른 식물 행(디펜바키아·트로픽스노우)은 음성에서 제외.
     λ: 원문은 47,375종에 2048, 정하는 규칙 없음 → 미리 정한 후보 {1(가중 없음), R−1(양성 1개 = 같은 칸 음성 전체와 같은 무게)}를 내부 검증.
학습 일정
  선형 탐색 후 전체 미세조정 2단계(PatchTST 원문 4.2 — Kumar et al. 2022 인용, "two-step ... can outperform only fine-tuning").
  1단계: 인코더 고정, 머리·식물 MLP·t·b만 lr 5e-4(SINR 원문 부록 C.2). 2단계: 전체, 인코더 lr 1e-4(PatchTST 공식 기본), 나머지 5e-4.
  Adam. 단계 수는 고정 상한, 250단계마다 내부 검증 AUC로 최고 시점 저장(조기 종료 대용).
내부 검증(후보 선택, 설계 2-6)
  바깥 겹의 학습 종을 기후형 층화로 다시 나눠 20%를 내부 검증 종으로(seed 0). 평가 겹 종은 전혀 쓰지 않음.
  지표 = 내부 검증 종별 AUC(그 종의 관측 칸 vs 무작위 칸 5,000개, 칸마다 고정 무작위 연도)의 평균.
  원예 한계온도(평가 정답)는 선택에 쓰지 않음 → 결과를 보고 고르는 순환 방지.
실행: python -m src.align --fold 0 --loss A --patch 16
      python -m src.align --fold 0 --loss B --patch 16 --lam 1
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn

from src.gee_annual_extremes import ROOT
from src.gee_annual_extremes import OUT
from src.model_data import CLIMATE_COL, FOLDS_F, Climate, PairSampler, PlantFeatures, plant_table

ABLATIONS = ["none", "no_pretrain", "stats_mlp", "no_ctype", "mean_input", "unbalanced"]
STAT_COLS = ["cold_ext", "hot_ext", "t_annual", "rh_annual", "rh_p05", "bio6", "bio5", "rh_dry_month"]  # 베이스라인 지표 8개


class StatsInput:
    """'요약통계 + MLP'용 입력: 기후 대신 칸 번호를 넘기고 모델(StatsTower)이 표를 찾아 씀. 연도 없음(n_years 1)."""

    def __init__(self):
        d = pd.read_parquet(OUT / "cell_climate_10y.parquet").sort_values("src_id")
        x = d[STAT_COLS].to_numpy(np.float32)
        self.table = torch.from_numpy((x - x.mean(0)) / x.std(0))  # 라벨 없는 기후값만으로 표준화
        self.n_src, self.n_years = len(d), 1

    def batch(self, src, year_idx):
        return torch.as_tensor(np.asarray(src), dtype=torch.long)


def make_inputs(ablate: str):
    return StatsInput() if ablate == "stats_mlp" else Climate(mean=ablate == "mean_input")


def build_model(ablate: str, patch: int, d_plant: int, C) -> "AlignModel":
    return AlignModel(patch, d_plant, pretrained=ablate != "no_pretrain",
                      stats=C.table if ablate == "stats_mlp" else None)
from src.model_patchtst import PatchTSTEncoder

RES = ROOT / "results" / "align"
PRE = ROOT / "results" / "pretrain"
E = 128
INNER_FRAC = 0.2
N_NEG_EVAL = 5000
MAX_POS_EVAL = 1000
SEED = 0


class ClimateTower(nn.Module):
    def __init__(self, patch: int, pretrained: bool = True):
        super().__init__()
        self.enc = PatchTSTEncoder(patch)
        if pretrained:  # 부품 제거 '사전학습 없음'이면 무작위 초기값
            st = torch.load(PRE / f"p{patch}" / "best.pt", map_location="cpu")
            self.enc.load_state_dict({k[len("encoder."):]: v for k, v in st["model"].items() if k.startswith("encoder.")})
        self.head = nn.Sequential(nn.Flatten(1), nn.Dropout(0.2),
                                  nn.Linear(self.enc.n_ch * self.enc.n_patch * self.enc.d, E))

    def forward(self, x):
        return F.normalize(self.head(self.enc(x)), dim=-1)


class StatsTower(nn.Module):
    """부품 제거 '요약통계 + MLP'(설계 2-3 ②): 베이스라인 기후 지표 8개(칸별 10년 통계) → MLP(식물 쪽과 같은 폭).
    입력이 연도와 무관하므로 연도 인자는 무시. 인코더가 없어 2단계 일정의 1단계(고정)도 같은 매개변수를 학습."""

    def __init__(self, stats: torch.Tensor):
        super().__init__()
        self.register_buffer("x", stats)  # (칸, 8) 표준화
        self.enc = nn.Identity()
        self.net = nn.Sequential(nn.Linear(stats.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(256, E))

    def forward(self, src):
        return F.normalize(self.net(self.x[src]), dim=-1)


class PlantTower(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(256, E))

    def forward(self, p):
        return F.normalize(self.net(p), dim=-1)


class AlignModel(nn.Module):
    def __init__(self, patch: int, d_plant: int, pretrained: bool = True, stats: torch.Tensor | None = None):
        super().__init__()
        self.climate = StatsTower(stats) if stats is not None else ClimateTower(patch, pretrained)
        self.plant = PlantTower(d_plant)
        self.t = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.0))

    def scale(self):
        return self.t.clamp(max=np.log(100)).exp()

    def logits(self, zc, zp):
        return zc @ zp.T * self.scale()


def loss_clip(model, zc, zp):
    lg = model.logits(zc, zp)
    y = torch.arange(len(zc), device=zc.device)
    return (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2


def loss_an_full(model, zc_pos, zc_rand, zp, pos_col, same_key, lam):
    """zc_pos (B,E) 양성 기후, pos_col (B,) 그 식물 열, same_key (B,R) 같은 종 다른 행(음성 제외)."""
    s = model.logits(zc_pos, zp) + model.b
    sr = model.logits(zc_rand, zp) + model.b
    pos = F.one_hot(pos_col, zp.shape[0]).bool()
    neg = ~pos & ~same_key
    l_pos = -lam * F.logsigmoid(s)[pos]                               # (B,)
    l_neg = -(F.logsigmoid(-s) * neg).sum(1)                          # 같은 칸, 다른 식물
    l_rand = -F.logsigmoid(-sr).sum(1)                                # 무작위 칸, 모든 식물
    return ((l_pos + l_neg + l_rand) / zp.shape[0]).mean()


def inner_split(table, rows, seed=SEED):
    # 층 = 베이스라인과 같은 2분류(온대 / 열대·아열대). 4분류는 바깥 학습 종에 '열대·아열대'가 1종뿐인 겹이 있어 층화 불가.
    sp = table.loc[rows].groupby("count_key")[CLIMATE_COL].first().eq("온대")
    sss = StratifiedShuffleSplit(1, test_size=INNER_FRAC, random_state=seed)
    tr, va = next(sss.split(sp.index, sp.values))
    va_keys = set(sp.index[va])
    is_va = table.loc[rows, "count_key"].isin(va_keys).to_numpy()
    return rows[~is_va], rows[is_va]


class InnerEval:
    """내부 검증 종: 관측 칸(최대 1,000개) vs 무작위 칸 5,000개, 칸마다 고정 연도."""

    def __init__(self, table, va_rows, C, seed=SEED):
        rng = np.random.default_rng(seed)
        s = PairSampler(table, va_rows)
        self.items = []
        for k in s.keys:
            cells = s.cells[k]
            cells = rng.choice(cells, min(len(cells), MAX_POS_EVAL), replace=False)
            self.items.append((k, s.rows_of[k], cells, rng.integers(0, C.n_years, len(cells))))
        self.neg = (rng.integers(0, C.n_src, N_NEG_EVAL), rng.integers(0, C.n_years, N_NEG_EVAL))
        self.C = C

    @torch.no_grad()
    def embed(self, model, src, yr, dev, bs=1024):
        return torch.cat([model.climate(self.C.batch(src[i:i + bs], yr[i:i + bs]).to(dev)) for i in range(0, len(src), bs)])

    @torch.no_grad()
    def __call__(self, model, P, dev):
        model.eval()
        zp = model.plant(P.to(dev))
        zn = self.embed(model, *self.neg, dev)
        aucs = {}
        for k, rows, cells, yrs in self.items:
            zc = self.embed(model, cells, yrs, dev)
            for r in rows:  # 같은 종 두 식물이면 둘 다 평가
                sp = (zc @ zp[r]).cpu().numpy()
                sn = (zn @ zp[r]).cpu().numpy()
                aucs[int(r)] = roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn])
        model.train()
        return float(np.mean(list(aucs.values()))), aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--loss", choices=["A", "B"], required=True)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--lam", default="1", help="B의 λ: 숫자 또는 'R-1'")
    ap.add_argument("--lp_steps", type=int, default=1000)
    ap.add_argument("--ft_steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--final", action="store_true", help="내부 분할 없이 바깥 학습 종 전체로(선택 후 본 학습)")
    ap.add_argument("--ablate", choices=ABLATIONS, default="none", help="부품 제거 실험(설계 3단계 6)")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(a.seed)

    table = plant_table()
    folds = pd.read_csv(FOLDS_F)
    assert (folds["식물명"].to_numpy() == table["식물명"].to_numpy()).all()
    outer_tr = np.flatnonzero(folds["fold"].to_numpy() != a.fold)
    tr, va = (outer_tr, np.array([], int)) if a.final else inner_split(table, outer_tr)
    C = make_inputs(a.ablate)
    pf = PlantFeatures(table, use_climate_type=a.ablate != "no_ctype").fit_scaler(tr)
    P = pf.tensor()
    sampler = PairSampler(table, tr, balanced=a.ablate != "unbalanced", seed=a.seed)
    keys_tr = table.loc[tr, "count_key"].to_numpy()
    R = len(tr)
    lam = float(R - 1) if a.lam == "R-1" else float(a.lam)
    tag = f"{a.loss}_p{a.patch}" + (f"_lam{a.lam}" if a.loss == "B" else "") + (f"_s{a.seed}" if a.seed != SEED else "") \
        + (f"_abl-{a.ablate}" if a.ablate != "none" else "")
    out = RES / ("final" if a.final else "select") / f"fold{a.fold}" / tag
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(a.ablate, a.patch, pf.dim, C).to(dev)
    evaluator = None if a.final else InnerEval(table, va, C)
    col_of_row = {r: i for i, r in enumerate(tr)}
    same_key_full = torch.from_numpy(keys_tr[:, None] == keys_tr[None, :]).to(dev)  # (R,R)
    P_tr = P[tr].to(dev)
    G = max(1, a.bs // len(np.unique(keys_tr)))

    def step_batch():
        if a.loss == "A":
            losses = []
            for _ in range(G):
                rows, src, yr = sampler.sample_unique(C.n_years)
                zc = model.climate(C.batch(src, yr).to(dev))
                zp = model.plant(P_tr[[col_of_row[r] for r in rows]])
                losses.append(loss_clip(model, zc, zp))
            return torch.stack(losses).mean()
        rows, src, yr = sampler.sample(a.bs, C.n_years)
        cols = torch.tensor([col_of_row[r] for r in rows], device=dev)
        rs, ry = sampler.random_cells(a.bs, C.n_src, C.n_years)
        zc = model.climate(torch.cat([C.batch(src, yr), C.batch(rs, ry)]).to(dev))
        same = same_key_full[cols] & ~F.one_hot(cols, R).bool()
        return loss_an_full(model, zc[:a.bs], zc[a.bs:], model.plant(P_tr), cols, same, lam)

    enc_params = list(model.climate.enc.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("climate.enc.")]
    log = open(out / "log.jsonl", "w")
    best, best_step, step, t0 = -1.0, -1, 0, time.time()
    json.dump({"args": vars(a), "lambda": lam, "R_train_plants": R, "n_inner_val_plants": len(va), "G": G,
               "inner_val_plants": table.loc[va, "식물명"].tolist()}, open(out / "config.json", "w"),
              ensure_ascii=False, indent=1)
    for phase, n_steps in [("lp", a.lp_steps), ("ft", a.ft_steps)]:
        for p in enc_params:
            p.requires_grad_(phase == "ft")
        groups = [{"params": other, "lr": 5e-4}] + ([{"params": enc_params, "lr": 1e-4}] if phase == "ft" and enc_params else [])
        opt = torch.optim.Adam(groups)
        if phase == "lp":
            model.climate.enc.eval()  # 고정 단계: 인코더 BatchNorm·드롭아웃도 고정
        run = 0.0
        for _ in range(n_steps):
            if phase == "ft":
                model.train()
            loss = step_batch()
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); step += 1
            if step % a.eval_every == 0:
                rec = {"step": step, "phase": phase, "train_loss": run / a.eval_every,
                       "scale": float(model.scale()), "minutes": round((time.time() - t0) / 60, 1)}
                run = 0.0
                if evaluator is not None:
                    auc, _ = evaluator(model, P, dev)
                    if phase == "lp":
                        model.climate.enc.eval()
                    rec["inner_auc"] = auc
                    if auc > best:
                        best, best_step = auc, step
                        torch.save({"model": model.state_dict(), "step": step}, out / "best.pt")
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
    if a.final:
        torch.save({"model": model.state_dict(), "step": step}, out / "best.pt")
    json.dump({"best_inner_auc": best, "best_step": best_step, "total_minutes": round((time.time() - t0) / 60, 1)},
              open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
