"""백로그 6번 2단계: PatchTST 가리고 복원하기 사전학습 (설계 docs/model_design.md 2-3, 모델 src/model_patchtst.py).

데이터: 칸 84,340 × 연도 10 = 1년 그래프 843,400개(라벨 없음). 평가 종 칸도 포함되지만 어떤 종의 칸인지 정보는 쓰지 않음
        (기후값만 사용 — 추천 때도 전 세계 기후는 주어진 정보).
검증: 칸의 2%(seed 0)를 10년 모두 빼 둠(같은 칸의 다른 해가 학습에 들어가 검증이 쉬워지는 것 방지).
      검증 가림은 고정 생성기(seed 0)로 매번 같게 → 에포크 간 비교 가능.
비교 기준(같은 가림): ① 0 채우기 = 전체 평균으로 채우기 ② 가린 날을 앞뒤 안 가린 날로 선형 보간.
학습: Adam, lr 1e-4, one-cycle(공식 코드 fit_one_cycle 기본), 에포크 10(공식 코드 n_epochs_pretrain 기본값; 원문 100은
      훨씬 작은 데이터 기준), 배치 256(속도 측정: 패치 16 약 10분/에포크, 패치 8 약 24분/에포크, M4 MPS).
      검증 손실이 가장 낮은 에포크를 저장. 에포크마다 이어 하기용 상태 저장.
출력: results/pretrain/p{P}/ — log.jsonl, best.pt, last.pt, summary.json
실행: python -m src.pretrain --patch 16   (이어 하기 자동)
"""

import argparse
import json
import time

import numpy as np
import torch

from src.gee_annual_extremes import ROOT
from src.model_data import Climate
from src.model_patchtst import N_CH, PatchTSTPretrain

RES = ROOT / "results" / "pretrain"
VAL_FRAC = 0.02
SEED = 0


def split(n_src: int, n_years: int):
    rng = np.random.default_rng(SEED)
    val_cells = np.sort(rng.choice(n_src, int(n_src * VAL_FRAC), replace=False))
    is_val = np.zeros(n_src, bool)
    is_val[val_cells] = True
    src, yr = np.meshgrid(np.arange(n_src), np.arange(n_years), indexing="ij")
    pairs = np.stack([src.ravel(), yr.ravel()], 1)
    return pairs[~is_val[pairs[:, 0]]], pairs[is_val[pairs[:, 0]]]


@torch.no_grad()
def evaluate(model, C, val, dev, bs=1024, baselines=False):
    """고정 가림으로 가린 패치 MSE(채널별). baselines=True면 같은 가림에서 0 채우기·선형 보간도."""
    model.eval()
    gen = torch.Generator().manual_seed(SEED)
    se = torch.zeros(N_CH, dtype=torch.float64)
    se0 = torch.zeros(N_CH, dtype=torch.float64)
    sei = torch.zeros(N_CH, dtype=torch.float64)
    cnt = torch.zeros(N_CH, dtype=torch.float64)
    for i in range(0, len(val), bs):
        x = C.batch(val[i:i + bs, 0], val[i:i + bs, 1]).to(dev)
        _, (target, pred, mask) = model(x, gen)
        m = mask[..., None].expand_as(target)
        se += (((pred - target) ** 2) * m).sum((0, 2, 3)).cpu().double()
        cnt += m.sum((0, 2, 3)).cpu().double()
        if baselines:
            t, mk = target.cpu().numpy(), m.cpu().numpy().astype(bool)
            se0 += torch.from_numpy(((t ** 2) * mk).sum((0, 2, 3)))
            b, c, n, p = t.shape
            ts, ms = t.reshape(b * c, n * p), mk.reshape(b * c, n * p)
            days = np.arange(n * p)
            fill = ts.copy()
            for j in range(b * c):
                k = ~ms[j]
                fill[j, ms[j]] = np.interp(days[ms[j]], days[k], ts[j, k])
            sei += torch.from_numpy((((fill - ts) ** 2) * ms).reshape(b, c, -1).sum((0, 2)))
    model.train()
    out = {"mse": (se / cnt).tolist(), "mse_all": float(se.sum() / cnt.sum())}
    if baselines:
        out.update({"mse_zero": (se0 / cnt).tolist(), "mse_zero_all": float(se0.sum() / cnt.sum()),
                    "mse_interp": (sei / cnt).tolist(), "mse_interp_all": float(sei.sum() / cnt.sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    out = RES / f"p{a.patch}"
    out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED)

    C = Climate()
    train, val = split(C.n_src, C.n_years)
    model = PatchTSTPretrain(a.patch).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    steps_per_epoch = len(train) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
    start, best = 0, float("inf")
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, best = st["epoch"] + 1, st["best"]
        print("이어 하기: 에포크", start, flush=True)
    log = open(out / "log.jsonl", "a")
    if start == 0:
        ref = evaluate(model, C, val, dev, baselines=True)
        log.write(json.dumps({"epoch": -1, "note": "학습 전 + 비교 기준", **ref}) + "\n"); log.flush()
        print("학습 전", ref, flush=True)
        json.dump({"args": vars(a), "n_train": len(train), "n_val": len(val), "n_val_cells": len(val) // C.n_years,
                   "params": sum(p.numel() for p in model.parameters()), "device": dev,
                   "torch": torch.__version__, "baselines": {k: v for k, v in ref.items() if k.startswith("mse_")}},
                  open(out / "config.json", "w"), ensure_ascii=False, indent=1)

    for ep in range(start, a.epochs):
        rng = np.random.default_rng(SEED + ep)
        order = rng.permutation(len(train))
        t0, run = time.time(), 0.0
        for s in range(steps_per_epoch):
            idx = train[order[s * a.bs:(s + 1) * a.bs]]
            loss, _ = model(C.batch(idx[:, 0], idx[:, 1]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if (s + 1) % 500 == 0:
                log.write(json.dumps({"epoch": ep, "step": s + 1, "train_mse": run / 500,
                                      "lr": sched.get_last_lr()[0]}) + "\n"); log.flush()
                run = 0.0
        ev = evaluate(model, C, val, dev)
        rec = {"epoch": ep, "val": ev, "minutes": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
        if ev["mse_all"] < best:
            best = ev["mse_all"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": ev, "patch": a.patch}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best}, out / "last.pt")
    json.dump({"best_val_mse": best, "epochs": a.epochs}, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
