"""추천 계산(고친 A)을 ONNX로 내보낸다 — 게임(Unity)에서 오프라인 실행용. (11-12 사용자 결정 ①)

신경망 모델을 넘기지 않는 이유: 야생 기후로 학습한 모델에 실내 그래프를 직접 넣으면 실패(docs 3-7).
여기서 내보내는 것은 src/recommend.py와 같은 계산이다:
  원예 규칙(src/teacher_scoring.py의 게이트·감점·그룹 가중치) + 식물별 값(원예 범위, 모델 건조 한계)을 상수로 넣은 계산.
규칙의 정의는 teacher_scoring.py 한 곳이고, 이 파일은 같은 식을 텐서 연산으로 옮긴 것이다.
같은지 검증: verify()가 무작위·시나리오 입력 수천 개에서 Python 원본(recommend.recommend)과 ONNX 결과를 대조한다.

입력 (B = 놓는 위치 수):
  t_min, t_max, t_mean, rh_mean, light_lux_mean : float32 [B, 365]
  context : int64 [B]  (0 = 실내, 1 = 실외)
출력:
  score, light, temp, humidity : float32 [B, P]  (P = 판단 가능한 식물 수, 순서는 plants.json)
  gate : int64 [B, P]  (0 없음, 1 재배구분 불일치, 2 최저기온 < 한계 하한, 3 최고기온 > 한계 상한)
  suitable : bool [B, P]  (gate == 0 이고 score > 40)
  warn_dry : bool [B, P]  (탈락 아님 이고 건조한 날 습도 < 모델 건조 한계)
  room : float32 [B, 6]  (t_min_year, t_max_year, t_mean, rh_mean, rh_dry_p05, light_lux_mean)
실행: python -m src.export_recommend_onnx [범위표 경로]
출력: results/tolerance/for_game/recommend_v1.onnx, plants.json(식물 순서·코드표), onnx_verify.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config, recommend
from src.gee_annual_extremes import ROOT
from src.teacher_scoring import REQUIRED_COLUMNS

OUT = ROOT / "results" / "tolerance" / "for_game"
CTX = {config.CULTIVATION_INDOOR: 0, config.CULTIVATION_OUTDOOR: 1}
GATE_CODES = {0: None, 1: "재배구분 불일치", 2: "기간 최저기온이 한계온도 하한 미만", 3: "기간 최고기온이 한계온도 상한 초과"}


def plant_constants(plants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ok = plants[list(REQUIRED_COLUMNS)].notna().all(axis=1) & (plants[config.NORM_CULTIVATION_TYPE] != config.CULTIVATION_UNKNOWN)
    p = plants[ok].reset_index(drop=True)
    f = lambda c: torch.tensor(p[c].to_numpy(float), dtype=torch.float32)
    lux_min, lux_max = f(config.NORM_LIGHT_LUX_MIN), f(config.NORM_LIGHT_LUX_MAX)
    comp = p[config.NORM_LIGHT_COMPENSATION_MIN].to_numpy(float)
    sat = p[config.NORM_LIGHT_SATURATION_MAX].to_numpy(float)
    r2 = config.SCORE_LIGHT_RATIO_LEVEL2
    lo2 = np.where(~np.isnan(comp) & (comp > 0) & (comp < lux_min.numpy()), np.maximum(r2, lux_min.numpy() / np.where(comp > 0, comp, 1)), r2)
    hi2 = np.where(~np.isnan(sat) & (sat > lux_max.numpy()) & (lux_max.numpy() > 0), np.maximum(r2, sat / np.where(lux_max.numpy() > 0, lux_max.numpy(), 1)), r2)
    hmax = p[config.NORM_HUMIDITY_MAX].fillna(100.0).to_numpy(float)
    w = np.array([config.GROUP_WEIGHTS.get(g, config.DEFAULT_GROUP_WEIGHTS) for g in p[config.NORM_PLANT_GROUP]], float)
    w = w / w.sum(1, keepdims=True)
    ct = p[config.NORM_CULTIVATION_TYPE].map({config.CULTIVATION_INDOOR: 0, config.CULTIVATION_OUTDOOR: 1,
                                              config.CULTIVATION_MIXED: 2}).to_numpy()
    c = {"lux_min": lux_min, "lux_max": lux_max, "lo2": torch.tensor(lo2, dtype=torch.float32),
         "hi2": torch.tensor(hi2, dtype=torch.float32),
         "omin": f(config.NORM_TEMP_OPTIMAL_MIN), "omax": f(config.NORM_TEMP_OPTIMAL_MAX),
         "lim_lo": f(config.NORM_TEMP_LIMIT_LOWER), "lim_hi": f(config.NORM_TEMP_LIMIT_UPPER),
         "hmin": f(config.NORM_HUMIDITY_MIN), "hmax": torch.tensor(hmax, dtype=torch.float32),
         "w": torch.tensor(w, dtype=torch.float32), "ctype": torch.tensor(ct, dtype=torch.int64),
         "dry": torch.tensor(p["dry_lim"].fillna(-1e9).to_numpy(float), dtype=torch.float32)}
    return p, c


def by_distance(d, e2, e3):
    """teacher_scoring._score_by_distance와 같은 식(분기 순서 그대로)."""
    L2 = config.SCORE_LEVEL2_END
    s2 = torch.where(e2 <= 0, torch.full_like(d, L2), 100.0 - (100.0 - L2) * d / torch.where(e2 > 0, e2, torch.ones_like(e2)))
    s3 = torch.where(e3 <= e2, torch.zeros_like(d), L2 * (1 - (d - e2) / torch.where(e3 > e2, e3 - e2, torch.ones_like(e2))))
    s = torch.where(d <= e2, s2, s3)
    s = torch.where(d >= e3, torch.zeros_like(d), s)
    return torch.where(d <= 0, torch.full_like(d, 100.0), s)


class RecommendNet(torch.nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        for k, v in c.items():
            self.register_buffer(k, v)

    def forward(self, t_min, t_max, t_mean, rh_mean, light_lux_mean, context):
        tmin_y = t_min.min(1).values[:, None]
        tmax_y = t_max.max(1).values[:, None]
        T = t_mean.mean(1)[:, None]
        H = rh_mean.mean(1)[:, None]
        Lx = light_lux_mean.mean(1)[:, None]
        srt = torch.sort(rh_mean, dim=1).values  # numpy percentile(5, linear): 위치 0.05·(n−1)
        pos = 0.05 * (rh_mean.shape[1] - 1)
        i0 = int(pos)
        p05 = (srt[:, i0] + (pos - i0) * (srt[:, i0 + 1] - srt[:, i0]))[:, None]

        # 광량: 적정범위 대비 배율(로그 보간)
        L2 = config.SCORE_LEVEL2_END
        below = Lx < self.lux_min
        ratio = torch.where(below, self.lux_min / Lx.clamp(min=1e-9), Lx / self.lux_max)
        r2 = torch.where(below, self.lo2, self.hi2)
        r3 = r2 * (config.SCORE_LIGHT_RATIO_LEVEL3 / config.SCORE_LIGHT_RATIO_LEVEL2)
        lr, l2, l3 = torch.log(ratio.clamp(min=1e-9)), torch.log(r2), torch.log(r3)
        ls = torch.where(lr <= l2, 100.0 - (100.0 - L2) * lr / l2, L2 * (1 - (lr - l2) / (l3 - l2)))
        ls = torch.where(lr >= l3, torch.zeros_like(ls), ls)
        ls = torch.where(ratio <= 1, torch.full_like(ls, 100.0), ls)
        ls = torch.where(Lx <= 0, torch.zeros_like(ls), ls)
        ls = torch.where((Lx >= self.lux_min) & (Lx <= self.lux_max), torch.full_like(ls, 100.0), ls)

        # 온도
        band = torch.clamp(self.omax - self.omin, min=config.SCORE_TEMP_MIN_BAND)
        low = T < self.omin
        d = torch.where(low, self.omin - T, T - self.omax)
        e3 = torch.where(low, self.omin - self.lim_lo, self.lim_hi - self.omax)
        e2 = torch.minimum(band, e3 * (1 - config.SCORE_TEMP_LEVEL3_MIN_RATIO))
        ts = by_distance(d, e2, e3)
        ts = torch.where((T >= self.omin) & (T <= self.omax), torch.full_like(ts, 100.0), ts)

        # 습도
        m = config.SCORE_HUMIDITY_MIN_BAND
        lowh = H < self.hmin
        dh = torch.where(lowh, self.hmin - H, H - self.hmax)
        lvl2 = torch.minimum(torch.full_like(self.hmin, config.SCORE_HUMIDITY_LEVEL2_LOWER), self.hmin - m)
        up_edge = torch.where(self.hmax >= config.SCORE_HUMIDITY_LEVEL2_UPPER,
                              torch.full_like(self.hmax, config.SCORE_HUMIDITY_LEVEL2_UPPER_EXCEPTION),
                              torch.full_like(self.hmax, config.SCORE_HUMIDITY_LEVEL2_UPPER))
        e2h = torch.where(lowh, self.hmin - lvl2, torch.clamp(up_edge - self.hmax, min=0.0))
        hs = by_distance(dh, e2h, e2h + m)
        hs = torch.where((H >= self.hmin) & (H <= self.hmax), torch.full_like(hs, 100.0), hs)

        # 게이트(원예 확정 0점 기준 2가지) — 우선순위는 teacher_scoring과 같음
        ctx = context[:, None]
        g_cult = (self.ctype != 2) & (self.ctype != ctx)
        g_cold = tmin_y < self.lim_lo
        g_heat = tmax_y > self.lim_hi
        zero = torch.zeros_like(ls, dtype=torch.int64)
        gate = torch.where(g_cult, zero + 1, torch.where(g_cold, zero + 2, torch.where(g_heat, zero + 3, zero)))
        gated = gate > 0
        score = ls * self.w[:, 0] + ts * self.w[:, 1] + hs * self.w[:, 2]
        z = torch.zeros_like(score)
        score, ls, ts, hs = (torch.where(gated, z, x) for x in (score, ls, ts, hs))
        suitable = (~gated) & (score > recommend.SUITABLE_MIN)
        warn = (~gated) & (p05 < self.dry)
        room = torch.cat([tmin_y, tmax_y, T, H, p05, Lx], 1)
        return score, ls, ts, hs, gate, suitable, warn, room


NAMES_IN = ["t_min", "t_max", "t_mean", "rh_mean", "light_lux_mean", "context"]
NAMES_OUT = ["score", "light", "temp", "humidity", "gate", "suitable", "warn_dry", "room"]


def test_inputs(n: int, seed: int = 0) -> dict:
    """서울 시나리오 5종 × 10년(src/indoor_check) + 무작위 변형 + 무작위 광량·재배구분."""
    from src.indoor_check import SEOUL, scenarios
    from src.model_data import Climate
    rng = np.random.default_rng(seed)
    base = np.concatenate(list(scenarios(np.asarray(Climate().clim[:, SEOUL]).astype(np.float32)).values()))  # (50,365,4)
    pick = base[rng.integers(0, len(base), n)].copy()
    pick[..., :3] += rng.normal(0, 6, (n, 1, 1)).astype(np.float32)      # 기온 평행 이동 → 한계 근처까지
    pick[..., 3] = np.clip(pick[..., 3] * rng.uniform(0.3, 1.5, (n, 1)), 0, 100)
    light = np.exp(rng.uniform(np.log(5), np.log(100000), (n, 1))) * rng.uniform(0.5, 1.5, (n, 365))
    return {"t_min": pick[..., 0], "t_max": pick[..., 2], "t_mean": pick[..., 1], "rh_mean": pick[..., 3],
            "light_lux_mean": light.astype(np.float32), "context": rng.integers(0, 2, n).astype(np.int64)}


def verify(onnx_path: Path, plants: pd.DataFrame, p: pd.DataFrame, n: int = 2000) -> dict:
    import onnxruntime as ort
    x = test_inputs(n)
    sess = ort.InferenceSession(str(onnx_path))
    out = dict(zip(NAMES_OUT, sess.run(None, x)))
    max_diff = {"score": 0.0, "light": 0.0, "temp": 0.0, "humidity": 0.0}
    mism = {"gate": 0, "suitable": 0, "warn_dry": 0}
    inv = {0: config.CULTIVATION_INDOOR, 1: config.CULTIVATION_OUTDOOR}
    for b in range(n):
        env, extra = recommend.summarize(x["t_min"][b], x["t_max"][b], x["rh_mean"][b], x["light_lux_mean"][b],
                                         inv[int(x["context"][b])], x["t_mean"][b])
        ref = recommend.recommend(plants, env, extra).set_index("식물명").loc[p[config.COL_NAME]]
        for k, col in [("score", "점수"), ("light", "광량"), ("temp", "온도"), ("humidity", "습도")]:
            max_diff[k] = max(max_diff[k], float(np.abs(out[k][b] - ref[col].to_numpy()).max()))
        g = np.array([GATE_CODES[int(v)] for v in out["gate"][b]], dtype=object)
        mism["gate"] += int((g != ref["탈락_사유"].to_numpy(dtype=object)).sum())
        mism["suitable"] += int((out["suitable"][b] != (ref["탈락_사유"].isna() & (ref["점수"] > recommend.SUITABLE_MIN)).to_numpy()).sum())
        mism["warn_dry"] += int((out["warn_dry"][b] != (ref["모델_건조_경고"].notna() & ref["탈락_사유"].isna()).to_numpy()).sum())
    # ONNX는 float32, 원본은 float64 → 허용 오차 0.01점
    return {"n_cases": n, "n_plants": len(p), "max_abs_diff": {k: round(v, 5) for k, v in max_diff.items()},
            "mismatch_counts": mism, "pass": all(v <= 0.01 for v in max_diff.values()) and not any(mism.values())}


def _num(v):
    return None if pd.isna(v) else round(float(v), 1)


def main() -> None:
    if len(sys.argv) > 1:
        recommend.TOL = Path(sys.argv[1])
    plants = recommend.load_plants()
    p, c = plant_constants(plants)
    net = RecommendNet(c).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "recommend_v1.onnx"
    x = test_inputs(4)
    args = tuple(torch.from_numpy(x[k]) for k in NAMES_IN)
    torch.onnx.export(net, args, str(path), input_names=NAMES_IN, output_names=NAMES_OUT, opset_version=17,
                      dynamic_axes={k: {0: "batch"} for k in NAMES_IN + NAMES_OUT}, dynamo=False)
    meta = {"plants": [{"index": i, "name": r[config.COL_NAME], "scientific_name": str(r[config.COL_SCIENTIFIC_NAME]).replace("?", " "),
                        "dry_limit_model": None if pd.isna(r["dry_lim"]) else round(float(r["dry_lim"]), 1),
                        # LLM 템플릿용 원예 값(11-13). 한계온도는 탈락 판정과 같은 열(teacher_scoring.is_gated)
                        "hort": {"cultivation": r[config.NORM_CULTIVATION_TYPE],
                                 "temp_limit_lower": _num(r[config.NORM_TEMP_LIMIT_LOWER]),
                                 "temp_limit_upper": _num(r[config.NORM_TEMP_LIMIT_UPPER]),
                                 "temp_optimal": [_num(r[config.NORM_TEMP_OPTIMAL_MIN]), _num(r[config.NORM_TEMP_OPTIMAL_MAX])],
                                 "humidity_optimal": [_num(r[config.NORM_HUMIDITY_MIN]), _num(r[config.NORM_HUMIDITY_MAX])],
                                 "light_lux_optimal": [_num(r[config.NORM_LIGHT_LUX_MIN]), _num(r[config.NORM_LIGHT_LUX_MAX])]}}
                       for i, r in p.iterrows()],
            "gate_codes": GATE_CODES, "context_codes": {"실내": 0, "실외": 1}, "suitable_rule": "gate == 0 and score > 40",
            "room_columns": ["t_min_year", "t_max_year", "t_mean", "rh_mean", "rh_dry_p05", "light_lux_mean"],
            "days": 365}  # 전달 파일에는 로컬 경로·용량을 넣지 않음(11-13 사용자 요청)
    (OUT / "plants.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    res = verify(path, plants, p)
    print("ONNX 크기(KB)", round(path.stat().st_size / 1024, 1))
    (OUT / "onnx_verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
