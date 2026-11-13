"""백로그 6번: 데이터 연결 — 식물·칸·연도를 모델 입력 텐서로 만든다. (설계: docs/model_design.md 2-2·2-4·2-5·2-6)

1) 기후 캐시(build_cache, 한 번만 실행)
   daily_features(연도 10개 × 조각 17개 parquet)를 배열 하나로 모은다.
     clim_f16.npy  : (연도 10, 칸 84,340, 날 365, 변수 4) float16 ≈ 2.46 GB — t_min, t_mean, t_max, rh_mean
     daylen_f16.npy: (칸, 365) float16 — 낮 길이는 위도·날짜로만 정해져 연도와 무관(2016·2024 대조 차이 0)
   일사량(ssrd)은 넣지 않는다(광량은 규칙 담당, 09-30 설계). 학습 중에는 메모리 맵으로 읽어 16 GB 메모리에 맞춘다.
   float16 해상도: 32~64 구간 간격 0.03(기온 최대 50.7℃) · 64~128 구간 간격 0.06(RH) — 측정 오차보다 작다.

2) 입력 1개 = 칸 1개 × 1개 연도(365일) × 채널 7개
   기온 3 + 습도 1(전체 칸·연도 평균·표준편차로 표준화 — 라벨 없는 기후값만 쓰므로 평가 종 정보 누설 없음)
   + 낮 길이(표준화) + 연중 위치 sin·cos(같은 날짜라도 남·북반구를 낮 길이로 구분, 09-30 설계).

3) 식물 입력(설계 2-4): 적정 온도·적정 습도·광량 관련 값 + 기후형·생육형 그룹·재배구분 원핫.
   한계온도 하한·상한은 평가 정답이라 넣지 않는다. 표준화 통계는 학습 겹 식물로만 맞춘다(fit_scaler).
   빈 값: light_hours_max 19개·humidity_max 1개는 원자료가 단일값('12', '60')이라 각 min으로 채운다.

4) 종 단위 5겹 분할(설계 2-6, Roberts et al. 2017): 묶음 = count_key(같은 종 두 식물은 같은 겹),
   기후형 4분류 비율을 겹마다 맞춘다(StratifiedKFold, seed 0). 결과를 보기 전에 data/dataset/folds.csv로 고정.

5) 배치 추출(PairSampler): 종을 고르게 뽑고(종 균형, 설계 2-5) → 그 종의 관측 칸 하나 → 10년 중 무작위 연도.
   의사 음성(손실 B, SINR)은 전체 육지 칸에서 균등 추출.

실행: python -m src.model_data   (캐시가 없으면 만들고, 분할 파일 저장, 점검 출력)
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.build_dataset import DS, eligible
from src.gee_annual_extremes import OUT, ROOT

YEARS = list(range(2016, 2026))
VARS = ["t_min", "t_mean", "t_max", "rh_mean"]
N_DAYS = 365
CACHE = OUT / "model_cache"
CLIM_F = CACHE / "clim_f16.npy"
CLIM_MEAN_F = CACHE / "clim_mean_f16.npy"  # 부품 제거 '10년 평균 1개 입력': (1, 칸, 365, 4) 날짜별 10년 평균
DAYLEN_F = CACHE / "daylen_f16.npy"
STATS_F = CACHE / "norm_stats.json"
FOLDS_F = DS / "folds.csv"
N_FOLDS = 5
SEED = 0
CLIMATE_COL = "기후형(열대/ 열대·아열대/아열대/온대)"


# ---------- 1) 기후 캐시 ----------
def build_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    n_src = len(pd.read_parquet(OUT / "cell_climate_10y.parquet", columns=["src_id"]))
    clim = np.lib.format.open_memmap(CLIM_F, mode="w+", dtype=np.float16, shape=(len(YEARS), n_src, N_DAYS, len(VARS)))
    daylen = np.zeros((n_src, N_DAYS), np.float16)
    s1 = np.zeros(len(VARS)); s2 = np.zeros(len(VARS)); cnt = 0
    for yi, y in enumerate(YEARS):
        seen = 0
        for f in sorted((OUT / "daily_features" / str(y)).glob("p*.parquet")):
            df = pd.read_parquet(f, columns=["src_id", "day", *VARS, "day_length_h"]).sort_values(["src_id", "day"])
            ids = df["src_id"].to_numpy()[::N_DAYS]
            if len(df) != len(ids) * N_DAYS or not (df["day"].to_numpy().reshape(-1, N_DAYS) == np.arange(1, 366)).all():
                raise SystemExit(f"{f}: 칸마다 365일이 아님")
            x = df[VARS].to_numpy(np.float64).reshape(len(ids), N_DAYS, len(VARS))
            clim[yi, ids] = x.astype(np.float16)
            s1 += x.sum((0, 1)); s2 += (x ** 2).sum((0, 1)); cnt += x.shape[0] * N_DAYS
            if yi == 0:
                daylen[ids] = df["day_length_h"].to_numpy().reshape(len(ids), N_DAYS)
            seen += len(ids)
        if seen != n_src:
            raise SystemExit(f"{y}: 칸 {seen}개 (기대 {n_src})")
        print(y, "완료", flush=True)
    clim.flush()
    np.save(DAYLEN_F, daylen)
    mean = s1 / cnt
    dl = daylen.astype(np.float64)
    stats = {"vars": VARS, "mean": mean.tolist(), "std": np.sqrt(s2 / cnt - mean ** 2).tolist(),
             "daylen_mean": float(dl.mean()), "daylen_std": float(dl.std()), "n_src": n_src, "years": YEARS}
    STATS_F.write_text(json.dumps(stats, indent=1))


def build_mean_cache() -> None:
    clim = np.load(CLIM_F, mmap_mode="r")
    out = np.lib.format.open_memmap(CLIM_MEAN_F, mode="w+", dtype=np.float16, shape=(1, *clim.shape[1:]))
    for i in range(0, clim.shape[1], 5000):
        out[0, i:i + 5000] = clim[:, i:i + 5000].astype(np.float32).mean(0).astype(np.float16)
    out.flush()


class Climate:
    """기후 캐시를 메모리 맵으로 열고 (칸, 연도) 묶음을 표준화된 (B, 365, 7) 텐서로 돌려준다.
    mean=True면 10년 평균 1개(연도 축 길이 1) — 표준화 통계는 같은 것(전체 칸·연도)을 쓴다."""

    def __init__(self, in_memory: bool = False, mean: bool = False):
        if mean and not CLIM_MEAN_F.exists():
            build_mean_cache()
        self.clim = np.load(CLIM_MEAN_F if mean else CLIM_F, mmap_mode=None if in_memory else "r")
        self.daylen = np.load(DAYLEN_F)
        st = json.loads(STATS_F.read_text())
        self.mean = np.asarray(st["mean"], np.float32)
        self.std = np.asarray(st["std"], np.float32)
        self.dl_mean, self.dl_std = st["daylen_mean"], st["daylen_std"]
        self.n_src, self.n_years = self.clim.shape[1], self.clim.shape[0]
        doy = 2 * np.pi * np.arange(N_DAYS) / N_DAYS
        self.season = np.stack([np.sin(doy), np.cos(doy)], -1).astype(np.float32)  # (365, 2)

    def batch(self, src: np.ndarray, year_idx: np.ndarray) -> torch.Tensor:
        x = (self.clim[year_idx, src].astype(np.float32) - self.mean) / self.std        # (B, 365, 4)
        dl = (self.daylen[src].astype(np.float32)[..., None] - self.dl_mean) / self.dl_std  # (B, 365, 1)
        se = np.broadcast_to(self.season, (len(src), N_DAYS, 2))
        return torch.from_numpy(np.concatenate([x, dl, se], -1))


# ---------- 3) 식물 입력 ----------
NUM_COLS = ["temp_optimal_min", "temp_optimal_max", "humidity_min", "humidity_max", "light_lux_min", "light_lux_max",
            "light_saturation_min", "light_saturation_max", "light_compensation_min", "light_compensation_max",
            "light_hours_min", "light_hours_max"]
CAT_COLS = {"climate": CLIMATE_COL, "group": "plant_group", "cultivation": "cultivation_type"}
LEAK_COLS = ["temp_limit_lower_min", "temp_limit_lower_max", "temp_limit_upper_min", "temp_limit_upper_max"]


def plant_table(version: str = "v_main", d: int = 25, cultivars: bool = True) -> pd.DataFrame:
    """학습 대상 식물 + 원예 입력. 같은 종 두 식물은 행 2개, 같은 count_key."""
    p = eligible(pd.read_csv(DS / "plants.csv"), version, d, cultivars)[["식물명", "count_key"]]
    h = pd.read_csv(ROOT / "data" / "processed" / "plants_normalized.csv").rename(columns={"식물명 (국명)": "식물명"})
    h["light_hours_max"] = h["light_hours_max"].fillna(h["light_hours_min"])
    h["humidity_max"] = h["humidity_max"].fillna(h["humidity_min"])
    t = p.merge(h[["식물명", *NUM_COLS, *CAT_COLS.values(), *LEAK_COLS]], on="식물명", how="left")
    if t[NUM_COLS + list(CAT_COLS.values())].isna().any().any():
        raise SystemExit("원예 입력에 빈 값")
    return t.reset_index(drop=True)


class PlantFeatures:
    """수치는 학습 겹 식물로 맞춘 평균·표준편차로 표준화, 범주는 전체 범주 목록으로 원핫(범주 목록은 라벨 아님)."""

    def __init__(self, table: pd.DataFrame, use_climate_type: bool = True):
        self.t = table
        cats = [c for k, c in CAT_COLS.items() if use_climate_type or k != "climate"]
        self.onehot = pd.get_dummies(table[cats].astype(str), dtype=np.float32).to_numpy()
        self.num = table[NUM_COLS].to_numpy(np.float32)
        self.mu = self.sd = None

    def fit_scaler(self, rows: np.ndarray) -> "PlantFeatures":
        self.mu = self.num[rows].mean(0)
        self.sd = self.num[rows].std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def tensor(self) -> torch.Tensor:
        if self.mu is None:
            raise RuntimeError("fit_scaler(학습 식물)를 먼저 호출")
        return torch.from_numpy(np.concatenate([(self.num - self.mu) / self.sd, self.onehot], 1))

    @property
    def dim(self) -> int:
        return self.num.shape[1] + self.onehot.shape[1]


# ---------- 4) 종 단위 분할 ----------
def make_folds(table: pd.DataFrame) -> pd.DataFrame:
    sp = table.groupby("count_key")[CLIMATE_COL].first().reset_index()
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    sp["fold"] = -1
    for k, (_, te) in enumerate(skf.split(sp, sp[CLIMATE_COL])):
        sp.loc[sp.index[te], "fold"] = k
    return table[["식물명", "count_key"]].merge(sp[["count_key", "fold"]], on="count_key")


# ---------- 5) 배치 추출 ----------
class PairSampler:
    """양성 짝(식물 행, 관측 칸, 연도) 추출. balanced=True면 종(count_key)을 균등하게 먼저 뽑는다."""

    def __init__(self, table: pd.DataFrame, rows: np.ndarray, version: str = "v_main", balanced: bool = True,
                 seed: int = 0):
        sc = pd.read_parquet(DS / "species_cells.parquet", columns=["count_key", "src_id", f"n_{version}"])
        sc = sc[sc[f"n_{version}"] > 0].drop_duplicates(["count_key", "src_id"])
        self.rng = np.random.default_rng(seed)
        self.t = table
        self.rows = np.asarray(rows)
        self.keys = np.unique(table.loc[self.rows, "count_key"])
        g = sc[sc["count_key"].isin(self.keys)].groupby("count_key")["src_id"]
        self.cells = {k: v.to_numpy() for k, v in g}
        self.rows_of = {k: self.rows[table.loc[self.rows, "count_key"].to_numpy() == k] for k in self.keys}
        sizes = np.array([len(self.cells[k]) for k in self.keys], float)
        self.p_key = None if balanced else sizes / sizes.sum()  # 균형 없음 = 기록(칸) 수에 비례

    def sample(self, b: int, n_years: int = len(YEARS)):
        ks = self.rng.choice(self.keys, b, p=self.p_key)
        plant = np.array([self.rng.choice(self.rows_of[k]) for k in ks])
        src = np.array([self.rng.choice(self.cells[k]) for k in ks])
        return plant, src, self.rng.integers(0, n_years, b)

    def sample_unique(self, n_years: int = len(YEARS)):
        """종마다 정확히 1개(손실 A — 묶음 안에 같은 종이 두 번 나오지 않게)."""
        ks = self.rng.permutation(self.keys)
        plant = np.array([self.rng.choice(self.rows_of[k]) for k in ks])
        src = np.array([self.rng.choice(self.cells[k]) for k in ks])
        return plant, src, self.rng.integers(0, n_years, len(ks))

    def random_cells(self, b: int, n_src: int, n_years: int = len(YEARS)):
        return self.rng.integers(0, n_src, b), self.rng.integers(0, n_years, b)


def main() -> None:
    if not CLIM_F.exists():
        build_cache()
    t = plant_table()
    folds = make_folds(t)
    folds.to_csv(FOLDS_F, index=False, encoding="utf-8-sig")
    print("겹별 식물/종 수:", folds.groupby("fold").agg(plants=("식물명", "size"), species=("count_key", "nunique")).to_dict())
    print(t.assign(fold=folds["fold"]).groupby(["fold", CLIMATE_COL]).size().unstack(fill_value=0).to_string())

    c = Climate()
    tr = np.flatnonzero(folds["fold"].to_numpy() != 0)
    pf = PlantFeatures(t).fit_scaler(tr)
    s = PairSampler(t, tr)
    plant, src, yr = s.sample(256)
    x = c.batch(src, yr)
    xn, yn = s.random_cells(256, c.n_src)
    print("식물 입력", tuple(pf.tensor().shape), "기후 배치", tuple(x.shape), "음성 배치", tuple(c.batch(xn, yn).shape))
    print("채널 평균", x.mean((0, 1)).numpy().round(2), "표준편차", x.std((0, 1)).numpy().round(2))
    assert not torch.isnan(x).any()
    assert set(t.loc[plant, "count_key"]).isdisjoint(set(t.loc[folds["fold"] == 0, "count_key"])), "평가 종이 학습 배치에 섞임"
    print("누설 점검 통과: 겹 0 평가 종은 학습 배치에 없음")


if __name__ == "__main__":
    main()
