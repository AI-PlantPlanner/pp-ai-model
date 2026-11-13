"""받은 일별 기상(data/weather/daily)을 모델 입력용 365일 형식으로 바꾼다. (백로그 3번 중 '일별 요약')

만드는 열 (원천 칸 src_id × 연도 × 1~365일):
  t_mean, t_min, t_max  : ℃ (ERA5-Land 일평균·시간값 최저·최고)
  rh_mean               : 상대습도(%) = 100·e(Td)/e(T), e(x)=6.1094·exp(17.625x/(243.04+x))
                          — Alduchov & Eskridge 1996, J Appl Meteorol 35:601–609 식 21(물 기준) 계수.
                          일평균 기온·이슬점으로 계산하므로 하루 안의 변동은 반영되지 않는다(한계로 기재).
  ssrd                  : 일사량 합(MJ/m²/일), 변환하지 않고 보관 — 광량은 규칙이 담당(09-30 결정)이라
                          모델 입력에는 쓰지 않는다. DLI 변환 계수는 출처마다 다르다(0.45×4.57≈2.06 vs 1.96).
  day_length_h          : 낮 길이(시간), 위도·날짜로 계산(천문식, Forsythe et al. 1995 CBM 모델, 일출~일몰 정의 p=0.8333°) — 남반구 계절
                          처리를 모델 단계에서 정할 수 있도록 함께 둔다. 날짜 이동은 하지 않는다.
윤년의 2월 29일은 뺀다(게임 입력이 365일 배열).

입력: data/weather/daily/<연도>/pNN.parquet, data/weather/daily_src_cells.csv
출력: data/weather/daily_features/<연도>/pNN.parquet (이미 있으면 건너뜀)
실행: python -m src.daily_features
"""

import numpy as np
import pandas as pd

from src.gee_annual_extremes import OUT

A, B, C = 6.1094, 17.625, 243.04


def sat_vp(t):
    return A * np.exp(B * t / (C + t))


def day_length_h(lat_deg, doy, p_deg=0.8333):
    """Forsythe et al. (1995) CBM 모델. p = 0.8333°: 태양 윗가장자리 + 대기 굴절 기준(일출~일몰, 기상청 일출·일몰과 같은 정의)."""
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))
    phi = np.arcsin(0.39795 * np.cos(theta))
    lat = np.radians(lat_deg)
    p = np.radians(p_deg)
    x = (np.sin(p) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    return 24 - (24 / np.pi) * np.arccos(np.clip(x, -1, 1))


def convert(df: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()
    df["year"] = df["date"].dt.year.astype("int16")
    # 2월 29일을 뺀 뒤의 1~365 일련번호
    doy = df["date"].dt.dayofyear
    leap = df["date"].dt.is_leap_year & (df["date"].dt.month > 2)
    df["day"] = (doy - leap.astype(int)).astype("int16")
    df["rh_mean"] = (100 * sat_vp(df["td_mean"]) / sat_vp(df["t_mean"])).clip(upper=100).astype("float32")
    lat = df["src_id"].map(src.set_index("src_id")["src_lat"])
    df["day_length_h"] = day_length_h(lat.to_numpy(), df["day"].to_numpy()).astype("float32")
    return df[["src_id", "year", "day", "t_mean", "t_min", "t_max", "rh_mean", "ssrd", "day_length_h"]]


def main() -> None:
    src = pd.read_csv(OUT / "daily_src_cells.csv")
    done = 0
    for path in sorted((OUT / "daily").glob("*/p*.parquet")):
        out = OUT / "daily_features" / path.parent.name / path.name
        if out.exists():
            continue
        df = pd.read_parquet(path)
        feat = convert(df, src)
        n_days = feat.groupby("src_id").size()
        if not (n_days == 365).all():
            raise SystemExit(f"{path}: 365일이 아닌 칸 {int((n_days != 365).sum())}개")
        out.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(out, index=False, compression="zstd")
        done += 1
        print(f"{path.parent.name}/{path.name}: {feat['src_id'].nunique()}칸, {out.stat().st_size / 1e6:.1f} MB")
    print("변환", done, "조각")


if __name__ == "__main__":
    main()
