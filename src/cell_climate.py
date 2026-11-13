"""기상 칸마다 10년(2016–2025) 기후 요약을 만든다. (백로그 5번 단순 통계 베이스라인의 입력)

지표(모두 해마다 계산한 뒤 10년 평균):
  추위  cold_ext : 연 최저 일최저기온 — USDA 식물 내한성 지도(2023) 정의 "mean annual extreme minimum temperature"
        bio6     : 가장 추운 달의 일최저기온 평균 — WorldClim BIO6 정의(O'Donnell & Ignizio 2012), 10-01 파일럿(ρ 0.85)과 같은 지표
  더위  hot_ext  : 연 최고 일최고기온
        bio5     : 가장 더운 달의 일최고기온 평균 — WorldClim BIO5
  평균  t_annual : 연평균 기온(BIO1) — 파일럿의 '적정온도 vs 연평균' 비교 재현용
  습도  rh_annual: 일평균 상대습도의 연평균
        rh_dry_month: 가장 건조한 달의 상대습도 평균(BIO6과 같은 '가장 나쁜 달' 방식)
        rh_p05   : 일평균 상대습도의 연 5퍼센타일(건조한 날들)
  습도 지표는 원예 습도 기준과의 관계를 다룬 선행 근거를 찾지 못해, 미리 정한 세 가지를 모두 보고한다(고르지 않음).

달은 2월 29일을 뺀 365일 달력 기준(평년 월 길이)으로 나눈다.
입력: data/weather/daily_features/<연도>/pNN.parquet
출력: data/weather/cell_climate_10y.parquet (src_id + 위 지표, 연도별 평균)
실행: python -m src.cell_climate
"""

import numpy as np
import pandas as pd

from src.gee_annual_extremes import OUT

YEARS = range(2016, 2026)
MONTH_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAY_TO_MONTH = np.repeat(np.arange(1, 13), MONTH_LEN)  # day 1..365 → month


def yearly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(month=DAY_TO_MONTH[df["day"].to_numpy() - 1])
    g = df.groupby(["src_id", "year"])
    y = pd.DataFrame({"cold_ext": g["t_min"].min(), "hot_ext": g["t_max"].max(), "t_annual": g["t_mean"].mean(),
                      "rh_annual": g["rh_mean"].mean(), "rh_p05": g["rh_mean"].quantile(0.05)})
    m = df.groupby(["src_id", "year", "month"]).agg(tmin=("t_min", "mean"), tmax=("t_max", "mean"),
                                                    rh=("rh_mean", "mean"))
    mg = m.groupby(level=["src_id", "year"])
    y["bio6"], y["bio5"], y["rh_dry_month"] = mg["tmin"].min(), mg["tmax"].max(), mg["rh"].min()
    return y.reset_index()


def main() -> None:
    parts = []
    for y in YEARS:
        for f in sorted((OUT / "daily_features" / str(y)).glob("p*.parquet")):
            parts.append(yearly(pd.read_parquet(f)))
        print(y, "완료", flush=True)
    yr = pd.concat(parts, ignore_index=True)
    n_years = yr.groupby("src_id").size()
    if not (n_years == len(YEARS)).all():
        raise SystemExit(f"10년이 아닌 칸 {int((n_years != len(YEARS)).sum())}개")
    clim = yr.drop(columns="year").groupby("src_id").mean().astype("float32").reset_index()
    clim.to_parquet(OUT / "cell_climate_10y.parquet", index=False)
    print(len(clim), "칸")
    print(clim.describe().round(1).to_string())


if __name__ == "__main__":
    main()
