"""격자 칸마다 하루 단위 기상(10년, 2016–2025)을 GEE에서 받아 parquet로 저장한다. (결정 E: 10년, 11-06 사용자 승인)

변수(ERA5-Land Daily Aggregated):
  t_mean / t_min / t_max  : temperature_2m / _min / _max (℃로 변환) — 일최저·최고는 시간값의 최저·최고
  td_mean                 : dewpoint_temperature_2m (℃) — 상대습도 계산용
  ssrd                    : surface_solar_radiation_downwards_sum (MJ/m²/일) — 일사량(DLI 환산용)
칸: data/weather/cells_v01_fill.csv의 land·filled 칸. filled는 50 km 안 가장 가까운 육지 칸(src_lat, src_lon) 값을 쓴다.

흐름: 조각(칸 CHUNK개 × 1년)마다 즉시 요청(computeFeatures)으로 받는다 — 칸 1,000개 × 1개월을 한 요청으로,
(블록 5개 × 12개월) 요청을 12개씩 동시에 보낸다. 2016–2025 169조각 ≈ 6.8시간(11-07 실측). 받은 즉시 float32 parquet(zstd)로 저장. 이미 저장된 조각은 건너뛴다.
(11-07 변경: 처음엔 Drive 일괄 내보내기를 썼으나 커뮤니티 등급에서 동시 1작업·조각당 약 60분이라 170조각 ≈ 7일.
 즉시 요청은 1,000칸 × 1년이 약 13초. 일괄 내보내기 함수는 기록용으로 남겨 둠.)

출력: data/weather/daily/<연도>/p<nn>.parquet (src_id, date, 변수 5개), data/weather/daily_src_cells.csv
      data/weather/gee_tasks.jsonl (작업별 사용량)
실행: python -m src.gee_daily --start 2016 --end 2025 [--only-chunk 0 --only-year 2025]
"""

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.gee_annual_extremes import DATASET, DRIVE_FOLDER, OUT, PROJECT, SCALE_M

CHUNK = 5000
BANDS = {"temperature_2m": "t_mean", "temperature_2m_min": "t_min", "temperature_2m_max": "t_max",
         "dewpoint_temperature_2m": "td_mean", "surface_solar_radiation_downwards_sum": "ssrd"}
K = 273.15


def src_cells() -> pd.DataFrame:
    c = pd.read_csv(OUT / "cells_v01_fill.csv")
    c = c[c["status"].isin(["land", "filled"])]
    src = c[["src_lat", "src_lon"]].drop_duplicates().sort_values(["src_lat", "src_lon"]).reset_index(drop=True)
    src["src_id"] = np.arange(len(src))
    src.to_csv(OUT / "daily_src_cells.csv", index=False)
    return src


def start_task(src: pd.DataFrame, year: int, name: str):
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.src_lon, r.src_lat]), {"src_id": int(r.src_id)})
                               for r in src.itertuples()])
    ic = ee.ImageCollection(DATASET).filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(list(BANDS))

    def per_day(im):
        d = im.date().format("YYYY-MM-dd")
        return im.sampleRegions(collection=fc, properties=["src_id"], scale=SCALE_M,
                                geometries=False).map(lambda f: f.set("date", d))

    task = ee.batch.Export.table.toDrive(collection=ic.map(per_day).flatten(), description=name,
                                         folder=DRIVE_FOLDER, fileNamePrefix=name, fileFormat="CSV",
                                         selectors=["src_id", "date"] + list(BANDS))
    task.start()
    return task


def log_task(task, name: str, n_cells: int, year: int, t0: float) -> dict:
    st = task.status()
    rec = {"name": name, "state": st["state"], "wall_s": round(time.time() - t0),
           "eecu_s": st.get("batch_eecu_usage_seconds"), "n_cells": n_cells, "years": [year, year],
           "error": st.get("error_message"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(OUT / "gee_tasks.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def month_eecu_h() -> float:
    """이번 달 기록된 EECU 사용량(시간). 커뮤니티 등급 월 150시간 한도 관리용."""
    month = time.strftime("%Y-%m")
    tot = 0.0
    for line in (OUT / "gee_tasks.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["at"].startswith(month):
            tot += r.get("eecu_s") or 0
    return tot / 3600


def fetch_and_delete(drv, name: str) -> pd.DataFrame:
    files = drv.files().list(q=f"name = '{name}.csv' and trashed = false", fields="files(id)").execute()["files"]
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, drv.files().get_media(fileId=files[0]["id"]), chunksize=64 * 1024 * 1024)
    done = False
    while not done:
        _, done = dl.next_chunk(num_retries=8)
    buf.seek(0)
    df = pd.read_csv(buf)
    for f in files:  # 우리가 만든 이 이름의 내보내기 파일만 지운다
        drv.files().delete(fileId=f["id"]).execute()
    return df


def to_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=BANDS)
    for c in ["t_mean", "t_min", "t_max", "td_mean"]:
        df[c] = (df[c] - K).astype("float32")
    df["ssrd"] = (df["ssrd"] / 1e6).astype("float32")
    df["src_id"] = df["src_id"].astype("int32")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["src_id", "date"]).reset_index(drop=True)


BLOCK = 1000
WORKERS = 12


def _month_block(args) -> pd.DataFrame:
    block, year, month = args
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.src_lon, r.src_lat]), {"src_id": int(r.src_id)})
                               for r in block.itertuples()])
    end = f"{year}-{month + 1:02d}-01" if month < 12 else f"{year + 1}-01-01"
    img = ee.ImageCollection(DATASET).filterDate(f"{year}-{month:02d}-01", end).select(list(BANDS)).toBands()
    samp = img.sampleRegions(collection=fc, properties=["src_id"], scale=SCALE_M, geometries=False)
    for attempt in range(6):
        try:
            wide = ee.data.computeFeatures({"expression": samp, "fileFormat": "PANDAS_DATAFRAME"})
            break
        except Exception as e:  # 일시적 거절(요청 과다 등)은 기다렸다 재시도
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))
    long = wide.melt(id_vars="src_id", var_name="col", value_name="v")
    long = long[long["col"] != "geo"]
    long["date"] = pd.to_datetime(long["col"].str[:8], format="%Y%m%d")
    long["band"] = long["col"].str[9:]
    return long.pivot_table(index=["src_id", "date"], columns="band", values="v").reset_index()


def fetch_interactive(src: pd.DataFrame, year: int) -> pd.DataFrame:
    jobs = [(src.iloc[b:b + BLOCK], year, m) for b in range(0, len(src), BLOCK) for m in range(1, 13)]
    with ThreadPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_month_block, jobs))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2025)
    a = ap.parse_args()

    ee.Initialize(project=PROJECT)
    src = src_cells()
    n_chunks = (len(src) + CHUNK - 1) // CHUNK
    todo = [(y, i) for y in range(a.start, a.end + 1) for i in range(n_chunks)
            if not (OUT / "daily" / str(y) / f"p{i:02d}.parquet").exists()]
    print(f"원천 칸 {len(src)}개, 조각 {n_chunks}개 × {a.end - a.start + 1}년, 남은 조각 {len(todo)}개", flush=True)
    t_all = time.time()
    for k, (y, i) in enumerate(todo, 1):
        t0 = time.time()
        part = src.iloc[i * CHUNK:(i + 1) * CHUNK]
        tab = to_table(fetch_interactive(part, y))
        n_days = tab.groupby("src_id").size()
        days_in_year = 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365
        bad = int((n_days != days_in_year).sum())
        path = OUT / "daily" / str(y) / f"p{i:02d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tab.to_parquet(path, index=False, compression="zstd")
        rec = {"name": f"daily_{y}_p{i:02d}", "mode": "interactive", "wall_s": round(time.time() - t0),
               "n_cells": len(part), "n_cells_got": int(tab["src_id"].nunique()), "n_cells_bad_days": bad,
               "years": [y, y], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(OUT / "gee_tasks.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        el = time.time() - t_all
        print(f"[{k}/{len(todo)}] {y}/p{i:02d}: {rec['n_cells_got']}/{len(part)}칸, 날짜 이상 {bad}칸, "
              f"{rec['wall_s']}초, {path.stat().st_size / 1e6:.1f} MB, 남은 예상 {el / k * (len(todo) - k) / 60:.0f}분",
              flush=True)


if __name__ == "__main__":
    main()
