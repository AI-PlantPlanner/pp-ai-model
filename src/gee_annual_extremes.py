"""기상 격자 칸마다 해마다의 극값(연 최저 일최저기온, 연 최고 일최고기온)을 GEE에서 뽑는다. (결정 E, 연도 실험)

근거:
  - 지표 = 연 극한 최저기온의 기간 평균. USDA 식물 내한성 지도(2023, PRISM)가 1991–2020년
    "mean annual extreme minimum temperature"로 만든다(https://prism.oregonstate.edu/phzm).
  - 기간 30년 = WMO 기후 표준 평년(1991–2020). 10년 = 기록의 82%가 2015년 이후. 1년 = 게임팀 입력 연도(2025).
  - 자료 = ERA5-Land Daily Aggregated(GEE ECMWF/ERA5_LAND/DAILY_AGGR). 일최저·일최고는 시간값의 최저·최고.
  - 칸 = ERA5-Land 0.1° 격자(칸 중심 좌표). SAGE의 "환경 자료 해상도로 묶기"와 같은 단위.

연도는 달력 연도(1–12월). ERA5-Land는 육지 칸만 값이 있어 해안 칸은 비어 나온다(채우기는 다음 단계).

입력: data/gbif/occurrences_clean.parquet (모든 버전에 쓰이는 칸의 합집합)
출력: data/weather/cells_v01.csv (cell_id, lat, lon), data/weather/annual_extremes_<시작>_<끝>.csv
      data/weather/gee_tasks.jsonl (작업별 EECU 사용량 기록)
실행: python -m src.gee_annual_extremes --start 1991 --end 2025 [--test 1000]
"""

import argparse
import io
import json
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

ROOT = Path(__file__).resolve().parents[1]
OCC = ROOT / "data" / "gbif" / "occurrences_clean.parquet"
OUT = ROOT / "data" / "weather"
PROJECT = "ai-plant-planner"
DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
GRID = 0.1
SCALE_M = 11132
DRIVE_FOLDER = "aiplant_gee"
CHUNK = 15000  # 칸 1.5만 개 ≈ 요청 2~3 MB
VERSIONS = ["v_main", "v_urban", "v_intro", "v_all", "v_coast1", "v_coast20"]


def cell_table() -> pd.DataFrame:
    o = pd.read_parquet(OCC, columns=["decimalLatitude", "decimalLongitude"] + VERSIONS)
    o = o[o[VERSIONS].any(axis=1)]
    lat = np.round(o["decimalLatitude"] / GRID).astype(int)
    lon = np.round(o["decimalLongitude"] / GRID).astype(int)
    lon = ((lon + 1800) % 3600) - 1800  # -180 경계 정리
    cells = pd.DataFrame({"ilat": lat, "ilon": lon}).drop_duplicates().sort_values(["ilat", "ilon"])
    cells["cell_id"] = cells["ilat"].astype(str) + "_" + cells["ilon"].astype(str)
    cells["lat"] = (cells["ilat"] * GRID).round(1)
    cells["lon"] = (cells["ilon"] * GRID).round(1)
    return cells[["cell_id", "lat", "lon"]].reset_index(drop=True)


def annual_image(year: int) -> ee.Image:
    ic = ee.ImageCollection(DATASET).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    tmin = ic.select("temperature_2m_min").min().rename(f"tmin_{year}")
    tmax = ic.select("temperature_2m_max").max().rename(f"tmax_{year}")
    return tmin.addBands(tmax)


def run_export(cells: pd.DataFrame, years: list, name: str) -> dict:
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"cell_id": r.cell_id})
                               for r in cells.itertuples()])
    img = ee.Image.cat([annual_image(y) for y in years])
    samp = img.sampleRegions(collection=fc, properties=["cell_id"], scale=SCALE_M, geometries=False)
    selectors = ["cell_id"] + [f"{v}_{y}" for y in years for v in ("tmin", "tmax")]
    task = ee.batch.Export.table.toDrive(collection=samp, description=name, folder=DRIVE_FOLDER,
                                         fileNamePrefix=name, fileFormat="CSV", selectors=selectors)
    task.start()
    t0 = time.time()
    while task.active():
        time.sleep(15)
    st = task.status()
    rec = {"name": name, "state": st["state"], "wall_s": round(time.time() - t0),
           "eecu_s": st.get("batch_eecu_usage_seconds"), "n_cells": len(cells), "years": [years[0], years[-1]],
           "error": st.get("error_message"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(OUT / "gee_tasks.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def download(name: str) -> pd.DataFrame:
    drv = build("drive", "v3", credentials=ee.data.get_persistent_credentials())
    files = drv.files().list(q=f"name = '{name}.csv' and trashed = false",
                             fields="files(id,name,createdTime)", orderBy="createdTime desc").execute()["files"]
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, drv.files().get_media(fileId=files[0]["id"]))
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return pd.read_csv(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1991)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--test", type=int, default=0, help="무작위 칸 수로 사용량만 시험")
    a = ap.parse_args()

    ee.Initialize(project=PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    cells = cell_table()
    cells.to_csv(OUT / "cells_v01.csv", index=False)
    years = list(range(a.start, a.end + 1))
    if a.test:
        cells = cells.sample(a.test, random_state=0)
        name = f"annual_extremes_test{a.test}_{a.start}_{a.end}"
    else:
        name = f"annual_extremes_{a.start}_{a.end}"
    print(f"칸 {len(cells)}개, {years[0]}–{years[-1]} ({len(years)}년)")
    # GEE 요청 크기 한도(10 MB) 때문에 칸을 나눠 보낸다. 조각 결과는 각각 저장해 두고 마지막에 합친다.
    parts = []
    for i, start in enumerate(range(0, len(cells), CHUNK)):
        part_name = f"{name}_p{i:02d}"
        part_file = OUT / "parts" / f"{part_name}.csv"
        if not part_file.exists():
            rec = run_export(cells.iloc[start:start + CHUNK], years, part_name)
            print(rec)
            if rec["state"] != "COMPLETED":
                raise SystemExit(f"{part_name} 실패: {rec['error']}")
            part_file.parent.mkdir(exist_ok=True)
            download(part_name).to_csv(part_file, index=False)
        parts.append(pd.read_csv(part_file))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    print("받은 칸", len(df), "/", len(cells), " (빠진 칸 = 해안 등 값 없음)")


if __name__ == "__main__":
    main()
