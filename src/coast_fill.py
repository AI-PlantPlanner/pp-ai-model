"""ERA5-Land 값이 없는 해안 칸을 50 km 안의 가장 가까운 육지 칸으로 대체한다. (결정 E, 11-06 사용자 승인)

근거: SAGE(Arens et al., arXiv 2609.31082 부록 "Missing-value imputation") — 해안·작은 섬처럼 자료가 없는 칸은
가장 가까운 유효 칸 값으로 채우고 반경(1 km 격자에서 50칸 = 50 km) 안에 없으면 제외. 우리 격자는 0.1°(≈11 km)라
칸 수가 아닌 거리(50 km)로 맞췄다.

방법: GEE에서 ERA5-Land 육지 마스크(0.1° 격자 전체)를 한 번 내려받아, 값 없는 칸마다 대원 거리로
가장 가까운 육지 칸 중심을 찾는다. 50 km 안에 없으면 제외(fill_cell 없음).

입력: data/weather/cells_v01.csv
출력: data/weather/era5land_landmask.npy, data/weather/cells_v01_fill.csv
      (cell_id, lat, lon, src_lat, src_lon, fill_km, status = land / filled / dropped)
실행: python -m src.coast_fill
"""

import io
import urllib.request

import ee
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.gee_annual_extremes import DATASET, OUT, PROJECT

MAX_KM = 50.0
EARTH_KM = 6371.0088
TRANSFORM = [0.1, 0, -180.05, 0, -0.1, 90.05]  # ERA5-Land 원 격자(칸 중심이 0.1° 배수)
NX, NY = 3600, 1801


def land_mask() -> np.ndarray:
    path = OUT / "era5land_landmask.npy"
    if path.exists():
        return np.load(path)
    img = ee.ImageCollection(DATASET).first().select("temperature_2m").mask().gt(0).unmask(0).toByte()
    url = img.getDownloadURL({"crs": "EPSG:4326", "crs_transform": TRANSFORM,
                              "dimensions": f"{NX}x{NY}", "format": "NPY"})
    with urllib.request.urlopen(url) as r:
        arr = np.load(io.BytesIO(r.read()))
    mask = arr["temperature_2m"].astype(bool) if arr.dtype.names else arr.astype(bool)
    np.save(path, mask)
    return mask


def main() -> None:
    ee.Initialize(project=PROJECT)
    mask = land_mask()
    iy, ix = np.nonzero(mask)
    land_lat = 90.0 - 0.1 * iy
    land_lon = -180.0 + 0.1 * ix
    print("육지 칸", mask.sum())

    cells = pd.read_csv(OUT / "cells_v01.csv")
    ry = np.round((90.0 - cells["lat"]) / 0.1).astype(int)
    rx = np.round((cells["lon"] + 180.0) / 0.1).astype(int) % NX
    on_land = mask[ry, rx]
    cells["status"] = np.where(on_land, "land", "dropped")
    cells["src_lat"], cells["src_lon"], cells["fill_km"] = cells["lat"], cells["lon"], 0.0

    tree = BallTree(np.radians(np.c_[land_lat, land_lon]), metric="haversine")
    off = ~on_land
    dist, idx = tree.query(np.radians(cells.loc[off, ["lat", "lon"]].to_numpy()), k=1)
    km = dist[:, 0] * EARTH_KM
    near = km <= MAX_KM
    rows = cells.index[off]
    cells.loc[rows, "fill_km"] = km
    cells.loc[rows, "src_lat"] = np.round(land_lat[idx[:, 0]], 1)
    cells.loc[rows, "src_lon"] = np.round(land_lon[idx[:, 0]], 1)
    cells.loc[rows[near], "status"] = "filled"
    cells.loc[rows[~near], ["src_lat", "src_lon"]] = np.nan
    cells.to_csv(OUT / "cells_v01_fill.csv", index=False)
    print(cells["status"].value_counts().to_string())
    print("채운 칸 거리(km):", cells.loc[cells.status == "filled", "fill_km"].describe().round(1).to_dict())


if __name__ == "__main__":
    main()
