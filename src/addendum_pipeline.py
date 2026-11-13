"""학명 재확인 식물 추가 반영 (11-12). 본 구축과 같은 함수·같은 기준을 추가 기록에만 적용하고 기존 결과 뒤에 붙인다.

왜 전체를 다시 돌리지 않나: 좌표 정제(CoordinateCleaner)의 검사는 기록 단위(수도·바다 등) 또는 종 단위(이상치·중복)라
다른 종 기록이 결과에 영향을 주지 않는다. 기상 칸 번호(src_id)를 다시 매기면 기존 기상 파일 전부를 바꿔야 하므로,
새 칸은 기존 번호 뒤에 이어 붙인다(새 칸 파일 이름 p16b — 기존 조각과 겹치지 않음).
바꾸기 전 파일은 *_pre_addendum.* 으로 한 번만 보관한다.

단계(순서대로):
  flag      추가 zip → WCVP 원산 표시(gbif_native_flag.flag_occurrences) → occurrences_flagged에 붙임
            + 추가분만 data/gbif/addendum/occurrences_flagged_add.parquet (R 입력)
  (Rscript src/gbif_clean.R data/gbif/addendum/occurrences_flagged_add.parquet data/gbif/addendum/cc_flags_add.parquet)
  merge_cc  추가 정제 표시를 cc_flags에 붙임
  (python -m src.gbif_apply_clean)
  cells     정제 기록의 새 칸을 cells_v01에 붙이고 coast_fill 다시 실행(같은 규칙), 새 원천 칸에 src_id 이어 붙임
  weather   새 원천 칸만 GEE 일별 기상(2016–2025) → data/weather/daily/<연도>/p16b.parquet
  (python -m src.daily_features, python -m src.cell_climate, python -m src.build_dataset)
실행: python -m src.addendum_pipeline <단계>
"""

import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

from src.gbif_addendum import ADD_DIR, REQ
from src.gbif_native_flag import OCC_COLS, flag_occurrences, wcvp_species_map
from src.gbif_taxon_match import OUT_DIR
from src.gee_annual_extremes import OUT as WOUT

ADD_FILE = "p16b.parquet"


def _backup(path):
    b = path.with_name(path.stem + "_pre_addendum" + path.suffix)
    if not b.exists():
        shutil.copy(path, b)
    return b


def flag() -> None:
    import json
    rec = json.loads(REQ.read_text())
    with zipfile.ZipFile(ADD_DIR / rec["file"]) as z, z.open(z.namelist()[0]) as f:
        occ = pd.read_csv(f, sep="\t", usecols=OCC_COLS, quoting=3, low_memory=False)
    species = wcvp_species_map()
    _backup(OUT_DIR / "wcvp_species_map.csv")
    species.to_csv(OUT_DIR / "wcvp_species_map.csv", index=False, encoding="utf-8-sig")
    print(species[species["식물명"].isin(["구즈베리", "마지나타", "드라세나'송오브인디아'"])].to_string())
    add = flag_occurrences(occ, species)
    print("추가 기록", len(add), add["native_status"].value_counts().to_dict())
    add.to_parquet(ADD_DIR / "occurrences_flagged_add.parquet", index=False)
    base = pd.read_parquet(_backup(OUT_DIR / "occurrences_flagged.parquet"))
    if base["gbifID"].isin(add["gbifID"]).any():
        raise SystemExit("기존 기록과 gbifID 겹침")
    pd.concat([base, add[base.columns]], ignore_index=True).to_parquet(OUT_DIR / "occurrences_flagged.parquet", index=False)


def merge_cc() -> None:
    base = pd.read_parquet(_backup(OUT_DIR / "cc_flags.parquet"))
    add = pd.read_parquet(ADD_DIR / "cc_flags_add.parquet")
    pd.concat([base, add[base.columns]], ignore_index=True).to_parquet(OUT_DIR / "cc_flags.parquet", index=False)
    print("정제 표시", len(base), "+", len(add))


def cells() -> None:
    from src import coast_fill
    from src.gee_annual_extremes import cell_table
    old = pd.read_csv(_backup(WOUT / "cells_v01.csv"))
    new_all = cell_table()
    add = new_all[~new_all["cell_id"].isin(old["cell_id"])]
    print("새 칸", len(add), "(정제 기록 전체 칸", len(new_all), ")")
    pd.concat([old, add], ignore_index=True).to_csv(WOUT / "cells_v01.csv", index=False)
    _backup(WOUT / "cells_v01_fill.csv")
    coast_fill.main()
    fill = pd.read_csv(WOUT / "cells_v01_fill.csv")
    fill = fill[fill["status"].isin(["land", "filled"])]
    src = pd.read_csv(_backup(WOUT / "daily_src_cells.csv"))
    have = set(zip(src["src_lat"].round(1), src["src_lon"].round(1)))
    cand = fill[["src_lat", "src_lon"]].drop_duplicates().sort_values(["src_lat", "src_lon"])
    new = cand[[(round(a, 1), round(b, 1)) not in have for a, b in zip(cand["src_lat"], cand["src_lon"])]].copy()
    new["src_id"] = np.arange(len(src), len(src) + len(new))
    pd.concat([src, new], ignore_index=True).to_csv(WOUT / "daily_src_cells.csv", index=False)
    new.to_csv(ADD_DIR / "new_src_cells.csv", index=False)
    print("새 원천 칸", len(new), "src_id", len(src), "~", len(src) + len(new) - 1)


def weather() -> None:
    import ee
    from src.gee_annual_extremes import PROJECT
    from src.gee_daily import fetch_interactive, to_table
    ee.Initialize(project=PROJECT)
    new = pd.read_csv(ADD_DIR / "new_src_cells.csv")
    for y in range(2016, 2026):
        path = WOUT / "daily" / str(y) / ADD_FILE
        if path.exists() or new.empty:
            continue
        tab = to_table(fetch_interactive(new, y))
        days = 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365
        n = tab.groupby("src_id").size()
        print(y, "칸", tab["src_id"].nunique(), "/", len(new), "날짜 이상", int((n != days).sum()), flush=True)
        tab.to_parquet(path, index=False, compression="zstd")


if __name__ == "__main__":
    {"flag": flag, "merge_cc": merge_cc, "cells": cells, "weather": weather}[sys.argv[1]]()
