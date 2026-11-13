"""데이터 구축 마지막 단계: 식물 표 + 식물·칸 표를 만들고 기상과 연결되는지 검증한다. (백로그 3번 10단계)

결정 사항(근거는 메모·회고록, 각 스크립트 머리말):
  - 칸 = ERA5-Land 0.1° 격자. 한 종이 한 칸에 기록이 여러 개여도 한 번만 센다(SAGE "환경 자료 해상도로 묶기").
  - 버전 = gbif_apply_clean.py의 6개(v_main 본 분석 / v_urban·v_intro·v_all·v_coast1·v_coast20 비교).
  - 기상 없는 해안 칸(50 km 안에 육지 칸 없음, status=dropped)은 뺀다. 그래서 칸 수는 "기상이 있는 칸" 기준.
  - 최소 칸 수 D = 25(본), 14(감도) — min_records_test.py.
  - 섬초롱꽃(7222087) 제외 — WCVP heterotypic 이명, SAGE는 homotypic만 병합.
  - 품종(B, 11-07 결정): 본 분석은 원종 기록으로 근사, 비교 버전은 품종 제외. 품종 = 원예 학명에 따옴표 품종명이 있는 것.
  - 한 GBIF 종을 두 식물이 가리키면(디펜바키아·트로픽스노우) 식물 표에는 둘 다 남기고, 칸 표는 종 단위로 한 번만 둔다.

기상은 복제하지 않는다. 칸 표의 src_id로 data/weather/daily_features/<연도>/pNN.parquet(src_id × 연도 × 1~365일)과 잇는다.
src_id가 어느 조각에 있는지 = src_id // 5000 (gee_daily.CHUNK).

출력(data/dataset/):
  plants.csv          식물 100개: 학명·GBIF 키·품종 여부·제외 사유·버전별 칸 수·학습 세트 포함 여부
  species_cells.parquet  GBIF 종 × 칸: 위경도·src_id·해안 대체 거리·버전별 기록 수(0이면 그 버전에 없음)
  build_report.json   검증 결과와 세트별 식물·종·칸 수
실행: python -m src.build_dataset
"""

import json

import numpy as np
import pandas as pd

from src.gee_annual_extremes import GRID, OCC, OUT, ROOT, VERSIONS
from src.gee_daily import CHUNK

DS = ROOT / "data" / "dataset"
D_MAIN, D_SENS = 25, 14
EXCLUDE_KEYS = {7222087: "heterotypic 이명(섬초롱꽃, SAGE 규칙상 병합 불가)"}
YEARS = range(2016, 2026)
# 11-06 학명 조사에서 사유가 밝혀진 식물(메모 project_aiplant_data_build). 학습 세트 판정에는 쓰지 않고 설명만 단다.
# 11-12: 구즈베리(→ R. uva-crispa)·마지나타(→ D. reflexa 원종 근사)는 src/name_overrides.py로 해소되어 목록에서 뺐다.
KNOWN_REASONS = {
    "콩고": "원예 품종(속명만, 원종 불명)",
    "디펜바키아 ‘마리안느’": "원예 품종(속명만, 원종 불명)",
    "거북알로카시아": "인공 교배종(원산 없음)",
    "아펠란드라": "WCVP Unplaced(원산 판정 불가)",
}


def species_cells() -> pd.DataFrame:
    o = pd.read_parquet(OCC, columns=["count_key", "decimalLatitude", "decimalLongitude"] + VERSIONS)
    o = o[o[VERSIONS].any(axis=1)]
    ilat = np.round(o["decimalLatitude"] / GRID).astype(int)
    ilon = ((np.round(o["decimalLongitude"] / GRID).astype(int) + 1800) % 3600) - 1800
    o = pd.DataFrame({"count_key": o["count_key"].astype(int),
                      "cell_id": ilat.astype(str) + "_" + ilon.astype(str),
                      **{v: o[v].to_numpy() for v in VERSIONS}})
    sc = o.groupby(["count_key", "cell_id"])[VERSIONS].sum().astype("int32").reset_index()
    sc = sc.rename(columns={v: f"n_{v}" for v in VERSIONS})

    cells = pd.read_csv(OUT / "cells_v01_fill.csv")
    src = pd.read_csv(OUT / "daily_src_cells.csv")
    for df in (cells, src):
        df["src_lat"], df["src_lon"] = df["src_lat"].round(1), df["src_lon"].round(1)
    cells = cells.merge(src, on=["src_lat", "src_lon"], how="left")
    sc = sc.merge(cells[["cell_id", "lat", "lon", "status", "src_id", "fill_km"]], on="cell_id", how="left")
    if sc["status"].isna().any():
        raise SystemExit("칸 표에 없는 칸이 있음")
    n_dropped = int((sc["status"] == "dropped").sum())
    sc = sc[sc["status"] != "dropped"].copy()
    sc["src_id"] = sc["src_id"].astype("int32")
    sc = sc.rename(columns={"status": "cell_status"})
    return sc.sort_values(["count_key", "cell_id"]).reset_index(drop=True), n_dropped


def plants_table(sc: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_csv(ROOT / "data" / "gbif" / "taxon_match.csv")
    w = pd.read_csv(ROOT / "data" / "gbif" / "wcvp_species_map.csv")[["식물명", "wcvp_accepted_name", "wcvp_status"]]
    # species_status_preclean.csv의 count_key 열에는 식물명이 들어 있다(정제 전 원산 판정 집계).
    st = pd.read_csv(ROOT / "data" / "gbif" / "species_status_preclean.csv")[["count_key", "native"]]
    st = st.rename(columns={"count_key": "식물명"})
    p = t[["식물명", "rawdata_학명", "count_key", "rank", "gbif_scientificName"]].merge(w, on="식물명", how="left")
    p = p.merge(st, on="식물명", how="left")
    p["is_cultivar"] = p["rawdata_학명"].str.contains("['‘’]", regex=True)

    n = sc.groupby("count_key")[[f"n_{v}" for v in VERSIONS]].agg(lambda s: int((s > 0).sum()))
    n.columns = [f"cells_{v}" for v in VERSIONS]
    p = p.merge(n, left_on="count_key", right_index=True, how="left")
    p[n.columns] = p[n.columns].fillna(0).astype(int)

    def reason(r):
        if r["식물명"] in KNOWN_REASONS:
            return KNOWN_REASONS[r["식물명"]]
        if r["rank"] != "SPECIES":
            return "속 단위 학명(종 미상, 원예팀에 대표 종 확인)"
        if r["count_key"] in EXCLUDE_KEYS:
            return EXCLUDE_KEYS[r["count_key"]]
        if r["cells_v_main"] >= D_MAIN:
            return ""
        if r["cells_v_main"] >= D_SENS:
            return f"칸 {D_SENS}~{D_MAIN - 1}(감도 세트만)"
        if not r["native"] > 0:
            return "원산 기록 없음"
        return f"원산 칸 부족(<{D_SENS})"

    p["note"] = p.apply(reason, axis=1)
    base = p["rank"].eq("SPECIES") & ~p["count_key"].isin(EXCLUDE_KEYS)
    p["set_main"] = base & (p["cells_v_main"] >= D_MAIN)
    p["set_main_d14"] = base & (p["cells_v_main"] >= D_SENS)
    p["set_no_cultivar"] = p["set_main"] & ~p["is_cultivar"]
    return p.drop(columns=["native"])


def eligible(plants: pd.DataFrame, version: str = "v_main", d: int = D_MAIN, cultivars: bool = True) -> pd.DataFrame:
    """모델 단계에서 버전·D·품종 조합별 학습 식물을 고를 때 쓰는 함수."""
    ok = plants["rank"].eq("SPECIES") & ~plants["count_key"].isin(EXCLUDE_KEYS) & (plants[f"cells_{version}"] >= d)
    if not cultivars:
        ok &= ~plants["is_cultivar"]
    return plants[ok]


def check_weather(sc: pd.DataFrame) -> dict:
    need = set(sc["src_id"].unique())
    miss = {}
    for y in YEARS:
        have = set()
        for f in sorted((OUT / "daily_features" / str(y)).glob("p*.parquet")):
            have |= set(pd.read_parquet(f, columns=["src_id"])["src_id"].unique())
        miss[y] = len(need - have)
    # src_id → 조각 번호 규칙 확인(표본)
    s = sc["src_id"].drop_duplicates().sample(200, random_state=0)
    wrong = 0
    for i, g in s.groupby(s // CHUNK):
        ids = set()  # 11-12: 추가 칸은 같은 조각 번호의 p16b 파일에 있음 → 조각 번호로 시작하는 파일 모두
        for f in (OUT / "daily_features" / "2025").glob(f"p{i:02d}*.parquet"):
            ids |= set(pd.read_parquet(f, columns=["src_id"])["src_id"])
        wrong += len(set(g) - ids)
    return {"src_ids_needed": len(need), "missing_by_year": miss, "chunk_rule_errors_in_200": wrong}


def main() -> None:
    DS.mkdir(parents=True, exist_ok=True)
    sc, n_dropped = species_cells()
    plants = plants_table(sc)
    sc.to_parquet(DS / "species_cells.parquet", index=False, compression="zstd")
    plants.to_csv(DS / "plants.csv", index=False, encoding="utf-8-sig")

    sets = {}
    for name, (v, d, cv) in {"main(v_main,D25,품종 근사)": ("v_main", 25, True),
                             "D14 감도": ("v_main", 14, True),
                             "품종 제외": ("v_main", 25, False),
                             **{f"{v} 감도": (v, 25, True) for v in VERSIONS[1:]}}.items():
        e = eligible(plants, v, d, cv)
        keys = set(e["count_key"])
        sub = sc[sc["count_key"].isin(keys) & (sc[f"n_{v}"] > 0)]
        sets[name] = {"plants": len(e), "species": len(keys), "species_cell_pairs": len(sub),
                      "cells": int(sub["cell_id"].nunique()), "records": int(sub[f"n_{v}"].sum())}
    report = {"species_cell_pairs_dropped_coast": n_dropped, "sets": sets,
              "weather": check_weather(sc),
              "excluded": plants.loc[~plants["set_main_d14"], ["식물명", "note"]].values.tolist()}
    (DS / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ["species_cell_pairs_dropped_coast", "sets", "weather"]},
                     ensure_ascii=False, indent=1))
    print("제외 사유:", plants["note"].replace("", "포함").value_counts().to_dict())


if __name__ == "__main__":
    main()
