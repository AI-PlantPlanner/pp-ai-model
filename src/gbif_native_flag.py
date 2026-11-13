"""GBIF 기록마다 해당 종의 WCVP 원산 지역 안인지(원산) 밖인지(도입) 표시한다. (결정 A: 둘 다 만들어 비교)

판단 규칙 (전 종에 똑같이 적용, 종별 예외 없음):
  1. WCVP 조회 이름 = GBIF 대조에 실제 쓰인 이름(canonicalName). GBIF가 품종명을 떼고 대조했으므로
     기록과 같은 이름 기준이 된다. 잡종 표기 '×'가 빠진 경우 '×'를 넣어 한 번 더 조회한다.
  2. WCVP에서 그 이름이 Accepted면 그 id, 아니면(이명·부적법명) WCVP가 지정한 accepted_plant_name_id.
     후보가 둘 이상 갈리면 판정하지 않는다(wcvp_status 기록).
  3. 기록 좌표 -> TDWG WGSRPD 3단계 지역(점-다각형 포함 판정, 해안 보정 없음).
  4. 지역이 WCVP 분포에 introduced=0 으로 있으면 native, introduced=1 이면 introduced,
     WCVP 분포 목록에 없으면 unlisted, 좌표가 어느 지역 다각형에도 안 들면 no_region,
     종의 WCVP 분포 자체가 없으면 no_wcvp.
  extinct·location_doubtful 표시는 판정에 쓰지 않고 그대로 따로 남긴다.

입력: data/gbif/occurrences_<key>.zip, data/gbif/taxon_match.csv, taxon_match_raw.jsonl,
      data/external/wcvp/*.csv, data/external/wgsrpd/level3.geojson
출력: data/gbif/occurrences_flagged.parquet, data/gbif/wcvp_species_map.csv
실행: python -m src.gbif_native_flag
"""

import json
import zipfile

import geopandas as gpd
import pandas as pd

from src.gbif_taxon_match import OUT_DIR, _get

EXT = OUT_DIR.parent / "external"
OCC_COLS = [
    "gbifID", "speciesKey", "taxonKey", "basisOfRecord", "decimalLatitude", "decimalLongitude",
    "coordinateUncertaintyInMeters", "year", "month", "day", "institutionCode", "countryCode",
    "datasetKey",
]


def load_occurrences() -> pd.DataFrame:
    req = json.loads((OUT_DIR / "download_request.json").read_text())
    with zipfile.ZipFile(OUT_DIR / req["file"]) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, sep="\t", usecols=OCC_COLS, quoting=3, low_memory=False)


def wcvp_species_map() -> pd.DataFrame:
    match = pd.read_csv(OUT_DIR / "taxon_match.csv")
    raw = {}
    for line in (OUT_DIR / "taxon_match_raw.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        raw[rec["query"]] = rec["response"]
    names = pd.read_csv(
        EXT / "wcvp" / "wcvp_names.csv", sep="|", dtype=str,
        usecols=["plant_name_id", "taxon_name", "taxon_status", "accepted_plant_name_id"],
    )
    rows = []
    for _, m in match[match["rank"] == "SPECIES"].iterrows():
        canonical = raw[m["조회_학명"]]["canonicalName"]
        tried = [canonical]
        parts = canonical.split()
        if len(parts) == 2 and "×" in m["조회_학명"]:
            tried.append(f"{parts[0]} × {parts[1]}")
        hit = pd.DataFrame()
        for name in tried:
            hit = names[names["taxon_name"] == name]
            if len(hit):
                break
        accepted = hit[hit["taxon_status"] == "Accepted"]["plant_name_id"].unique()
        if len(accepted) == 1:
            wid, status = accepted[0], "accepted"
        else:
            targets = hit["accepted_plant_name_id"].dropna().unique()
            if len(targets) == 1:
                wid, status = targets[0], "via_" + "/".join(sorted(hit["taxon_status"].unique()))
            else:
                wid, status = None, ("not_found" if hit.empty else
                                     "ambiguous:" + "/".join(sorted(hit["taxon_status"].unique())))
        rows.append({"식물명": m["식물명"], "count_key": int(m["count_key"]),
                     "wcvp_query": name if len(hit) else canonical,
                     "wcvp_plant_name_id": wid, "wcvp_status": status})
    out = pd.DataFrame(rows)
    acc = names.set_index("plant_name_id")["taxon_name"]
    out["wcvp_accepted_name"] = out["wcvp_plant_name_id"].map(acc)
    return out


def main() -> None:
    occ = load_occurrences()
    species = wcvp_species_map()
    species.to_csv(OUT_DIR / "wcvp_species_map.csv", index=False, encoding="utf-8-sig")
    occ = flag_occurrences(occ, species)
    occ.to_parquet(OUT_DIR / "occurrences_flagged.parquet", index=False)
    print(occ["native_status"].value_counts().to_string())


def flag_occurrences(occ: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    """기록에 count_key·WGSRPD 지역·WCVP 원산 표시를 붙인다(11-12: 추가 다운로드에도 같은 규칙을 쓰려고 main에서 분리)."""
    # 요청 키가 종보다 낮은 계급일 수 있다(예: 섬초롱꽃 -> GBIF 정명이 C. punctata var. punctata).
    # 기록에는 speciesKey가 있으므로, 요청 키마다 GBIF가 매긴 speciesKey를 조회해 그 값으로 붙인다.
    # 서로 다른 요청 키가 같은 speciesKey를 가지면 구분할 수 없으므로 멈춘다.
    keys = sorted(set(species["count_key"]))
    sp_key = {k: _get(f"species/{k}", {})["speciesKey"] for k in keys}
    if len(set(sp_key.values())) != len(sp_key):
        raise ValueError("서로 다른 요청 키가 같은 speciesKey를 가짐 — 기록을 종에 붙일 수 없음")
    by_species = {v: k for k, v in sp_key.items()}
    occ["count_key"] = occ["speciesKey"].map(by_species)
    unmatched = occ["count_key"].isna()
    print(f"기록 {len(occ)}건, 우리 종 키에 안 붙는 기록 {unmatched.sum()}건")
    occ["count_key"] = occ["count_key"].astype("Int64")

    regions = gpd.read_file(EXT / "wgsrpd" / "level3.geojson")[["LEVEL3_COD", "geometry"]]
    pts = gpd.GeoDataFrame(
        occ[["gbifID"]],
        geometry=gpd.points_from_xy(occ["decimalLongitude"], occ["decimalLatitude"]),
        crs=regions.crs,
    )
    joined = gpd.sjoin(pts, regions, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # 경계선 위 점은 첫 지역
    occ["area_code_l3"] = joined["LEVEL3_COD"].values

    dist = pd.read_csv(EXT / "wcvp" / "wcvp_distribution.csv", sep="|", dtype=str)
    id_by_key = species.drop_duplicates("count_key").set_index("count_key")["wcvp_plant_name_id"]
    occ["wcvp_plant_name_id"] = occ["count_key"].map(id_by_key)
    d = dist[dist["plant_name_id"].isin(set(species["wcvp_plant_name_id"].dropna()))]
    d = d.drop_duplicates(["plant_name_id", "area_code_l3"])
    occ = occ.merge(
        d[["plant_name_id", "area_code_l3", "introduced", "extinct", "location_doubtful"]],
        left_on=["wcvp_plant_name_id", "area_code_l3"],
        right_on=["plant_name_id", "area_code_l3"], how="left",
    ).drop(columns="plant_name_id")

    has_dist = occ["wcvp_plant_name_id"].isin(set(dist["plant_name_id"]))
    occ["native_status"] = "unlisted"
    occ.loc[occ["introduced"] == "0", "native_status"] = "native"
    occ.loc[occ["introduced"] == "1", "native_status"] = "introduced"
    occ.loc[occ["area_code_l3"].isna(), "native_status"] = "no_region"
    occ.loc[~has_dist, "native_status"] = "no_wcvp"
    return occ


if __name__ == "__main__":
    main()
