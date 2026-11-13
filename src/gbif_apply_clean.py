"""좌표 정제 표시(cc_flags)와 원산 표시(occurrences_flagged)를 합쳐 실제로 기록을 거른다. (결정 F, A)

근거 (모두 data/external/SOURCES.md·메모 참조):
  - SAGE (Arens et al., arXiv 2609.31082 v1, 부록 A.2): CoordinateCleaner로 국가·주 중심점, 수도, GBIF 본부,
    생물 기관, 바다, 도시, 위경도 동일, (0,0), 종 안 중복 좌표 기록을 제거. 이상치 검사는 쓰지 않음.
    coordinateUncertaintyInMeters ≤ 1000 m 또는 누락만 유지. 정제 후 원산 다각형(WCVP L3, 버퍼 없음) 안만 유지,
    원산 = introduced·extinct·location_doubtful 표시가 없는 지역. 원산 범위가 없는 종은 제외.
  - 검사 기준값은 CoordinateCleaner 기본값(Zizka et al., 2019) 그대로 (수도 반경 10 km 등).
  - 원산/도입 비교, 목록 밖(unlisted) 포함 비교: Zhou et al. 2023 (결정 A).
  - 해안 감도 1 km(rWCVP, Brown et al. 2023)·20 km(RuHere) (결정 A).

순서: 종 선택 -> 좌표 불확실도 -> 좌표 정제(도시 제외) -> 원산 판정 -> 종 안 중복 좌표 제거.
중복 제거는 CoordinateCleaner의 .dpl 표시를 쓰지 않고 다른 필터를 다 거친 뒤 다시 한다.
.dpl은 전체 기록에서 첫 기록만 남기므로, 남은 첫 기록이 불확실도 등 다른 필터로 빠지면 같은 좌표가 통째로 사라진다.

버전(기록별 True/False 열):
  v_main        원산 + 도시 제외                     (본 분석)
  v_urban       원산 + 도시 포함                     (도시 검사 감도)
  v_intro       원산·도입 + 도시 제외                (결정 A 비교)
  v_all         원산·도입·목록 밖 + 도시 제외        (Zhou 방식 감도)
  v_coast1      원산 + 원산 다각형 1 km 안 해안 기록  (해안 감도)
  v_coast20     원산 + 원산 다각형 20 km 안 해안 기록 (해안 감도)
해안 감도 버전은 바다 검사를 그대로 적용한다(다른 정제는 모든 버전에서 같게 유지).

입력: data/gbif/occurrences_flagged.parquet, data/gbif/cc_flags.parquet, data/gbif/taxon_match.csv,
      data/gbif/wcvp_species_map.csv, data/external/wcvp/wcvp_distribution.csv, data/external/wgsrpd/level3.geojson
출력: data/gbif/occurrences_clean.parquet  (어느 버전에든 들어가는 기록 + 버전 열)
      data/gbif/clean_steps.csv            (단계별 남은 기록 수 — 논문 흐름표용)
      data/gbif/species_counts_clean.csv   (종 × 버전 기록 수)
실행: python -m src.gbif_apply_clean
"""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

from src.gbif_taxon_match import OUT_DIR

EXT = OUT_DIR.parent / "external"
UNCERTAINTY_MAX_M = 1000
CC_ALWAYS = [".zer", ".equ", ".cen", ".cap", ".gbf", ".inst", ".sea"]
COAST_KM = (1, 20)
VERSIONS = ["v_main", "v_urban", "v_intro", "v_all", "v_coast1", "v_coast20"]


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def native_polygons(wcvp_ids) -> dict:
    dist = pd.read_csv(EXT / "wcvp" / "wcvp_distribution.csv", sep="|", dtype=str)
    d = dist[dist["plant_name_id"].isin(set(wcvp_ids))
             & (dist["introduced"] == "0") & (dist["extinct"] == "0")
             & (dist["location_doubtful"] == "0")]
    regions = gpd.read_file(EXT / "wgsrpd" / "level3.geojson")[["LEVEL3_COD", "geometry"]]
    geom = regions.set_index("LEVEL3_COD")["geometry"]
    return {pid: geom.loc[geom.index.isin(g["area_code_l3"])].unary_union
            for pid, g in d.groupby("plant_name_id")}


def coast_distance_km(rows: pd.DataFrame, polys: dict) -> pd.Series:
    """원산 다각형까지 거리(km). 경위도 평면에서 가장 가까운 점을 찾고 그 두 점의 대원 거리를 잰다."""
    out = pd.Series(np.nan, index=rows.index)
    for pid, g in rows.groupby("wcvp_plant_name_id"):
        poly = polys.get(pid)
        if poly is None:
            continue
        for idx, lon, lat in zip(g.index, g["decimalLongitude"], g["decimalLatitude"]):
            pt = gpd.points_from_xy([lon], [lat])[0]
            near = nearest_points(pt, poly)[1]
            out[idx] = _haversine_km(lon, lat, near.x, near.y)
    return out


def main() -> None:
    occ = pd.read_parquet(OUT_DIR / "occurrences_flagged.parquet")
    cc = pd.read_parquet(OUT_DIR / "cc_flags.parquet")
    occ = occ.merge(cc, on="gbifID", how="left", validate="one_to_one")
    steps = [("GBIF 다운로드(좌표 있음, 표본·iNat 연구등급, 지리 이슈 없음)", len(occ))]

    match = pd.read_csv(OUT_DIR / "taxon_match.csv")
    species_keys = set(match.loc[match["rank"] == "SPECIES", "count_key"].astype(int))
    occ = occ[occ["count_key"].isin(species_keys)]
    steps.append(("종 단위로 대조된 종만(속 단위 7종 제외)", len(occ)))

    unc = occ["coordinateUncertaintyInMeters"]
    occ = occ[unc.isna() | (unc <= UNCERTAINTY_MAX_M)]
    steps.append((f"좌표 불확실도 ≤ {UNCERTAINTY_MAX_M} m 또는 누락", len(occ)))

    for col in CC_ALWAYS:
        occ = occ[occ[col].fillna(True)]
    steps.append(("CoordinateCleaner: 0좌표·위경도 동일·중심점·수도·GBIF본부·기관·바다", len(occ)))
    occ = occ.copy()
    occ["urban"] = ~occ[".urb"].fillna(True)

    clean_native = ((occ["native_status"] == "native")
                    & (occ["extinct"] == "0") & (occ["location_doubtful"] == "0"))
    polys = native_polygons(occ.loc[occ["native_status"] == "no_region", "wcvp_plant_name_id"].dropna())
    coast = occ["native_status"] == "no_region"
    occ["coast_km"] = np.nan
    occ.loc[coast, "coast_km"] = coast_distance_km(occ[coast], polys)

    not_urban = ~occ["urban"]
    occ["v_main"] = clean_native & not_urban
    occ["v_urban"] = clean_native
    occ["v_intro"] = (clean_native | (occ["native_status"] == "introduced")) & not_urban
    occ["v_all"] = (clean_native | occ["native_status"].isin(["introduced", "unlisted"])) & not_urban
    for km in COAST_KM:
        occ[f"v_coast{km}"] = (clean_native | (occ["coast_km"] <= km)) & not_urban

    occ = occ[occ[VERSIONS].any(axis=1)]
    # 버전마다 종 안 같은 좌표는 하나만 남긴다(gbifID가 가장 작은 기록). 버전마다 남는 기록 집합이 달라서 따로 한다.
    occ = occ.sort_values("gbifID")
    for v in VERSIONS:
        sub = occ[occ[v]]
        dup = sub.duplicated(["count_key", "decimalLongitude", "decimalLatitude"])
        occ.loc[sub.index[dup], v] = False
    occ = occ[occ[VERSIONS].any(axis=1)]

    for v, label in [("v_urban", "원산만(도시 포함), 종 안 중복 좌표 제거"),
                     ("v_main", "본 분석: 원산만 + 도시 제외, 중복 제거")]:
        steps.append((label, int(occ[v].sum())))

    keep = ["gbifID", "count_key", "wcvp_plant_name_id", "decimalLatitude", "decimalLongitude",
            "coordinateUncertaintyInMeters", "year", "month", "day", "basisOfRecord", "countryCode",
            "datasetKey", "native_status", "area_code_l3", "urban", "coast_km", ".otl"] + VERSIONS
    occ[keep].to_parquet(OUT_DIR / "occurrences_clean.parquet", index=False)

    pd.DataFrame(steps, columns=["단계", "남은 기록"]).to_csv(
        OUT_DIR / "clean_steps.csv", index=False, encoding="utf-8-sig")

    names = (match[match["rank"] == "SPECIES"].groupby("count_key")["식물명"]
             .apply(lambda s: " / ".join(s)))
    counts = occ.groupby("count_key")[VERSIONS].sum().reindex(sorted(species_keys), fill_value=0)
    counts.insert(0, "식물명", counts.index.map(names))
    counts.to_csv(OUT_DIR / "species_counts_clean.csv", encoding="utf-8-sig")

    print(pd.DataFrame(steps, columns=["단계", "남은 기록"]).to_string(index=False))
    print(occ[VERSIONS].sum().to_string())


if __name__ == "__main__":
    main()
