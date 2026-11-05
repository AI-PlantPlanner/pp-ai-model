"""rawdata 100종 학명을 GBIF Backbone Taxonomy에 대조하고, 종별 GBIF 기록 수를 센다.

이 스크립트는 판단을 내리지 않는다. 사람이 결정해야 할 것(품종→종 대응, 속 단위 종 처리,
자생지 한정 여부 등)을 정하기 위한 사실 자료만 만든다.

학명 전처리는 기계적 정리만 한다(원본 파일은 고치지 않음):
  - '?' -> 공백: 3차 원본 엑셀에 공백 자리로 '?'가 들어가 있다(인코딩 손상으로 보임).
  - 굽은 따옴표(‘ ’) -> 곧은 따옴표('): 품종명 표기 통일.
  - 연속 공백 1개로.
오탈자 교정·품종명 제거·이명 치환은 하지 않는다. GBIF 응답(matchType, rank, status)을 그대로 기록한다.

기록 수는 원예팀에 보낸 GBIF 사용 요청 메일(2026-10)에 적은 기준대로 센다:
  - 표본(basisOfRecord=PRESERVED_SPECIMEN)
  - iNaturalist 연구 등급 관측(GBIF 데이터셋 50c9509d-..., DOI 10.15468/ab3s5x)
    iNaturalist는 재배·사육 개체를 Casual로 분류해 연구 등급에서 제외한다(iNaturalist 도움말).
  - 좌표 있음, GBIF 지리 이슈 없음
비교용으로 필터 없는 좌표 보유 기록 수도 함께 센다.

출력: data/gbif/taxon_match.csv, data/gbif/taxon_match_raw.jsonl (API 원응답, 재현용)
실행: python -m src.gbif_taxon_match
"""

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from src import config

GBIF_API = "https://api.gbif.org/v1"
INAT_RESEARCH_GRADE_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
OUT_DIR = config.PROJECT_ROOT / "data" / "gbif"


def clean_name(raw: str) -> str:
    name = raw.replace("?", " ").replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", name).strip()


def _get(path: str, params: dict) -> dict:
    url = f"{GBIF_API}/{path}?{urllib.parse.urlencode(params, doseq=True)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def _count(taxon_key: int, **filters) -> int:
    params = {"taxonKey": taxon_key, "hasCoordinate": "true", "limit": 0, **filters}
    return _get("occurrence/search", params)["count"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plants = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    rows, raw_lines = [], []

    for _, p in plants.iterrows():
        raw_name = p[config.COL_SCIENTIFIC_NAME]
        query = clean_name(raw_name)
        m = _get("species/match", {"name": query, "kingdom": "Plantae", "verbose": "true"})
        raw_lines.append(json.dumps({"query": query, "response": m}, ensure_ascii=False))

        # 이명(SYNONYM)이면 GBIF가 지정한 정명(accepted) 키로 센다. 기록은 정명 키 아래에 모이기 때문.
        key = m.get("acceptedUsageKey") or m.get("usageKey")
        row = {
            "식물명": p[config.COL_NAME],
            "rawdata_학명": raw_name,
            "조회_학명": query,
            "matchType": m.get("matchType"),
            "confidence": m.get("confidence"),
            "rank": m.get("rank"),
            "status": m.get("status"),
            "gbif_scientificName": m.get("scientificName"),
            "gbif_accepted": m.get("species") if m.get("status") == "SYNONYM" else None,
            "count_key": key,
            "family": m.get("family"),
        }
        if key:
            row["n_coord_all"] = _count(key)
            row["n_specimen"] = _count(
                key, basisOfRecord="PRESERVED_SPECIMEN", hasGeospatialIssue="false"
            )
            row["n_inat_rg"] = _count(
                key, datasetKey=INAT_RESEARCH_GRADE_DATASET, hasGeospatialIssue="false"
            )
            row["n_wild"] = row["n_specimen"] + row["n_inat_rg"]
        rows.append(row)
        print(f"{row['식물명']:<16} {query:<40} {row['matchType']:<10} {row['rank']} "
              f"{row['status']} wild={row.get('n_wild')}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "taxon_match.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "taxon_match_raw.jsonl").write_text("\n".join(raw_lines), encoding="utf-8")
    meta = {"run_at": datetime.now(timezone.utc).isoformat(), "gbif_api": GBIF_API,
            "inat_dataset": INAT_RESEARCH_GRADE_DATASET, "n_plants": len(df)}
    (OUT_DIR / "taxon_match_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
