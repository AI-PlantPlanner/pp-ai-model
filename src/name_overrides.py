"""원예 원자료 학명을 고치지 않고, 확인된 학명으로 GBIF 대조 결과만 바꾼다. (11-12)

근거:
  - 구즈베리: 원예 학명 Ribes grossularia L. → WCVP에서 R. uva-crispa의 이명. GBIF는 속(Ribes)으로만 대조돼 학습에서 빠져 있었다.
    사용자가 원예팀에 확인 메일 발송 후 R. uva-crispa로 쓰기로 결정(11-12).
  - 마지나타: 원예 학명 Dracaena marginata는 명명자에 따라 갈림(Lam. 1786 = D. reflexa var. angustifolia의 이명,
    Aiton 1789 = 부적법 후명). 원예 자료 설명(실내 관엽 관목)으로 D. reflexa var. angustifolia로 봄.
    변종이라 11-07 규칙("품종·종 아래 단위는 원종 기록으로 근사")에 따라 종 D. reflexa로 대조(사용자 결정 11-12).
    → 드라세나'송오브인디아'와 같은 GBIF 종(디펜바키아·트로픽스노우와 같은 처리).
원자료(data/raw)와 정규화 표는 그대로 두고 data/gbif/taxon_match.csv의 해당 행만 바꾼다. 바꾸기 전 파일은
data/gbif/taxon_match_pre_override.csv로 한 번만 보관한다.
실행: python -m src.name_overrides
"""

import json
import shutil

import pandas as pd

from src.gbif_taxon_match import INAT_RESEARCH_GRADE_DATASET, OUT_DIR, _count, _get

OVERRIDES = {  # 식물명: (조회 학명, 근거 요약)
    "구즈베리": ("Ribes uva-crispa", "WCVP 정명(R. grossularia 이명), 11-12 사용자 결정"),
    "마지나타": ("Dracaena reflexa", "D. reflexa var. angustifolia → 원종 근사(11-07 규칙), 11-12 사용자 결정"),
}

# 게임·응답에 보이는 학명(11-13 사용자 결정 "표기 맞춰"). 원자료·정규화 표는 그대로, 출력할 때만 바꾼다.
# 마지나타는 GBIF 대조는 종(D. reflexa)으로 근사했지만, 표시는 확인된 분류군(변종)으로 쓴다.
DISPLAY_NAMES = {
    "구즈베리": "Ribes uva-crispa",
    "마지나타": "Dracaena reflexa var. angustifolia",
}


def main() -> None:
    path = OUT_DIR / "taxon_match.csv"
    backup = OUT_DIR / "taxon_match_pre_override.csv"
    if not backup.exists():
        shutil.copy(path, backup)
    match = pd.read_csv(backup)
    raw_path = OUT_DIR / "taxon_match_raw.jsonl"
    raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
    have = {json.loads(l)["query"] for l in raw_lines}
    match["override"] = None
    for plant, (query, why) in OVERRIDES.items():
        m = _get("species/match", {"name": query, "kingdom": "Plantae", "verbose": "true"})
        if m.get("rank") != "SPECIES" or m.get("matchType") != "EXACT":
            raise SystemExit(f"{plant}: {query} 대조가 종·정확 일치가 아님 ({m.get('rank')}, {m.get('matchType')})")
        if query not in have:
            raw_lines.append(json.dumps({"query": query, "response": m}, ensure_ascii=False))
        key = m.get("acceptedUsageKey") or m.get("usageKey")
        i = match.index[match["식물명"] == plant][0]
        sp = _count(key, basisOfRecord="PRESERVED_SPECIMEN", hasGeospatialIssue="false")
        inat = _count(key, datasetKey=INAT_RESEARCH_GRADE_DATASET, hasGeospatialIssue="false")
        match.loc[i, ["조회_학명", "matchType", "confidence", "rank", "status", "gbif_scientificName", "gbif_accepted",
                      "count_key", "family", "n_coord_all", "n_specimen", "n_inat_rg", "n_wild", "override"]] = [
            query, m.get("matchType"), m.get("confidence"), m.get("rank"), m.get("status"), m.get("scientificName"),
            m.get("species") if m.get("status") == "SYNONYM" else None, key, m.get("family"), _count(key), sp, inat,
            sp + inat, why]
        print(plant, "→", m.get("scientificName"), "key", key, "wild", sp + inat)
    match.to_csv(path, index=False, encoding="utf-8-sig")
    raw_path.write_text("\n".join(raw_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
