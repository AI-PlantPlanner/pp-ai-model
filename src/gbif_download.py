"""GBIF 다운로드 API로 종 단위 대조된 식물의 야생 기록을 요청·수령한다.

다운로드 API를 쓰는 이유: 요청마다 DOI가 발급되어 논문에 데이터 묶음을 인용할 수 있다
(GBIF 데이터 이용 규약). 검색 API 페이징으로 긁으면 DOI가 없다.

요청 범위 (gbif_taxon_match.py 결과 기준, 판단 없이 기계적으로):
  - rank == SPECIES 로 대조된 종의 count_key (이명은 GBIF 정명 키). 속 단위 대조 7종은 제외
    — 원예팀 종 확인 전이라 어떤 종의 기록인지 정할 수 없다.
  - 품종·이명·기록 부족 종도 일단 모두 받는다. 포함 여부(B~D)는 받은 뒤 필터로 정하므로
    결정이 바뀌어도 다시 받을 필요가 없다.
기록 조건 (원예팀 GBIF 사용 요청 메일에 적은 기준):
  - basisOfRecord = PRESERVED_SPECIMEN  또는  iNaturalist 연구 등급 데이터셋
  - 좌표 있음, GBIF 지리 이슈 없음
원산지/귀화 구분, 좌표 정제, 공간 솎기는 이 단계에서 하지 않는다(받은 뒤 별도 단계).

인증: ~/.config/gbif/credentials (GBIF_USER=..., GBIF_PWD=...) — 저장소에 넣지 않는다.

실행:
  python -m src.gbif_download request   # 요청 제출, 키를 data/gbif/download_request.json에 저장
  python -m src.gbif_download status    # 처리 상태와 DOI 확인
  python -m src.gbif_download fetch     # 완료되면 zip 내려받기
"""

import base64
import json
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.gbif_taxon_match import GBIF_API, INAT_RESEARCH_GRADE_DATASET, OUT_DIR

CRED_PATH = Path.home() / ".config" / "gbif" / "credentials"
REQUEST_PATH = OUT_DIR / "download_request.json"


def _credentials() -> tuple[str, str]:
    values = dict(
        line.split("=", 1) for line in CRED_PATH.read_text().splitlines() if "=" in line
    )
    return values["GBIF_USER"].strip(), values["GBIF_PWD"].strip()


def _taxon_keys() -> list[int]:
    match = pd.read_csv(OUT_DIR / "taxon_match.csv")
    species = match[match["rank"] == "SPECIES"]
    return sorted({int(k) for k in species["count_key"]})


def build_predicate(user: str, keys: list[int]) -> dict:
    return {
        "creator": user,
        "format": "SIMPLE_CSV",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "in", "key": "TAXON_KEY", "values": [str(k) for k in keys]},
                {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
                {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
                {
                    "type": "or",
                    "predicates": [
                        {"type": "equals", "key": "BASIS_OF_RECORD", "value": "PRESERVED_SPECIMEN"},
                        {"type": "equals", "key": "DATASET_KEY", "value": INAT_RESEARCH_GRADE_DATASET},
                    ],
                },
            ],
        },
    }


def request() -> None:
    user, pwd = _credentials()
    keys = _taxon_keys()
    body = build_predicate(user, keys)
    auth = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    req = urllib.request.Request(
        f"{GBIF_API}/occurrence/download/request",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        download_key = resp.read().decode().strip()
    record = {
        "download_key": download_key,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "n_taxon_keys": len(keys),
        "request_body": {**body, "creator": "<redacted>"},
    }
    REQUEST_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"요청 완료: {download_key} (분류군 키 {len(keys)}개)")


def _meta() -> dict:
    key = json.loads(REQUEST_PATH.read_text())["download_key"]
    with urllib.request.urlopen(f"{GBIF_API}/occurrence/download/{key}", timeout=30) as resp:
        return json.load(resp)


def status() -> None:
    m = _meta()
    print({k: m.get(k) for k in ("key", "status", "doi", "totalRecords", "size", "downloadLink")})


def fetch() -> None:
    m = _meta()
    if m.get("status") != "SUCCEEDED":
        print(f"아직 준비 안 됨: {m.get('status')}")
        return
    dest = OUT_DIR / f"occurrences_{m['key']}.zip"
    # 대용량이라 연결이 끊기면 잘린 파일이 남는다 -> curl 이어받기 + GBIF가 알려준 크기와 대조.
    subprocess.run(
        ["curl", "-sSL", "--retry", "5", "-C", "-", "-o", str(dest), m["downloadLink"]],
        check=True,
    )
    if dest.stat().st_size != m["size"]:
        raise RuntimeError(f"크기 불일치: 받은 {dest.stat().st_size} != GBIF {m['size']} — 다시 실행")
    if not zipfile.is_zipfile(dest):
        raise RuntimeError("zip 형식 아님 — 파일 삭제 후 다시 실행")
    record = json.loads(REQUEST_PATH.read_text())
    record.update({"doi": m.get("doi"), "total_records": m.get("totalRecords"),
                   "fetched_at": datetime.now(timezone.utc).isoformat(), "file": dest.name})
    REQUEST_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {dest} / DOI {m.get('doi')} / {m.get('totalRecords')}건")


if __name__ == "__main__":
    {"request": request, "status": status, "fetch": fetch}[sys.argv[1]]()
