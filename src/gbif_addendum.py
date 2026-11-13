"""학명 재확인 식물의 GBIF 기록 추가 다운로드 (11-12, 사용자 결정: 구즈베리 = Ribes uva-crispa).

본 다운로드(gbif_download.py, DOI 10.15468/dl.fdf4xu)와 같은 조건(build_predicate)으로 추가 종만 따로 요청한다.
추가 다운로드도 DOI가 따로 발급되므로 논문에 두 DOI를 함께 인용한다.
근거: 구즈베리 원예 학명 Ribes grossularia L. → WCVP 정명 R. uva-crispa(이명), 사용자가 원예팀에 확인 메일 발송 후 사용 결정(11-12).
실행: python -m src.gbif_addendum request|status|fetch
"""
import json
import sys
import urllib.request
import base64
import subprocess
import zipfile
from datetime import datetime, timezone

from src.gbif_download import GBIF_API, OUT_DIR, _credentials, build_predicate

ADD_DIR = OUT_DIR / "addendum"
REQ = ADD_DIR / "download_request.json"
TAXA = {"구즈베리": ("Ribes uva-crispa", 2986185)}


def request() -> None:
    ADD_DIR.mkdir(parents=True, exist_ok=True)
    user, pwd = _credentials()
    body = build_predicate(user, [k for _, k in TAXA.values()])
    auth = base64.b64encode(f"{user}:{pwd}".encode()).decode()
    req = urllib.request.Request(f"{GBIF_API}/occurrence/download/request", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        key = r.read().decode().strip()
    REQ.write_text(json.dumps({"download_key": key, "requested_at": datetime.now(timezone.utc).isoformat(),
                               "taxa": TAXA, "request_body": {**body, "creator": "<redacted>"}},
                              ensure_ascii=False, indent=2))
    print("요청:", key)


def _meta() -> dict:
    key = json.loads(REQ.read_text())["download_key"]
    with urllib.request.urlopen(f"{GBIF_API}/occurrence/download/{key}", timeout=30) as r:
        return json.load(r)


def status() -> None:
    m = _meta()
    print({k: m.get(k) for k in ("key", "status", "doi", "totalRecords", "size")})


def fetch() -> None:
    m = _meta()
    if m.get("status") != "SUCCEEDED":
        print("아직:", m.get("status")); return
    dest = ADD_DIR / f"occurrences_{m['key']}.zip"
    subprocess.run(["curl", "-sSL", "--retry", "5", "-C", "-", "-o", str(dest), m["downloadLink"]], check=True)
    if dest.stat().st_size != m["size"] or not zipfile.is_zipfile(dest):
        raise RuntimeError("받은 파일 이상 — 다시 실행")
    rec = json.loads(REQ.read_text())
    rec.update({"doi": m.get("doi"), "total_records": m.get("totalRecords"), "file": dest.name,
                "fetched_at": datetime.now(timezone.utc).isoformat()})
    REQ.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print("저장:", dest, m.get("doi"), m.get("totalRecords"))


if __name__ == "__main__":
    {"request": request, "status": status, "fetch": fetch}[sys.argv[1]]()
