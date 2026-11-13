# 외부 참조 자료 출처

원본 파일은 용량 때문에 git에서 제외(.gitignore). 아래 URL·버전으로 다시 받을 수 있다.

## WCVP (World Checklist of Vascular Plants)
- 용도: GBIF 기록이 해당 종의 원산 지역 안인지(원산) 밖인지(도입) 표시 — 원산지만/귀화 포함 비교 실험(결정 A)
- 받은 곳: https://sftp.kew.org/pub/data-repositories/WCVP/wcvp.zip (서버 Last-Modified 2026-06-04)
- 받은 날: 2026-11-05
- 파일: wcvp_names.csv, wcvp_distribution.csv, README_WCVP.xlsx (구분자 `|`)
- 분포 컬럼: area_code_l3 (TDWG WGSRPD 3단계), introduced (0 원산 / 1 도입), extinct, location_doubtful
- 인용(README_WCVP.xlsx 기재 그대로): Govaerts R (ed.). 2026. WCVP: World Checklist of Vascular Plants. Facilitated by the Royal Botanic Gardens, Kew. https://doi.org/10.34885/egs6-cp24 [accessed 04 Jun 2026].
- 방법 논문: Govaerts R. et al. (2021) The World Checklist of Vascular Plants, a continuously updated resource for exploring global plant diversity. Scientific Data 8:215. https://doi.org/10.1038/s41597-021-00997-6

## TDWG WGSRPD 3단계 지역 경계
- 용도: GBIF 좌표를 WCVP 분포 단위(area_code_l3)로 바꾸기
- 받은 곳: https://raw.githubusercontent.com/tdwg/wgsrpd/master/geojson/level3.geojson
- 받은 날: 2026-11-05
- 인용: Brummitt R.K. (2001) World Geographical Scheme for Recording Plant Distributions, Edition 2. TDWG. (인용 형식은 투고 전 TDWG 저장소에서 재확인)

## ERA5-Land Daily Aggregated (Google Earth Engine)
- 용도: 칸별 일 기상(기온 평균·최저·최고, 이슬점, 일사량) 2016–2025, 연 극값 1991–2025(연도 실험)
- 받은 곳: GEE `ECMWF/ERA5_LAND/DAILY_AGGR`, 프로젝트 ai-plant-planner(비상업 커뮤니티 등급)
- 받은 날: 2026-11-06(연 극값, 일괄 내보내기) ~ 2026-11-07(일별, 즉시 요청 computeFeatures)
- 격자: 0.1°, 칸 중심 = 0.1° 배수(transform [0.1,0,-180.05,0,-0.1,90.05]). 육지 칸만 값 있음 → 해안 칸은 src/coast_fill.py
- 인용: Muñoz Sabater J. (2019) ERA5-Land hourly data from 1950 to present. Copernicus C3S CDS. doi:10.24381/cds.e2161bac (일 집계본 DOI는 투고 전 확인)
  방법 논문: Muñoz-Sabater J. et al. (2021) ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst. Sci. Data 13:4349–4383. doi:10.5194/essd-13-4349-2021
- 원본 산출물은 data/weather/(git 제외). 요청·사용량 기록만 data/weather/gee_tasks.jsonl로 추적.
