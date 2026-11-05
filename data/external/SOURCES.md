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
