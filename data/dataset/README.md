# 학습 데이터 (백로그 3번 데이터 구축 결과)

GBIF 자생지 기록과 ERA5-Land 기상을 원예 100종에 연결한 데이터. 모델 단계(백로그 6번)는 이 폴더와
`data/weather/daily_features/`만 읽으면 된다. 수치는 `build_report.json`(2026-11-07 생성) 기준.

## 파일
| 파일 | 내용 | 키 |
|---|---|---|
| `plants.csv` | 원예 식물 100개: 학명, GBIF 종 키, 품종 여부, 버전별 칸 수, 학습 세트 포함 여부, 제외 사유 | 식물명, count_key |
| `species_cells.parquet` | GBIF 종 × 0.1° 칸 (94천~145천 쌍): 위경도, 기상 칸 src_id, 해안 대체 거리, 버전별 기록 수 | count_key, cell_id |
| `build_report.json` | 세트별 식물·종·칸·기록 수, 기상 연결 검증 결과 | |
| `../weather/daily_features/<연도>/pNN.parquet` | 기상 칸 × 연도(2016–2025) × 1~365일: t_mean, t_min, t_max(℃), rh_mean(%), ssrd(MJ/m²), day_length_h | src_id, year, day |

기상 조각 번호 = `src_id // 5000`. `.parquet`는 git에서 제외되므로 아래 순서로 다시 만든다.

## 만드는 순서
1. `src/gbif_taxon_match.py` → `gbif_download.py` (DOI 10.15468/dl.fdf4xu, 819,609건)
2. `src/gbif_native_flag.py` (WCVP + WGSRPD 3단계로 원산/도입 표시)
3. `src/gbif_clean.R` (CoordinateCleaner 3.0.1 기본값) → `src/gbif_apply_clean.py` (정제·버전 6개)
4. `src/gee_annual_extremes.py` → `src/year_window_experiment.py` (기간 결정), `src/min_records_test.py` (D 결정)
5. `src/coast_fill.py` (해안 칸 50 km 대체) → `src/gee_daily.py` (일 기상 10년) → `src/daily_features.py`
6. `src/build_dataset.py` (이 폴더)

## 결정과 근거 (자세한 내용은 각 스크립트 머리말)
| 항목 | 결정 | 근거 |
|---|---|---|
| 좌표 정제 | SAGE 부록 A.2 검사 + CC 기본값, 불확도 ≤1000 m 또는 미기재, 이상치 검사 안 함 | Arens et al. 2026(arXiv 2609.31082), Zizka et al. 2019 |
| 도시 기록 | 본 분석 제외, 포함 버전(v_urban) 비교 | 원산 기록 중 도시 비율 팔손이 84%·수국 66%(실측), Zizka et al. 2020 |
| 원산 | WCVP 원산 지역 안 기록만(본), 도입 포함(v_intro)·전체(v_all) 비교 | SAGE, Brown et al. 2023(rWCVP) |
| 해안 경계 | 원산 다각형 1 km·20 km 안 포함 버전 비교 | 실측(경계 바로 밖 기록) |
| 칸 | ERA5-Land 0.1° 격자, 종마다 칸당 1번 | SAGE "환경 자료 해상도로 묶기" |
| 기상 기간 | 10년(2016–2025), 30년은 감도 비교용 | 연도 실험: 1년은 더위 쪽 일치도 낮음(0.18), 10년·30년 동등. 기록 82%가 2015년 이후 |
| 해안 칸 | 50 km 안 가장 가까운 육지 칸 값으로 대체, 없으면 제외(135쌍) | SAGE 결측 대체 규칙 |
| 최소 칸 수 D | 25(본), 14(감도) | 반복 표본 실험(min_records_test.py), van Proosdij et al. 2016 |
| 이명 | homotypic만 병합, 섬초롱꽃(heterotypic) 제외 | SAGE |
| 품종 | 원종 기록으로 근사(본), 품종 제외 세트 비교 | 11-07 결정. 근사 가정의 영향을 실험으로 확인 |
| 상대습도 | Magnus 식, Alduchov & Eskridge 1996 식 21 계수 | 일평균 기온·이슬점 사용(하루 변동 미반영) |
| 낮 길이 | Forsythe et al. 1995 CBM, p=0.8333° | 일출~일몰 정의 |
| 일사량 | 변환 없이 보관(광량은 규칙이 담당) | DLI 환산 계수가 출처마다 다름 |

## 학습 세트 (`plants.csv`의 set_* 열, 또는 `src.build_dataset.eligible()`)
| 세트 | 식물 | GBIF 종 | 종·칸 쌍 | 칸 |
|---|---|---|---|---|
| 본 분석 (v_main, D25, 품종 근사) `set_main` | 76 | 75 | 94,091 | 63,629 |
| D14 감도 `set_main_d14` | 82 | 81 | 94,202 | 63,677 |
| 품종 제외 `set_no_cultivar` | 68 | 68 | 93,135 | 62,983 |
| v_urban 감도 | 76 | 75 | 106,281 | 68,145 |
| v_intro 감도 | 79 | 78 | 129,845 | 76,831 |
| v_all 감도 | 84 | 83 | 144,582 | 81,226 |
| v_coast1 감도 | 76 | 75 | 94,799 | 64,119 |
| v_coast20 감도 | 76 | 75 | 95,797 | 64,830 |

식물 수와 종 수가 1 차이 나는 이유: 디펜바키아와 디펜바키아 '트로픽스노우'가 같은 종(*Dieffenbachia seguine*)을 가리킨다.
품종 제외 세트가 8개 적은 이유: 품종 11개 중 본 세트에 든 것이 8개다(칼란디바는 칸 2개, 콩고·마리안느는 속명만 있음).

## 검증 (build_report.json)
- 칸 표의 기상 칸 84,340개가 2016–2025 모든 해에 빠짐없이 있음. 조각 번호 규칙 오류 0(표본 200).
- 일 기상: 결측 0, t_min > t_max 0, 칸·해마다 365일(2월 29일 제외).
- 즉시 요청 값 = 일괄 내보내기 값(2025년 7월, 1,000칸, 최대 차이 0).

## 제외된 식물 24개 (plants.csv의 note)
- 감도 세트(D14)에만 포함 6개: 둥근잎꿩의비름, 아글라오네마, 금유선인장, 피나타라벤더, 삭소롬, 아레카야자
- 원산 칸 부족(<14) 6개, 원산 기록 없음 1개(산취선인장)
- 원예팀 확인 대기 6개: 속명만(카틀레아·필로덴드론·덴드로비움·헬리코니아), 구즈베리, 마지나타 — 답이 오면 GBIF 추가 다운로드 후 이 단계부터 다시 만든다
- 해결 불가 5개: 원예 품종 속명만(콩고·마리안느), 인공 교배종(거북알로카시아), WCVP Unplaced(아펠란드라), heterotypic 이명(섬초롱꽃)

## 한계 (논문에 적을 것)
- 기상은 실외 재분석 자료(ERA5-Land 0.1°)라 극값이 관측소보다 평활되고, 실내 환경과 다르다.
- 상대습도는 일평균값으로 계산해 하루 안의 최저 습도를 반영하지 못한다.
- 남반구 계절 처리는 모델 단계에서 정한다(day_length_h 제공).
- 품종은 원종의 내성 범위를 가진다고 가정했다(품종 제외 세트로 영향 확인).
