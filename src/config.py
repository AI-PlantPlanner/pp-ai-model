"""경로, 컬럼명 매핑 등 정규화/필터링 단계에서 공유하는 상수 모음."""

from pathlib import Path

# 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pp-rawdata-2차.xlsx"
RAW_SHEET_NAME = "100개 버전"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NORMALIZED_OUTPUT_PATH = PROCESSED_DIR / "plants_normalized.csv"
NORMALIZATION_ISSUES_PATH = PROCESSED_DIR / "normalization_issues.csv"

# 원본 헤더는 2번째 줄(스킵 대상 1개 행: Column1..17 placeholder 헤더)
RAW_HEADER_SKIPROWS = 1

# 원본 컬럼명
COL_NAME = "식물명 (국명)"
COL_SCIENTIFIC_NAME = "학명"
COL_FAMILY = "과"
COL_GROWTH_FORM = "생육형태"
COL_LIGHT_HOURS = "적정 광주기(하루 n시간)"
COL_LIGHT_LUX = "적정 광량(lux)"
COL_TEMP_OPTIMAL = "적정 온도(℃)"
COL_TEMP_LIMIT = "한계 온도(℃)"
COL_HUMIDITY = "적정 습도(%)"
COL_WATERING_FREQ = "물주기 빈도(주 n회)"
COL_FERTILIZING_FREQ = "시비 주기(월 n회)"
COL_PRUNING_FREQ = "전정(연 n회)"
COL_PEST_DISEASE = "병해충"
COL_SOURCE = "출처"
COL_REMARKS = "비고"
COL_CULTIVATION_TYPE = "재배구분(실내/실외/실내,실외)"
COL_APPRECIATION_POINT = "감상포인트 (관엽/관화)"

# ---- 단순 min/max 범위 파싱 대상 컬럼: 원본 컬럼명 -> 내부 접두어 ----
# 2차 데이터부터는 시비 주기도 괄호 메모 없이 순수 숫자/범위라 여기에 포함한다.
RANGE_COLUMNS = {
    COL_LIGHT_HOURS: "light_hours",
    COL_LIGHT_LUX: "light_lux",
    COL_TEMP_OPTIMAL: "temp_optimal",
    COL_TEMP_LIMIT: "temp_limit",
    COL_HUMIDITY: "humidity",
    COL_WATERING_FREQ: "watering_freq",
    COL_FERTILIZING_FREQ: "fertilizing_freq",
}

# ---- 숫자 범위 또는 "수시"(as-needed) 표현이 섞인 컬럼: 원본 컬럼명 -> 내부 접두어 ----
# 2차 데이터에서는 전정만 "수시" 값이 남아있다(그 외 컬럼은 전부 숫자/범위).
AS_NEEDED_RANGE_COLUMNS = {
    COL_PRUNING_FREQ: "pruning_freq",
}

# 숫자가 전혀 없는 "상시/수시" 표현으로 취급할 키워드
AS_NEEDED_KEYWORDS = ("수시",)

# ---- 재배구분 표준 카테고리 ----
CULTIVATION_INDOOR = "실내"
CULTIVATION_OUTDOOR = "실외"
CULTIVATION_MIXED = "실내·실외"
CULTIVATION_UNKNOWN = "미상"

# ---- plants_normalized.csv 컬럼명 (teacher_scoring에서 사용) ----
NORM_LIGHT_LUX_MIN = "light_lux_min"
NORM_LIGHT_LUX_MAX = "light_lux_max"
NORM_TEMP_OPTIMAL_MIN = "temp_optimal_min"
NORM_TEMP_OPTIMAL_MAX = "temp_optimal_max"
NORM_TEMP_LIMIT_MIN = "temp_limit_min"
NORM_TEMP_LIMIT_MAX = "temp_limit_max"
NORM_HUMIDITY_MIN = "humidity_min"
NORM_HUMIDITY_MAX = "humidity_max"
NORM_CULTIVATION_TYPE = "cultivation_type"

# ---- 룰 기반 티처 스코어링(v0) ----
# feature별 가중치(가중합) — 합이 1이 되도록 유지
SCORE_WEIGHT_LIGHT = 1 / 3
SCORE_WEIGHT_TEMP = 1 / 3
SCORE_WEIGHT_HUMIDITY = 1 / 3

# 적정범위를 벗어났을 때, 범위 폭의 이 비율만큼 벗어나면 0점까지 선형 감점
SCORE_TOLERANCE_RATIO = 0.5

# 습도처럼 "≥N"(상한 없음) 표기가 있는 컬럼에서, 하한 미달 시 감점 계산에 쓸 "가정 범위 폭".
# 데이터 내 습도 적정범위 폭(10~30, 중앙값 20)을 참고해 정했다 — 실제 범위 폭이 아니라
# 하한이 없는 case의 감점 기울기를 다른 종들과 비슷하게 맞추기 위한 근사치.
SCORE_HUMIDITY_ASSUMED_RANGE_WIDTH = 20
