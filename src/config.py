"""경로, 컬럼명 매핑 등 정규화/필터링 단계에서 공유하는 상수 모음."""

from pathlib import Path

# 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pp-rawdata-1차.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NORMALIZED_OUTPUT_PATH = PROCESSED_DIR / "plants_normalized.csv"
NORMALIZATION_ISSUES_PATH = PROCESSED_DIR / "normalization_issues.csv"

# 원본 CSV 헤더는 2번째 줄(스킵 대상 1개 행: 병합 헤더)
RAW_HEADER_SKIPROWS = 1

# 원본 컬럼명
COL_NAME = "식물명"
COL_SCIENTIFIC_NAME = "학명"
COL_TAXONOMY = "분류"
COL_LIGHT_HOURS = "적정 광주기(하루 n시간)"
COL_LIGHT_LUX = "적정 광량(lux)"
COL_TEMP_OPTIMAL = "적정 온도(℃)"
COL_TEMP_LIMIT = "한계 온도(℃)"
COL_HUMIDITY = "적정 습도(%)"
COL_WATERING_FREQ = "물주기 빈도(주 n회)"
COL_FERTILIZING_FREQ = "시비 주기(월 n회)"
COL_PRUNING_FREQ = "전정(연 n회)"
COL_PEST_DISEASE = "병해충"
COL_REMARKS = "비고"
COL_CULTIVATION_TYPE = "재배구분(실내/실외/실내·실외)"
COL_APPRECIATION_POINT = "감상포인트 (관엽/관화)"

# ---- 단순 min/max 범위 파싱 대상 컬럼: 원본 컬럼명 -> 내부 접두어 ----
RANGE_COLUMNS = {
    COL_LIGHT_HOURS: "light_hours",
    COL_LIGHT_LUX: "light_lux",
    COL_TEMP_OPTIMAL: "temp_optimal",
    COL_TEMP_LIMIT: "temp_limit",
    COL_HUMIDITY: "humidity",
    COL_WATERING_FREQ: "watering_freq",
}

# ---- "숫자(텍스트)" 패턴 컬럼: 원본 컬럼명 -> 내부 접두어 ----
COUNT_WITH_NOTE_COLUMNS = {
    COL_FERTILIZING_FREQ: "fertilizing_freq",
    COL_PRUNING_FREQ: "pruning_freq",
}

# 숫자가 전혀 없는 "상시/수시" 표현으로 취급할 키워드
AS_NEEDED_KEYWORDS = ("수시",)

# ---- 재배구분 표준 카테고리 ----
CULTIVATION_INDOOR = "실내"
CULTIVATION_OUTDOOR = "실외"
CULTIVATION_MIXED = "실내·실외"
CULTIVATION_UNKNOWN = "미상"
