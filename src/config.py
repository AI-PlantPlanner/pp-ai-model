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

# ---- 4단계: 합성 환경 데이터 샘플링 범위 ----
# "방/주방/거실/마당" 같은 세부 공간 구분은 제외하고 AI 모델 입력에는 실내/실외 여부만 반영한다.
# 아래 범위는 조사 자료 기반 잠정치 — 진단 모델(사진/센서 → 환경값 추정) 스펙이 확정되면 실제 입력 분포에 맞춰 다시 조정해야 한다.
#
# 실내 광량: 방 종류가 아니라 "창문과의 거리/방향"이 결정 요인이라는 조사 결과에 따라
# 창가 위치 기준(구석~직사광 통과)으로 잡았다. KS 조도 설계기준(거실 100~200lux 등)은
# 인공조명 설계용이라 식물이 실제로 받는 채광과는 다르므로 쓰지 않았다.
# 상한은 20,000이었으나 plants_normalized.csv 기준 실내 종의 light_lux_max가 최대
# 37,700까지 있어(햇빛 잘 드는 남향 창가/선룸 등) 범위를 넓혔다 — 기존 범위로는 그런
# 고광량 실내 종의 적정 상한 근처를 학습 데이터가 아예 다뤄본 적이 없었다.
SYNTHETIC_INDOOR_LIGHT_LUX_RANGE = (100, 40_000)

# 실내 온도/습도: 계절별 냉난방 기준(겨울 18~20℃/40~50%, 여름 22~26℃/50~60%)에 주방 조리 시
# 습도 상승분까지 여유를 둔 범위였으나, plants_normalized.csv 기준 실내·혼합 종의
# temp_limit_max(한계온도)가 최대 15℃까지 있어 기존 하한(18)이 모든 실내 종의 한계온도보다
# 항상 높았다 — 즉 냉난방이 항상 잘 되는 상황만 가정하다 보니 실내 컨텍스트에서는 온도
# 게이트(생존 한계)가 학습 데이터에서 단 한 번도 발동해본 적이 없었다. 난방 없는 베란다/
# 창가처럼 실내로 분류되지만 저온에 노출되는 경우와, 냉방 없는 여름 실내 고온(최대 33℃
# 부근, 실내 종 temp_optimal_max 최대 32와 정합)까지 포괄하도록 범위를 넓혔다(5~33).
#
# 그 다음 재점검에서, 혼합('실내·실외') 종은 temp_limit_max가 -25℃까지 내려가는데(예:
# 오색마삭줄 -5℃) 실내 컨텍스트 하한(5)이 여전히 이보다 높아, "혼합 종 x 실내 컨텍스트 x
# 영하권" 조합에서 근사오차가 최대 30~37점까지 남는 게 확인됐다(게이트가 아니라 연속
# 점수 구간이라 predict.py의 안전장치로도 못 잡는다). 실외 하한(-10)과 동일한 근거로
# — 냉난방이 완전히 끊긴 실내(정전, 단열 없는 창고 등) 극단 상황까지 고려해 하한을
# -10으로 더 낮췄다. 다만 -25처럼 그보다도 낮은 한계온도를 가진 극소수 종(애기달맞이꽃
# -25, 돌단풍/레몬밤 -20 등)은 실외 하한(-10)도 마찬가지로 못 미쳐 실외 컨텍스트에서도
# 게이트가 검증된 적이 없다 — 이건 기존 실외 범위 설계와 동일한 논리(한국 실제 기후에서
# 그 정도로 내려가는 경우가 드물어 실익이 낮음)로 지금은 의도적으로 남겨둔 한계다.
SYNTHETIC_INDOOR_TEMP_RANGE = (-10, 33)
SYNTHETIC_INDOOR_HUMIDITY_RANGE = (30, 80)

# 실외 광량: 흐린 날/그늘(약 1,000lux)부터 맑은 날 직사광선(최대 약 120,000lux)까지.
# plants_normalized.csv 기준 실외 종 light_lux_max 최대 70,000을 이미 여유 있게 포괄한다.
SYNTHETIC_OUTDOOR_LIGHT_LUX_RANGE = (1_000, 120_000)

# 실외 온도/습도: 한국 사계절 변동 반영(여름 평균습도 79.9%, 봄가을 63.6% 등).
SYNTHETIC_OUTDOOR_TEMP_RANGE = (-10, 35)
SYNTHETIC_OUTDOOR_HUMIDITY_RANGE = (30, 90)

# 실내 범위(광량/온도)가 넓어진 만큼, 같은 샘플 수(300)로는 단위 면적당 밀도가 낮아져
# 게이트 경계 근처가 성기게 학습될 수 있어 실내 샘플 수를 늘렸다. 온도 하한을 5->-10으로
# 한 번 더 넓히면서(범위 폭 28->43) 밀도 유지를 위해 800으로 추가 상향. 실외는 범위를
# 바꾸지 않았으므로 그대로 둔다.
SYNTHETIC_N_SAMPLES_INDOOR = 800
SYNTHETIC_N_SAMPLES_OUTDOOR = 300
SYNTHETIC_RANDOM_SEED = 42

SYNTHETIC_OUTPUT_PATH = PROCESSED_DIR / "synthetic_training_data.csv"

# ---- 5단계: 모델 학습 ----
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_PATH = MODELS_DIR / "recommendation_model_v0.joblib"
# v1: 원본 값만 쓰던 v0에 사용자값-적정범위 거리(마진)/게이트 근접도 피처를 추가한 버전.
MODEL_OUTPUT_PATH_V1 = MODELS_DIR / "recommendation_model_v1.joblib"
MODEL_TEST_SIZE = 0.2
MODEL_RANDOM_SEED = 42
# humidity_max가 NaN인 건("≥70"처럼 상한 없는 개방형 범위) 파싱 실패가 아니라 "습도는
# 100%(물리적 상한)까지 전부 적정"이라는 뜻이라, 모델 입력 피처로는 100으로 채운다 —
# 임의 추정이 아니라 teacher_scoring이 이 경우를 다루는 것과 동일한 논리를 수치화한 것뿐이다.
MODEL_HUMIDITY_MAX_FILL = 100.0
