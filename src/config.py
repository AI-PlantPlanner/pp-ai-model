"""경로, 컬럼명 매핑, 채점 기준 상수 등 파이프라인 전 단계에서 공유하는 상수 모음.

## 채점 기준의 출처
2단계 채점 규칙(SCORE_*)은 원예팀이 전달한 '그룹별 가중치'/'그룹별 판정 기준' 시트를 근거로 한다.
다만 원예팀 답변에는 아직 확정되지 않은 부분이 있어(③ 생육 정지와 ④ 피해 발생이 한 칸에 병기됨 등),
그 부분은 AI팀 판단으로 잠정 결정하고 원예팀에 확인 요청을 보낸 상태다.
답변이 오면 이 파일의 상수만 교체하고 파이프라인을 재실행하면 되도록,
판정 경계값은 로직에 상수를 직접 쓰지 않고 전부 여기에 모아둔다.
어느 값이 확정이고 어느 값이 잠정인지는 각 상수 주석에 표시했다.
"""

from pathlib import Path

# 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pp-rawdata-3차.xlsx"
# 3차 파일 안의 시트명은 2차 시절 이름 그대로 남아있다(원본 무수정 방침이라 우리가 고치지 않음).
RAW_SHEET_NAME = "raw data - pp-rawdata-2차"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NORMALIZED_OUTPUT_PATH = PROCESSED_DIR / "plants_normalized.csv"
NORMALIZATION_ISSUES_PATH = PROCESSED_DIR / "normalization_issues.csv"
# 적정 광량과 광보상점/광포화점이 서로 모순되는 종 목록(원예팀 확인 요청용 별첨).
LIGHT_CONFLICT_PATH = PROCESSED_DIR / "light_range_conflicts.csv"

# 3차 원본은 1행 제목 + 2행 Column1..18 placeholder 뒤 3행이 실제 헤더다.
RAW_HEADER_SKIPROWS = 2

# ---- 원본 컬럼명 ----
COL_NAME = "식물명 (국명)"
COL_SCIENTIFIC_NAME = "학명"
COL_FAMILY = "과"
COL_GROWTH_FORM = "생육형태"
COL_LIGHT_HOURS = "적정 광주기(하루 n시간)"
COL_LIGHT_LUX = "적정 광량(lux)"
COL_TEMP_OPTIMAL = "적정 온도(℃)"
COL_HUMIDITY = "적정 습도(%)"
COL_WATERING_FREQ = "물주기 빈도(주 n회)"
COL_FERTILIZING_FREQ = "시비 주기(월 n회)"
COL_PRUNING_FREQ = "전정(연 n회)"
COL_PEST_DISEASE = "병해충"
COL_SOURCE = "출처"
COL_REMARKS = "비고"
COL_CULTIVATION_TYPE = "재배구분(실내/실외/실내,실외)"
COL_APPRECIATION_POINT = "감상포인트 (관엽/관화)"

# ---- 3차에서 추가된 컬럼 ----
# 2차의 "한계 온도(℃)" 단일 컬럼이 상·하한으로 분리됐다.
COL_TEMP_LIMIT_LOWER = "한계 온도 하한(℃)"
COL_TEMP_LIMIT_UPPER = "한계 온도 상한(℃)"
COL_LIGHT_SATURATION = "광포화점(lux)"
COL_LIGHT_COMPENSATION = "광보상점(lux)"
COL_CLIMATE_TYPE = "기후형(열대/ 열대·아열대/아열대/온대)"
# 원예팀이 채워준 생육형 그룹. 우리가 만들어둔 빈 '생육형그룹' 컬럼 대신 이 컬럼에 값이 들어왔다.
COL_PLANT_GROUP = "감상 상세구분(다육·선인장류/목본류/관엽/초본 화훼)"

# ---- 단순 min/max 범위 파싱 대상 컬럼: 원본 컬럼명 -> 내부 접두어 ----
RANGE_COLUMNS = {
    COL_LIGHT_HOURS: "light_hours",
    COL_LIGHT_LUX: "light_lux",
    COL_LIGHT_SATURATION: "light_saturation",
    COL_LIGHT_COMPENSATION: "light_compensation",
    COL_TEMP_OPTIMAL: "temp_optimal",
    COL_TEMP_LIMIT_LOWER: "temp_limit_lower",
    COL_TEMP_LIMIT_UPPER: "temp_limit_upper",
    COL_HUMIDITY: "humidity",
    COL_WATERING_FREQ: "watering_freq",
    COL_FERTILIZING_FREQ: "fertilizing_freq",
}

# ---- 숫자 범위 또는 "수시"(as-needed) 표현이 섞인 컬럼: 원본 컬럼명 -> 내부 접두어 ----
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
NORM_LIGHT_SATURATION_MIN = "light_saturation_min"
NORM_LIGHT_SATURATION_MAX = "light_saturation_max"
NORM_LIGHT_COMPENSATION_MIN = "light_compensation_min"
NORM_LIGHT_COMPENSATION_MAX = "light_compensation_max"
NORM_TEMP_OPTIMAL_MIN = "temp_optimal_min"
NORM_TEMP_OPTIMAL_MAX = "temp_optimal_max"
NORM_TEMP_LIMIT_LOWER = "temp_limit_lower_min"
NORM_TEMP_LIMIT_UPPER = "temp_limit_upper_max"
NORM_HUMIDITY_MIN = "humidity_min"
NORM_HUMIDITY_MAX = "humidity_max"
NORM_CULTIVATION_TYPE = "cultivation_type"
NORM_PLANT_GROUP = "plant_group"

# ---- 생육형 그룹 ----
# raw data 시트와 '그룹별 가중치' 시트의 그룹명 표기가 달라(관엽 vs 관엽식물 등) 여기서 통일한다.
GROUP_FOLIAGE = "관엽식물"
GROUP_SUCCULENT = "다육·선인장류"
GROUP_WOODY = "목본류"
GROUP_HERBACEOUS_FLOWER = "초본 화훼류"

PLANT_GROUP_ALIASES = {
    "관엽": GROUP_FOLIAGE,
    "관엽식물": GROUP_FOLIAGE,
    "다육·선인장류": GROUP_SUCCULENT,
    "다육, 선인장류": GROUP_SUCCULENT,
    "목본": GROUP_WOODY,
    "목본류": GROUP_WOODY,
    "초본 화훼": GROUP_HERBACEOUS_FLOWER,
    "초본 화훼류": GROUP_HERBACEOUS_FLOWER,
}

# ---- 2단계: 그룹별 가중치 (원예팀 '그룹별 가중치' 시트, 확정값) ----
# (광량, 온도, 습도) — 합계 100. 채점 시 100으로 나눠 쓴다.
GROUP_WEIGHTS = {
    GROUP_FOLIAGE: (30, 40, 30),
    GROUP_SUCCULENT: (35, 40, 25),
    GROUP_WOODY: (40, 40, 20),
    GROUP_HERBACEOUS_FLOWER: (40, 40, 20),
}
# 그룹 미상 종에 쓸 기본 가중치(동일 비율). 3차 데이터는 100종 전부 그룹이 채워져 있어
# 현재는 쓰이지 않지만, 종 확장 시 그룹 누락을 대비해 남겨둔다.
DEFAULT_GROUP_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

# ---- 2단계: 4단계 판정 -> 점수 변환 ----
# 원예팀은 ① 거의 문제없음 ~ ④ 피해 발생의 4단계만 표기하고, 점수 변환은 AI팀이 맡기로 했다.
#   ① 적정범위 안        -> 100점
#   ② 생육 둔화 구간 끝  -> SCORE_LEVEL2_END (100점에서 여기까지 선형 감점)
#   ③ 생육 정지 구간 끝  -> 0점
#   ④ 피해 발생          -> 게이트(총점 0점)
# ④를 적용하는 조건은 원예팀이 「0점 기준」으로 확정한 두 가지(한계온도 이탈, 재배구분 불일치)뿐이고,
# 판정 기준 시트에서 "③/④"로 병기된 나머지 구간은 ③(감점)으로 처리한다 — AI팀 잠정 판단.
SCORE_LEVEL2_END = 40.0

# 광량: 종별 요구량이 100배까지 차이나서(적정 하한 300 ~ 상한 70,000lux) 절대 lux 차이로 감점하면
# 요구량이 큰 종이 어두운 환경에서도 높은 점수를 받는다. 적정범위 대비 "배율"로 감점한다.
#   적정범위의 SCORE_LIGHT_RATIO_LEVEL2 배까지 -> ② 생육 둔화
#   SCORE_LIGHT_RATIO_LEVEL3 배까지            -> ③ 생육 정지
# 원예팀이 구간을 배율로 지정해준 것은 아니라 AI팀 잠정값이다.
SCORE_LIGHT_RATIO_LEVEL2 = 2.0
SCORE_LIGHT_RATIO_LEVEL3 = 5.0

# 습도: 원예팀 기준은 "적정범위~80%가 ②, 80% 이상이 ③/④"(하한 쪽은 10~20%).
# 적정 습도 상한이 80% 이상인 종이 9종 있어 적정범위와 피해구간이 겹치므로,
# 80% 고정이 아니라 "80%와 종별 적정 상한 중 높은 쪽"을 ② 구간 끝으로 쓴다 — AI팀 잠정 판단.
SCORE_HUMIDITY_LEVEL2_UPPER = 80.0
SCORE_HUMIDITY_LEVEL2_LOWER = 20.0
# 적정범위와 ②구간 끝이 겹칠 때 확보할 최소 완충 폭(%p). 없으면 100점에서 곧바로 급락한다.
SCORE_HUMIDITY_MIN_BAND = 10.0

# 온도: 원예팀 기준은 "적정~한계온도 사이가 ②, 한계온도 밖이 0점".
# 다만 한계온도 '상한'이 41~50℃(4그룹 3종류)로 생육 한계보다 치사 온도에 가까워 보여,
# 이걸 ② 구간 끝으로 그대로 쓰면 35℃에서도 감점이 거의 없다.
# ② 구간은 적정온도 범위 폭(최소 SCORE_TEMP_MIN_BAND)만큼만 잡고, 한계온도까지는 ③으로 둔다.
# 한계온도 상한의 근거는 원예팀에 확인 요청한 상태다.
SCORE_TEMP_MIN_BAND = 5.0
# ②/③ 경계가 한계온도에 붙어 ③ 구간이 사라지는 것을 막기 위해, 적정~한계 구간의 이 비율만큼은 ③으로 남긴다.
SCORE_TEMP_LEVEL3_MIN_RATIO = 0.3

# ---- 2단계: 누적 환경 입력 ----
# 게임에서 일정 기간(3개월/6개월 등) 측정·계산한 환경값 배열을 받는다.
# 단일 시점 스냅샷보다 누적값이 타당하다는 게임팀 판단에 따른 구조다.
#
# 감점 채점(광량/온도/습도)은 기간 평균을 쓰고, 게이트 판정(한계온도)만 기간 극값을 쓴다.
# 게이트까지 평균으로 판정하면 겨울을 포함한 기간에서 한계온도 이탈이 평균에 묻힌다 —
# 실측 결과 6개월 평균 기준으로는 24종, 1년 평균 기준으로는 28종이 겨울에 죽는데도
# "적합"으로 추천됐다. 한계온도는 평균이 아니라 극값 문제라서다.
# (여름처럼 기간 내 변동이 작으면 평균과 극값이 같아 차이가 없다.)
ENV_WINDOW_DAYS = 90

# ---- 4단계: 합성 환경 데이터 샘플링 범위 ----
# "방/주방/거실/마당" 세부 구분은 제외, 실내/실외 여부만 반영.
# TODO(출처 보강): 아래 범위는 조사 자료 기반 잠정치인데 근거 문헌이 남아있지 않다.
# 기상청 평년값(온도/습도) 등 확인 가능한 출처로 교체하고 여기에 링크를 남길 것.
# 진단 모델(사진/센서 -> 환경값 추정) 스펙이 확정되면 실제 입력 분포로 다시 조정해야 한다.

# 실내 광량: 창가 위치(창문과의 거리/방향) 기준. 실내 종 light_lux_max가 37,700까지
# 있어(선룸 등) 상한을 40,000으로 잡았다.
SYNTHETIC_INDOOR_LIGHT_LUX_RANGE = (100, 40_000)

# 실내 온도: 혼합종 한계온도가 -25℃, 실내 종 적정온도 상한이 32℃까지 있어 하한 -10(냉난방
# 끊긴 극단 상황 포함) ~ 상한 33으로 잡았다.
SYNTHETIC_INDOOR_TEMP_RANGE = (-10, 33)
SYNTHETIC_INDOOR_HUMIDITY_RANGE = (30, 80)

# 실외 광량: 흐린 날/그늘(약 1,000lux)부터 직사광선(최대 약 120,000lux)까지.
SYNTHETIC_OUTDOOR_LIGHT_LUX_RANGE = (1_000, 120_000)

# 실외 온도/습도: 한국 사계절 변동 반영.
SYNTHETIC_OUTDOOR_TEMP_RANGE = (-10, 35)
SYNTHETIC_OUTDOOR_HUMIDITY_RANGE = (30, 90)

# 실내/실외 모두 800. 실외는 원래 300이었는데, 광량 범위가 1,000~120,000lux로 실내보다
# 넓은데도 샘플이 적어 고온·고습·강광이 겹치는 한여름 구간의 모델 근사 오차가 컸다(최대 8.4점).
SYNTHETIC_N_SAMPLES_INDOOR = 800
SYNTHETIC_N_SAMPLES_OUTDOOR = 800
SYNTHETIC_RANDOM_SEED = 42

# 기간 내 일별 변동폭(표준편차). 같은 평균이라도 변동이 크면 한계온도 이탈 가능성이 달라지므로,
# 평균만 흔들지 말고 변동폭도 함께 샘플링해야 게이트가 학습된다.
# 실내는 냉난방으로 변동이 작고, 실외는 계절·일교차로 크다.
# TODO(출처 보강): SYNTHETIC_* 전반과 마찬가지로 근거 문헌 필요. 향후 기상청 평년값 및
# 실측 프로파일로 대체 예정(README 참고사항 참고).
SYNTHETIC_INDOOR_TEMP_SPREAD_RANGE = (0.5, 4.0)
SYNTHETIC_OUTDOOR_TEMP_SPREAD_RANGE = (2.0, 10.0)
SYNTHETIC_INDOOR_HUMIDITY_SPREAD_RANGE = (2.0, 10.0)
SYNTHETIC_OUTDOOR_HUMIDITY_SPREAD_RANGE = (5.0, 18.0)
# 광량은 로그 스케일로 다루므로 변동폭도 로그 공간의 표준편차로 준다.
SYNTHETIC_INDOOR_LIGHT_LOG_SPREAD_RANGE = (0.1, 0.6)
SYNTHETIC_OUTDOOR_LIGHT_LOG_SPREAD_RANGE = (0.2, 0.9)

SYNTHETIC_OUTPUT_PATH = PROCESSED_DIR / "synthetic_training_data.csv"

# LLM으로 생성한 환경 프로파일(현실적인 배치 상황별 광량/온도/습도 통계량).
# src/env_profiles.py가 생성·검증하고, 4단계가 이 파일이 있으면 균등 샘플링 대신 이걸 쓴다.
# 파일이 없으면 위 SYNTHETIC_*_RANGE 기반 균등 샘플링으로 폴백한다(비교 베이스라인).
ENV_PROFILES_PATH = PROCESSED_DIR / "environment_profiles.json"

# ---- 5단계: 모델 학습 ----
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_PATH = MODELS_DIR / "recommendation_model_v0.joblib"
MODEL_OUTPUT_PATH_V1 = MODELS_DIR / "recommendation_model_v1.joblib"
# v2: 3차 데이터(그룹 가중치 + 4단계 판정 기준) 기준으로 재학습한 버전.
MODEL_OUTPUT_PATH_V2 = MODELS_DIR / "recommendation_model_v2.joblib"
MODEL_TEST_SIZE = 0.2
MODEL_RANDOM_SEED = 42
# humidity_max가 NaN인 건("≥70"처럼 상한 없는 개방형 범위) 파싱 실패가 아니라 "습도는
# 100%(물리적 상한)까지 전부 적정"이라는 뜻이라, 모델 입력 피처로는 100으로 채운다.
MODEL_HUMIDITY_MAX_FILL = 100.0
