"""환경값을 한 시점 스냅샷이 아니라 여러 시점 기록을 누적한 평균으로 모델에 넣는 로직.

## 왜 필요한가
지금까지는 사용자가 "지금 이 순간" 조건 하나(예: 창가 근처/보통 온도/보통 습도)를 한 번 고르면
그 값을 그대로 모델에 넣었습니다. 그런데 광량/온도/습도는 계절과 날짜에 따라 계속 바뀌기
때문에, 한 시점 스냅샷보다 일정 기간(예: 3개월) 동안 사용자가 남긴 기록들의 평균값으로
추천하는 게 더 타당하다는 게 이 모듈을 만든 이유입니다.

## 기록(entry)이 실제로 어떻게 쌓이는가
지금 앱에는 센서가 없어서 자동으로 값이 쌓이지 않습니다. 현실적인 방식은 "사용자가 주기적으로
(예: 2주~한 달마다) input_options.json 선택지를 다시 골라 기록을 남기는" 것입니다. 이 모듈은
그렇게 쌓인 기록 리스트를 받아서 "모델에 넣을 EnvironmentWindow"로 변환하는 순수 계산만
담당하고, 기록을 어디에(Unity 로컬 저장/서버 DB) 저장할지는 이 모듈의 관심사가 아닙니다 —
호출부가 저장소에서 기록을 꺼내 여기 넘겨주면 됩니다.

## 평균/극값 계산을 직접 하지 않는 이유 (유리님 쪽 구조로 교체)
원래는 이 모듈이 직접 light/temp/humidity 평균을 냈는데, 유리님이 src/teacher_scoring.py에
정확히 같은 목적의 summarize_environment()/EnvironmentWindow를 만들면서 한계온도 게이트는
평균이 아니라 "기간 중 최저/최고"(극값)로 판정해야 한다는 게 명확해졌습니다 — 겨울 한파 며칠이
평균에 묻히면 실제로는 겨울에 죽는 식물이 "적합"으로 추천되는 문제가 있었기 때문입니다
(config.ENV_WINDOW_DAYS 주석에 실측 수치가 있습니다: 6개월 평균 기준 24종, 1년 평균 기준
28종이 겨울에 죽는데도 적합 판정됨). 이 모듈이 자체적으로 평균만 내던 예전 방식은 극값을
아예 만들지 않아 같은 문제가 재현되므로, 지금은 기록별 값을 배열로 모아
summarize_environment()에 그대로 넘기고 극값 계산까지 위임합니다.

## 아직 팀 상의가 필요한 것
- 누적 기간(WINDOW_DAYS)은 src/config.py의 ENV_WINDOW_DAYS(현재 90일/3개월)를 그대로
  따릅니다. 유리님 쪽에서 실측 근거(6개월/1년 평균 시 겨울철 오추천 종수)까지 남겨서 정한
  값이라, 이 모듈에서 별도 값을 갖지 않고 그 값을 그대로 참조합니다 — 나중에 그 값이 바뀌면
  여기도 코드 수정 없이 자동으로 따라갑니다.
- 재배구분(실내/실외)은 평균을 낼 수 있는 값이 아니라서, 기간 내 가장 최근 기록의 재배구분을
  그대로 씁니다(예: 사용자가 최근에 화분을 실내로 옮겼다면 실내 기준으로 판단). 기간 중간에
  실내/실외가 섞여 있는 경우를 어떻게 다룰지는 더 상의가 필요합니다.
- 기록이 하나도 없을 때는 None을 반환합니다 — 호출부(Unity)가 "환경값을 먼저 입력해주세요"
  안내를 하거나, 그때만 예전처럼 1회성 선택 화면으로 보내야 합니다.
"""

from datetime import datetime, timedelta
from typing import Optional

from src.config import ENV_WINDOW_DAYS
from src.teacher_scoring import EnvironmentWindow, summarize_environment
from ui_config.input_resolver import load_input_options, resolve_model_input

DEFAULT_WINDOW_DAYS = ENV_WINDOW_DAYS  # src/config.py 값을 그대로 따름(현재 90일/3개월)


def _parse_timestamp(entry: dict) -> datetime:
    return datetime.fromisoformat(entry["timestamp"])


def resolve_averaged_model_input(
    entries: list[dict],
    as_of: Optional[datetime] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Optional[tuple[EnvironmentWindow, dict]]:
    """여러 시점 기록 -> (EnvironmentWindow, 참고용 메타 정보) 튜플.

    entries: [{"timestamp": ISO8601 문자열, "cultivation_context": "indoor"|"outdoor",
               "light": 선택지id, "temperature": 선택지id, "humidity": 선택지id}, ...]
    as_of: 기준 시점(생략하면 지금). 과거 시점 기준으로 재계산해볼 때 쓸 수 있습니다.
    window_days: 이 기간(일) 안의 기록만 포함합니다.

    반환값: (EnvironmentWindow, meta). EnvironmentWindow는 build_recommendation_cards()/
    predict_all()/score_all_plants()에 그대로 넘길 수 있습니다. 기록이 아예 없으면 None.
    """
    if not entries:
        return None

    as_of = as_of or datetime.now().astimezone()
    input_options = load_input_options()

    sorted_entries = sorted(entries, key=_parse_timestamp, reverse=True)
    window_start = as_of - timedelta(days=window_days)
    in_window = [e for e in sorted_entries if _parse_timestamp(e) >= window_start]

    used_fallback = False
    if not in_window:
        # 기간 안에 기록이 없으면(예: 오랜만에 다시 접속) 평균 대신 가장 최근 기록 1개로 대체.
        in_window = sorted_entries[:1]
        used_fallback = True

    resolved_list = [
        resolve_model_input(
            e["cultivation_context"], e["light"], e["temperature"], e["humidity"], input_options
        )
        for e in in_window
    ]

    env = summarize_environment(
        light=[r["user_light"] for r in resolved_list],
        temp=[r["user_temp"] for r in resolved_list],
        humidity=[r["user_humidity"] for r in resolved_list],
        # 재배구분은 평균 낼 수 없는 값이라, 기간 내(또는 대체된 기록 중) 가장 최근 값을 사용.
        cultivation_context=resolved_list[0]["user_cultivation_context"],
    )
    meta = {
        "entry_count_used": len(in_window),
        "window_days": window_days,
        "used_fallback_single_entry": used_fallback,
    }
    return env, meta
