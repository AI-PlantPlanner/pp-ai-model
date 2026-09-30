"""environment_log.py가 실제로 잘 동작하는지 확인하는 데모 스크립트.

실행: 리포 루트에서 `python -m ui_config.demo_environment_log_check`
"""
from datetime import datetime, timedelta

import pandas as pd

from src import config
from ui_config.card_builder import build_recommendation_cards
from ui_config.environment_log import resolve_averaged_model_input

# 가상 시나리오: 지난 3개월 동안 사용자가 한 달에 한 번씩 환경을 다시 골랐다고 가정.
# (겨울 지나 봄 되면서 광량/온도가 점점 올라가는 느낌으로 값을 다르게 넣어봄)
now = datetime.now().astimezone()
entries = [
    {
        "timestamp": (now - timedelta(days=75)).isoformat(),
        "cultivation_context": "indoor",
        "light": "low",       # 1500 lux
        "temperature": "cold",  # 15도
        "humidity": "dry",     # 40%
    },
    {
        "timestamp": (now - timedelta(days=45)).isoformat(),
        "cultivation_context": "indoor",
        "light": "medium",    # 8000 lux
        "temperature": "normal",  # 21도
        "humidity": "normal", # 55%
    },
    {
        "timestamp": (now - timedelta(days=10)).isoformat(),
        "cultivation_context": "indoor",
        "light": "high",      # 20000 lux
        "temperature": "warm",  # 26도
        "humidity": "normal", # 55%
    },
]


def main() -> None:
    result = resolve_averaged_model_input(entries, as_of=now, window_days=90)
    env, meta = result
    print("3개월치 기록 3건 -> 요약된 EnvironmentWindow:", env)
    print("메타 정보:", meta)

    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    cards = build_recommendation_cards(plants_df, env)

    print("\n[비교] 누적 요약값 기준 상위 5종:")
    for c in cards[:5]:
        print(f"  {c['name']:12s} score={c['score']}")

    # 비교: 마지막(가장 최근) 기록 1개만 썼을 때와 결과가 어떻게 달라지는지
    latest_env, _ = resolve_averaged_model_input(entries[-1:], as_of=now, window_days=90)
    cards_latest = build_recommendation_cards(plants_df, latest_env)
    print("\n[비교] 가장 최근 기록 1개만 썼을 때 상위 5종:")
    for c in cards_latest[:5]:
        print(f"  {c['name']:12s} score={c['score']}")

    # 기록이 아예 없을 때는 None을 반환하는지 확인
    assert resolve_averaged_model_input([]) is None
    print("\n기록 없음 -> None 반환 확인 완료")


if __name__ == "__main__":
    main()
