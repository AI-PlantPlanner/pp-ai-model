"""광량/온도/습도를 선택지가 아니라 실제 숫자로 직접 입력해서 추천 결과를 확인하는 스크립트.

interactive_check.py는 input_options.json의 선택지(예: "창가 근처")를 거쳐야만 값을 넣을 수
있는데, 이건 그 선택지 단계를 건너뛰고 숫자를 바로 넣어보는 용도입니다. 실제 값(예: 온도 23도,
습도 62%)이 있을 때 모델이 어떻게 반응하는지 바로 확인하거나, input_options.json에 없는
값으로도 테스트해보고 싶을 때 씁니다.

실행: 리포 루트에서 `python -m ui_config.interactive_raw_check`
"""
import pandas as pd

from src import config
from src.teacher_scoring import summarize_environment
from ui_config.card_builder import build_recommendation_cards

# 모델 학습 범위(src/config.py의 SYNTHETIC_*_RANGE). 이 범위를 벗어난 입력은 predict.py 상단
# 주석에 나온 대로 예측 오차가 커질 수 있어서, 막지는 않고 경고만 띄웁니다.
TRAINED_RANGES = {
    "indoor": {
        "light": config.SYNTHETIC_INDOOR_LIGHT_LUX_RANGE,
        "temp": config.SYNTHETIC_INDOOR_TEMP_RANGE,
        "humidity": config.SYNTHETIC_INDOOR_HUMIDITY_RANGE,
    },
    "outdoor": {
        "light": config.SYNTHETIC_OUTDOOR_LIGHT_LUX_RANGE,
        "temp": config.SYNTHETIC_OUTDOOR_TEMP_RANGE,
        "humidity": config.SYNTHETIC_OUTDOOR_HUMIDITY_RANGE,
    },
}


def _input_number(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("숫자로 입력해주세요 (예: 21, -5, 62.5).")


def _input_context() -> str:
    while True:
        raw = input("재배 환경 (1: 실내, 2: 실외): ").strip()
        if raw == "1":
            return "indoor"
        if raw == "2":
            return "outdoor"
        print("1 또는 2를 입력해주세요.")


def _warn_if_out_of_range(ctx_key: str, light: float, temp: float, humidity: float) -> None:
    ranges = TRAINED_RANGES[ctx_key]
    checks = [
        ("광량(lux)", light, ranges["light"]),
        ("온도(℃)", temp, ranges["temp"]),
        ("습도(%)", humidity, ranges["humidity"]),
    ]
    for label, value, (lo, hi) in checks:
        if not (lo <= value <= hi):
            print(f"  ⚠ {label} 값 {value}가 모델 학습 범위({lo}~{hi})를 벗어났어요 — 예측 오차가 커질 수 있어요.")


def main() -> None:
    print("광량/온도/습도를 숫자로 직접 입력해서 테스트합니다.\n")
    ctx_key = _input_context()
    ctx_value = config.CULTIVATION_INDOOR if ctx_key == "indoor" else config.CULTIVATION_OUTDOOR

    light = _input_number("광량(lux, 예: 8000): ")
    temp = _input_number("온도(°C, 예: 21, 영하는 -5처럼): ")
    humidity = _input_number("습도(%, 예: 55): ")

    _warn_if_out_of_range(ctx_key, light, temp, humidity)

    env = summarize_environment(light=light, temp=temp, humidity=humidity, cultivation_context=ctx_value)

    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    cards = build_recommendation_cards(plants_df, env)

    print(f"\n입력값: 광량={light}lux, 온도={temp}°C, 습도={humidity}%, 재배환경={ctx_value}")
    print(f"추천 상위 10종 (전체 {len(cards)}종 중):")
    for i, c in enumerate(cards[:10], 1):
        note = "" if c["availability"] == "available" else f"  [{c['availability']}]"
        print(f"  {i:2d}. {c['name']:14s} ({c['scientific_name']:30s}) 점수={c['score']:5.1f}{note}")


if __name__ == "__main__":
    main()
