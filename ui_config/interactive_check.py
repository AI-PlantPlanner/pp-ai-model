"""터미널에서 직접 선택지를 골라 즉시 추천 결과를 확인하는 대화형 스크립트.

Unity 클라이언트가 실제로 할 일(사용자가 화면에서 버튼 고르기 -> 모델 입력값 변환 ->
추천 카드 계산)을 터미널에서 손으로 흉내내볼 수 있게 만든 것입니다. input_options.json의
선택지 목록을 그대로 읽어서 번호로 고르게 하므로, 나중에 선택지 문구/개수가 바뀌어도 이
스크립트는 수정할 필요가 없습니다.

실행: 리포 루트에서 `python -m ui_config.interactive_check`
"""
import pandas as pd

from src import config
from ui_config.card_builder import build_recommendation_cards
from ui_config.input_resolver import load_input_options, resolve_model_input


def _choose(options: list, prompt: str) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt['label']}")
    while True:
        raw = input("번호를 입력하세요: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]["id"]
        print("잘못된 입력이에요. 목록에 있는 번호를 입력해주세요.")


def main() -> None:
    input_options = load_input_options()

    ctx_id = _choose(input_options["cultivation_context_options"], "1) 재배 환경을 골라주세요:")
    light_id = _choose(input_options["light_options"][ctx_id], "2) 광량을 골라주세요:")
    temp_id = _choose(input_options["temperature_options"][ctx_id], "3) 온도를 골라주세요:")
    humidity_id = _choose(input_options["humidity_options"][ctx_id], "4) 습도를 골라주세요:")

    resolved = resolve_model_input(ctx_id, light_id, temp_id, humidity_id, input_options)
    print("\n입력하신 선택 -> 모델 입력값:", resolved)

    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    cards = build_recommendation_cards(plants_df, **resolved)

    print(f"\n추천 상위 10종 (재배 가능 판정된 것 중 전체 {len(cards)}종 중):")
    for i, c in enumerate(cards[:10], 1):
        note = "" if c["availability"] == "available" else f"  [{c['availability']}]"
        print(f"  {i:2d}. {c['name']:14s} ({c['scientific_name']:30s}) 점수={c['score']:5.1f}{note}")


if __name__ == "__main__":
    main()
