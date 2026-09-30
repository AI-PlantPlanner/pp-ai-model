"""ui_config/card_builder.py가 실제로 잘 동작하는지 확인하는 데모 스크립트.

실행: 리포 루트에서 `python -m ui_config.demo_card_check`
"""
import json

import pandas as pd

from src import config
from ui_config.card_builder import build_recommendation_cards
from ui_config.input_resolver import resolve_environment_window


def main() -> None:
    env = resolve_environment_window("indoor", "medium", "normal", "normal")
    print("사용자 선택 -> 모델 입력값(EnvironmentWindow):", env)

    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    cards = build_recommendation_cards(plants_df, env)

    print(f"\n총 카드 수: {len(cards)}")
    print("\n상위 3개 카드 (JSON):")
    for card in cards[:3]:
        print(json.dumps(card, ensure_ascii=False, indent=2))

    gated = [c for c in cards if c["availability"] != "available"]
    print(f"\navailability != available 인 카드 수: {len(gated)}")
    if gated:
        print("예시 1개:")
        print(json.dumps(gated[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
