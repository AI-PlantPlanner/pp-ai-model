"""sentence_composer.py가 실제로 잘 동작하는지 확인하는 데모 스크립트.

실행: 리포 루트에서 `python -m ui_config.demo_sentence_check`
"""
import pandas as pd

from src import config
from ui_config.card_builder import build_recommendation_cards
from ui_config.input_resolver import resolve_model_input
from ui_config.sentence_composer import compose_factor_caution_sentence


def main() -> None:
    resolved = resolve_model_input("indoor", "medium", "normal", "normal")
    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    cards = build_recommendation_cards(plants_df, **resolved)

    available_with_grade = [c for c in cards if c["availability"] == "available" and c["grade"] is not None]
    available_without_grade = [c for c in cards if c["availability"] == "available" and c["grade"] is None]
    unavailable = [c for c in cards if c["availability"] != "available"]

    print(f"재배 가능 + 등급 있음: {len(available_with_grade)}건")
    print(f"재배 가능 + 등급 없음(원예팀 검증 대기): {len(available_without_grade)}건")
    print(f"재배 불가: {len(unavailable)}건\n")

    print("[지금 바로 문장이 나오는 카드] 재배 불가 카드 2개 예시:")
    for c in unavailable[:2]:
        print(f"  - {c['name']}: {c['recommendation_sentence']}")

    print("\n[아직 문장이 안 나오는 카드] 재배 가능하지만 등급 미정인 카드 2개 예시:")
    for c in available_without_grade[:2]:
        print(f"  - {c['name']}: score={c['score']}, grade={c['grade']}, recommendation_sentence={c['recommendation_sentence']}")

    print("\n[요소별 주의 문구 미리보기] - 아직 카드에는 연결 안 됨, 문구만 확인:")
    sample_name = cards[0]["name"]
    for factor_key in [
        "light_too_low", "light_too_high",
        "temp_too_low", "temp_too_high",
        "humidity_too_low", "humidity_too_high",
    ]:
        print(f"  - {factor_key}: {compose_factor_caution_sentence(sample_name, factor_key)}")


if __name__ == "__main__":
    main()
