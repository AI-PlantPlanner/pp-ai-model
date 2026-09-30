"""sentence_composer.py가 실제로 잘 동작하는지 확인하는 데모 스크립트.

실행: 리포 루트에서 `python -m ui_config.demo_sentence_check`
"""
import pandas as pd

from src import config
from ui_config.card_builder import build_recommendation_cards
from ui_config.input_resolver import resolve_environment_window


def main() -> None:
    env = resolve_environment_window("indoor", "medium", "normal", "normal")
    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    cards = build_recommendation_cards(plants_df, env)

    available_with_grade = [c for c in cards if c["availability"] == "available" and c["grade"] is not None]
    available_without_grade = [c for c in cards if c["availability"] == "available" and c["grade"] is None]
    unavailable = [c for c in cards if c["availability"] != "available"]

    print(f"재배 가능 + 등급 있음: {len(available_with_grade)}건")
    print(f"재배 가능 + 등급 없음(원예팀 검증 대기): {len(available_without_grade)}건")
    print(f"재배 불가: {len(unavailable)}건\n")

    print("[지금 바로 문장이 나오는 카드] 재배 불가 카드 2개 예시:")
    for c in unavailable[:2]:
        print(f"  - {c['name']}: {c['recommendation_sentence']}")

    print("\n[아직 등급 문장이 안 나오는 카드] 재배 가능하지만 등급 미정인 카드 2개 예시:")
    for c in available_without_grade[:2]:
        print(
            f"  - {c['name']}: score={c['score']}, grade={c['grade']}, "
            f"factor_scores={c['factor_scores']}, recommendation_sentence={c['recommendation_sentence']}"
        )

    with_caution = [c for c in cards if c["availability"] == "available" and c["caution_sentence"]]
    without_caution = [c for c in cards if c["availability"] == "available" and not c["caution_sentence"]]
    print(f"\n재배 가능 + 요인별 주의 문구 있음(어느 요인이 40점 미만): {len(with_caution)}건")
    print(f"재배 가능 + 주의 문구 없음(전 요인 40점 이상): {len(without_caution)}건")

    print("\n[요인별 주의 문구 예시] 2개:")
    for c in with_caution[:2]:
        print(f"  - {c['name']}: factor_scores={c['factor_scores']}, caution_sentence=\"{c['caution_sentence']}\"")


if __name__ == "__main__":
    main()
