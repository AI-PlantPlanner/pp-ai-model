"""teacher_scoring.score_all_plants()의 결과가 상식적으로 말이 되는지 눈으로 확인하는 스크립트.

pytest 없이 python -m tests.teacher_scoring_manual_check 로 바로 실행한다.
각 시나리오의 Top-5가 방 환경과 맞는 식물인지(예: 저온 시나리오에서 내한성 약한
관엽식물이 낮은 점수/제외되는지) 직접 눈으로 검토하는 용도.
"""

import pandas as pd

from src import config
from src.teacher_scoring import score_all_plants

SCENARIOS = [
    {
        "name": "저광량 + 고온다습 (욕실/구석 자리)",
        "light": 3000, "temp": 24, "humidity": 80,
        "cultivation_context": config.CULTIVATION_INDOOR,
    },
    {
        "name": "고광량 + 저습 (남향 창가, 건조)",
        "light": 20000, "temp": 22, "humidity": 30,
        "cultivation_context": config.CULTIVATION_INDOOR,
    },
    {
        "name": "저온 (난방 없는 베란다, 겨울)",
        "light": 10000, "temp": 5, "humidity": 50,
        "cultivation_context": config.CULTIVATION_OUTDOOR,
    },
]


def main() -> None:
    df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    for scenario in SCENARIOS:
        print(f"\n=== {scenario['name']} "
              f"(광량={scenario['light']}lux, 온도={scenario['temp']}℃, 습도={scenario['humidity']}%, "
              f"재배구분={scenario['cultivation_context']}) ===")
        result = score_all_plants(
            df, scenario["light"], scenario["temp"], scenario["humidity"], scenario["cultivation_context"]
        )
        print(result.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
