"""1번 작업 보조 모듈: input_options.json의 선택 id들을 실제 모델 입력값(숫자)으로 변환.

원래 demo_check.py 안에만 있던 resolve_model_input()을 여기로 옮겼습니다. 이제 단발성 확인용
데모뿐 아니라 environment_log.py(여러 시점 기록을 누적해서 평균 내는 로직)에서도 같은 변환
함수를 재사용해야 해서, 공용 모듈로 분리했습니다.
"""

import json
from pathlib import Path
from typing import Optional

INPUT_OPTIONS_PATH = Path(__file__).parent / "input_options.json"

_input_options_cache: Optional[dict] = None


def load_input_options(path: Path = INPUT_OPTIONS_PATH) -> dict:
    global _input_options_cache
    if _input_options_cache is None:
        _input_options_cache = json.loads(path.read_text(encoding="utf-8"))
    return _input_options_cache


def resolve_model_input(
    cultivation_id: str,
    light_id: str,
    temp_id: str,
    humidity_id: str,
    input_options: Optional[dict] = None,
) -> dict:
    """Unity가 할 일과 동일: 사용자가 고른 선택지 id들을 모델이 요구하는 숫자 입력값으로 변환."""
    if input_options is None:
        input_options = load_input_options()

    ctx_option = next(o for o in input_options["cultivation_context_options"] if o["id"] == cultivation_id)
    ctx_value = ctx_option["value"]  # "실내" or "실외"
    ctx_key = cultivation_id  # "indoor" or "outdoor" -> light/temp/humidity_options의 키와 동일

    light_option = next(o for o in input_options["light_options"][ctx_key] if o["id"] == light_id)
    temp_option = next(o for o in input_options["temperature_options"][ctx_key] if o["id"] == temp_id)
    humidity_option = next(o for o in input_options["humidity_options"][ctx_key] if o["id"] == humidity_id)

    return {
        "user_light": light_option["value_lux"],
        "user_temp": temp_option["value_celsius"],
        "user_humidity": humidity_option["value_percent"],
        "user_cultivation_context": ctx_value,
    }
