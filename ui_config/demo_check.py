"""ui_config/input_options.json이 predict.py와 실제로 잘 맞물리는지 확인하는 데모 스크립트.

실행: 리포 루트에서 `python -m ui_config.demo_check`
(src.predict를 import해야 해서 리포 루트를 기준으로 모듈처럼 실행해야 합니다)
"""
import pandas as pd

from src import config
from src.predict import predict_all
from ui_config.input_resolver import resolve_environment_window


def main() -> None:
    # 예시: "실내 / 창가 근처(medium) / 보통 온도(normal) / 보통 습도(normal)"를 고른 사용자
    env = resolve_environment_window("indoor", "medium", "normal", "normal")
    print("사용자 선택 -> 모델 입력값(EnvironmentWindow):", env)

    plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    result = predict_all(plants_df, env)
    print("\n추천 상위 10종:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
