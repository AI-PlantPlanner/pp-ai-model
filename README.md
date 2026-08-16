# pp-ai-model
AI 식물플래너 진단/추천 모델

## 개요
- 식물 100종 데이터 + 사용자 환경값(광량/온도/습도/실내·실외) 비교 → 식물별 적합도 점수(0~100점) 산출
- 실사용자 데이터 없는 단계라 룰 기반 채점 로직을 "정답"으로 삼아 ML 모델이 근사하도록 학습(teacher-student 방식)

## 필요 스택
- Python 3.11, 가상환경 `.venv` 사용
- pandas, numpy: 데이터 처리
- scikit-learn: 모델 학습(GradientBoostingRegressor)
- joblib: 모델 저장/로드
- skl2onnx, onnx, onnxruntime: ONNX 변환 및 검증

## 환경 설정
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 파이프라인
- 1단계 `data_normalization.py`: 원본 엑셀 텍스트 범위값 → min/max 숫자 컬럼 정규화
- 2단계 `teacher_scoring.py`: 룰 기반 채점, 사용자 환경 vs 식물 적정범위 비교 → 0~100점
- 3단계 `synthetic_data.py`: 환경값 대량 샘플링 + 2단계 룰로 자동 라벨링 → 학습 데이터 생성
- 4단계 `train_model.py`: "환경+식물 → 점수" 예측 회귀모델 학습 (v0/v1)
- 5단계 `predict.py`: 서빙용 예측 함수, ML 예측에 룰 기반 안전장치 덮어씌움
- 6단계 `export_onnx.py`: v1 모델 ONNX 변환
- 실행 순서: 가상환경 활성화 후 `python -m src.<파일명>` 순서대로 실행

## 사용법
- `predict.py`의 `predict_all()` 함수 하나만 호출하면 됨
- 입력: 광량(lux), 온도(℃), 습도(%), 실내/실외 여부
- 출력: 식물명 / 학명 / 점수, 내림차순 정렬된 표

```python
from src.predict import predict_all
import pandas as pd
from src import config

plants_df = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
result = predict_all(
    plants_df,
    user_light=15000,
    user_temp=25,
    user_humidity=55,
    user_cultivation_context=config.CULTIVATION_INDOOR,  # 또는 CULTIVATION_OUTDOOR
)
```

- 점수 계산은 ML 모델(`v1.joblib`) 기반, 생존/재배 불가능 조합은 항상 0점 강제
- 추천 이유 설명 문장이 필요하면 → `teacher_scoring.py`의 `score_plant()` 사용 권장. 계산 방식은 동일하지만 감점 사유(광량 부족, 온도 초과 등)가 룰 자체에 있어 해석이 쉬움
- 파이썬이 아닌 환경(게임 엔진 등)에서는 `v1.onnx` 사용 (변환만 해둔 상태)

## 현재 상태
- 데이터 구조 설계, 룰 기반 채점, 추천 점수 계산, ML 모델 학습·검증 완료
- 테스트셋 기준 예측 오차 100점 만점 중 1점 미만, 교차검증으로 안정성 확인

## 참고사항
- 학습 데이터 샘플링 범위(`config.py`의 `SYNTHETIC_*`)는 조사 자료 기반 잠정치 → 진단 모델(사진/센서 → 환경값 추정) 스펙 확정 시 재조정 필요
- 식물 종 100 → 300종 이상 증가 시 모델 성능 재검증 필요
- 희귀 조합(내한성 강한 혼합종 × 실내 환경 × 영하권)은 예측 오차 잔존
