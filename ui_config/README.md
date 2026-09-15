# ui_config

Unity 클라이언트가 `pp-ai-model`의 예측 함수(`src/predict.py`의 `predict_all`, 또는 `models/recommendation_model_v1.onnx`)를 호출할 때 필요한 4개 입력값(광량, 온도, 습도, 재배구분)을 만들기 위한 화면단 선택지 매핑입니다.

## 왜 필요한가
모델 입력은 lux, ℃, % 같은 원시 수치인데 일반 사용자는 자기 집 광량을 lux 단위로 답할 수 없습니다. 그래서 사람이 고를 수 있는 보기(예: "창가 바로 옆")를 만들고, 각 보기에 모델이 이해하는 대표 수치를 매핑해뒀습니다.

## 파일
- `input_options.json`: 실제 매핑 데이터. Unity는 이 파일을 그대로 파싱해서 선택지 UI를 그리고, 사용자가 고른 `id`에 대응하는 `value_*`를 모아 모델 입력으로 넘기면 됩니다. 하단 `model_input_mapping.example`이 조합 예시입니다.

## 수치 근거
- 1차로 `data/processed/plants_normalized.csv`(100종 실측 데이터)의 실내/실외별 광량·온도·습도 25/50/75퍼센타일을 참고해 "보통" 구간을 잡았습니다.
- 2차로 `src/config.py`의 `SYNTHETIC_INDOOR_*_RANGE` / `SYNTHETIC_OUTDOOR_*_RANGE`(모델 학습 시 샘플링한 범위)를 넘지 않도록 상/하한을 잡았습니다 — 학습 범위 밖 입력은 모델 예측 오차가 커질 수 있기 때문입니다(`predict.py` 상단 주석 참고).
- 실외 습도의 "여름/봄가을" 구간은 README 초안에 있던 "여름 평균습도 79.9%, 봄가을 63.6%" 참고자료를 반영했습니다.

## 아직 확정 아닌 것 (다음 단계에서 다룰 것)
- 선택지 라벨 문구와 단계 수(현재 5단계)는 UX 검토 전 초안입니다.
- 점수 등급 구간, 감점 사유 문장 템플릿은 원예팀 검증 결과가 나온 뒤 별도 파일(`grade_config.json` 등, 추후 작업)에서 다룹니다. 이 파일은 "값 → 등급" 변환과는 무관하며, 모델에 넣을 입력값만 다룹니다.
- 향후 진단 모델(사진/센서 → 환경값 자동 추정)이 붙으면 이 선택지 UI 자체가 없어질 수 있습니다(`README.md`의 "참고사항" 항목).

## card_builder.py (2번 작업: 추천 카드 구조)

`build_recommendation_cards(plants_df, user_light, user_temp, user_humidity, user_cultivation_context)`
하나만 호출하면 점수 내림차순으로 정렬된 카드 리스트(JSON 직렬화 가능한 dict의 list)가 나옵니다.
`predict.py`의 `predict_score()`를 종별로 직접 호출해서, 점수 계산과 동시에 `plants_normalized.csv`의
나머지 속성(과/생육형태/감상포인트/관리주기/병해충)을 같은 카드에 합칩니다.

카드 필드:
- `plant_key`: 학명 기반 slug. Unity에서 스프라이트 등 에셋 조회용 키로 쓸 수 있게 만들었습니다.
- `score`: 0~100 원점수 (반올림 1자리).
- `grade`: **아직 null.** 3번 작업(점수 구간 -> 등급) 완료 후 채웁니다. 카드 스키마는 지금 확정해두고
  값만 나중에 채우는 구조입니다.
- `availability`: `"available"` / `"unavailable_context"`(실내·실외 불일치) /
  `"unavailable_temperature"`(한계온도 미달). 이건 원예팀 검증이나 4번 작업과 무관하게 지금 바로
  계산 가능해서 미리 넣었습니다 — 점수가 0점이어도 "그냥 적합도가 낮음"과 "애초에 재배 불가"는
  사용자에게 다른 문구로 보여줘야 하기 때문입니다. `teacher_scoring.is_gated()`와 같은 조건을
  미러링한 것이라 그쪽 로직이 바뀌면 여기도 같이 수정해야 합니다(카드 파일 상단 주석 참고).
- `caution_sentence`: **아직 null.** 4/5번 작업(감점 사유 -> 문장) 협의 후 채웁니다.

### 확인 방법
```bash
PYTHONPATH=. python -m ui_config.demo_card_check
```
100종 전체에 대해 카드를 만들고, 상위 3개 카드의 JSON과 `availability != available`인 카드 예시
1개를 출력합니다.

### 유리님과 상의할 것 (4번 작업 관련, 메모)
`score_plant()`/`predict_score()`가 지금은 광량/온도/습도 감점을 합산한 최종 점수만 반환합니다.
`caution_sentence`를 채우려면 요소별 점수(또는 어느 요소가 얼마나 부족한지)가 따로 필요합니다.
또한 `is_gated()`가 사유(재배구분/한계온도 중 어느 쪽인지)까지 같이 반환해주면 `_availability()`의
미러링 로직을 안 써도 돼서 더 안전할 것 같습니다.

## grade_resolver.py (3번 작업: 점수 구간 -> 등급)

`grade_config.json`에 등급 목록(`optimal/good/caution/poor` + 별도 `unavailable`)을 정의해뒀는데,
**각 등급의 점수 기준(`min_score`)은 전부 `null`로 비워뒀습니다.** 원예팀 검증 결과가 나오기 전이라
숫자를 임의로 지어내지 않았습니다.

- `availability != "available"`인 카드(실내·실외 불일치, 한계온도 미달)는 점수와 무관하게 지금
  바로 `"unavailable"` 등급이 매겨집니다 — 이건 원예팀 검증과 상관없는 정보라서요.
- 그 외 카드는 `min_score`가 비어있는 동안 `grade: null`을 유지합니다(카드 스키마에 `grade` 필드
  자리는 이미 있고, 값만 비어있는 상태).

**원예팀 검증 후 할 일**: `grade_config.json`의 각 grade 항목 `min_score`에 숫자만 채우면 끝입니다.
`grade_resolver.py`나 `card_builder.py` 코드는 건드릴 필요 없습니다. 등급 이름/개수/문구는
지금 짠 것이 초안이라 바뀔 수 있습니다.

확인:
```bash
PYTHONPATH=. python -m ui_config.demo_card_check
```
"unavailable" 등급으로 잡힌 카드 예시가 마지막에 출력됩니다.

## environment_log.py (환경값 누적 평균 방식 — 제안)

한 시점 스냅샷 대신, 사용자가 여러 시점에 남긴 환경 선택 기록을 누적해서 평균 낸 값으로
추천하는 방식을 제안/설계했습니다. 계절이나 날짜에 따라 광량/온도/습도가 계속 바뀌기 때문에,
한 순간의 값보다 일정 기간의 평균이 더 안정적인 추천 기준이 될 수 있다는 아이디어입니다.

`resolve_averaged_model_input(entries, window_days=90)`에 최근 기록들을 리스트로 넘기면,
기간(기본 90일=3개월) 안의 기록들을 각각 숫자로 변환해 평균 낸 다음, 기존 `predict_all()` /
`build_recommendation_cards()`에 그대로 넣을 수 있는 입력값을 돌려줍니다. 재배구분(실내/실외)은
평균 낼 수 없는 값이라 기간 내 가장 최근 기록 기준을 따릅니다.

이 모듈은 "기록 리스트를 받아서 평균 계산"만 하고, 기록을 실제로 어디에 저장할지(Unity 로컬
저장 vs 서버 DB)는 다루지 않습니다 — 그 결정은 호출부(앱 아키텍처)에 달려 있습니다.

### 확인
```bash
PYTHONPATH=. python -m ui_config.demo_environment_log_check
```
가상의 3개월치 기록(광량이 점점 밝아지는 시나리오)으로 평균값을 계산하고, "평균값 기준
추천"과 "가장 최근 기록 1개만 썼을 때 추천"이 실제로 다른 식물 순위를 준다는 것까지
확인합니다.

### 팀과 상의할 것
- 누적 기간: 3개월 vs 6개월 (지금은 90일 기본값, `DEFAULT_WINDOW_DAYS`만 바꾸면 됨)
- 기록을 어떻게 쌓을지: 사용자가 얼마나 자주 재입력하게 할지(예: 2주/한 달 주기 알림), 기록
  저장 위치(로컬/서버)
- 기간 중간에 실내→실외처럼 재배구분이 바뀐 경우 처리 방식(지금은 "가장 최근 것 우선")
- 기록이 부족할 때(가입 초반 등) 어떻게 보여줄지 — 지금은 있는 기록만으로 평균/대체 처리

## interactive_check.py (터미널에서 직접 선택해서 테스트)

파일을 열어서 값을 고쳐가며 테스트하는 게 번거로워서, 번호를 입력하면 바로 결과를 보여주는
대화형 스크립트를 추가했습니다. `input_options.json`의 선택지 목록을 그대로 읽어서 보여주기
때문에 선택지 문구나 개수가 바뀌어도 이 스크립트는 그대로 씁니다.

```bash
PYTHONPATH=. python -m ui_config.interactive_check
```

재배 환경 -> 광량 -> 온도 -> 습도 순서로 번호를 고르면, 실제 모델 입력값과 추천 상위 10종
(재배 불가 사유 포함)이 바로 출력됩니다. Unity에서 사용자가 버튼을 누르는 것과 동일한 흐름을
터미널에서 손으로 미리 체험해볼 수 있는 용도입니다.

## interactive_raw_check.py (숫자로 직접 입력해서 테스트)

`interactive_check.py`는 input_options.json 선택지를 거쳐야 하는데, 정확한 수치나 극단적인
값으로 바로 테스트하고 싶을 때는 이 스크립트를 씁니다. 광량(lux)/온도(°C)/습도(%)를 숫자로
직접 입력받아 선택지 매핑 없이 바로 추천 결과를 계산합니다.

```bash
PYTHONPATH=. python -m ui_config.interactive_raw_check
```

`src/config.py`의 `SYNTHETIC_*_RANGE`(모델 학습 범위)를 벗어난 값을 넣으면 경고를 띄웁니다
(막지는 않고 계속 진행합니다) — `predict.py` 상단 주석에 나온 대로 학습 범위 밖에서는 예측
오차가 커질 수 있기 때문입니다. 실제로 실내 온도를 학습범위(-10~33°C) 밖인 50°C로 넣으면
경고와 함께 점수가 크게 떨어지는 것도 확인했습니다.

## sentence_composer.py / korean_particles.py (등급·재배가능여부 문장 미리 작성)

"언제 이 문장을 쓸지 판단하는 로직"(등급 점수 구간, 요소별 감점 사유)과 "문장 자체의 문구"는
별개라는 점에 착안해, 문구는 원예팀 검증/유리님 데이터를 기다리지 않고 미리 다 작성해뒀습니다.
`sentence_templates.json`에 문구가 있고, `sentence_composer.py`가 여기에 식물 이름을
조사(은/는, 이/가)까지 자연스럽게 맞춰서 끼워 넣습니다(`korean_particles.py`).

- **지금 바로 카드에 연결된 것**: 재배 불가 사유 문장(`unavailable_context`,
  `unavailable_temperature`). `availability`는 이미 계산 가능한 정보라, 카드의
  `recommendation_sentence` 필드에 바로 채워집니다.
- **문구는 준비됐지만 아직 안 나오는 것**: 등급 문장(`optimal`/`good`/`caution`/`poor`). 카드의
  `grade`가 `grade_config.json`의 점수 기준이 채워지기 전까지는 `None`이라, 등급 문장도
  자동으로 `None`입니다. 점수 기준만 채우면 코드 수정 없이 바로 나옵니다.
- **문구는 준비됐지만 카드에는 아직 연결 안 한 것**: 요소별 주의 문구
  (`light_too_low`/`temp_too_high` 등 6가지). `compose_factor_caution_sentence()`로 미리
  만들어뒀지만, "광량/온도/습도 중 어느 게 문제인지" 판단할 데이터가 아직 없어서
  `card_builder.py`에는 연결하지 않았습니다. 유리님이 요소별 점수를 주시면 이 함수를 그대로
  카드에 연결하면 됩니다.

### 확인
```bash
PYTHONPATH=. python -m ui_config.demo_sentence_check
```
재배 불가 카드는 문장이 바로 나오고, 재배 가능하지만 등급 미정인 카드는 문장이 아직 `None`인
것, 그리고 요소별 주의 문구 6가지 미리보기까지 확인할 수 있습니다.
