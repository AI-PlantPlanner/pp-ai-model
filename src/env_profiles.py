"""3단계 보조: LLM으로 현실적인 환경 프로파일을 생성하고 기계적으로 검증한다.

## 왜 필요한가
4단계 합성 데이터는 지금까지 config.SYNTHETIC_* 범위 안에서 균등 샘플링으로 환경을 만들었다.
그 범위는 조사 자료 기반 잠정치인데 근거 문헌이 없고, 무엇보다 분포가 실제와 다르다.
실제로 광량을 선형 균등으로 뽑았을 때 실사용에서 가장 흔한 100~500lux(창에서 떨어진 거실
안쪽)가 전체 샘플의 1%밖에 안 잡혀서, 그 구간의 모델 근사 오차가 MAE 2.42까지 올라갔다.
로그 균등으로 바꾸자 0.81로 떨어졌다 — 입력 분포를 현실화하면 모델이 좋아진다는 뜻이다.

균등/로그 균등은 여전히 "아무 근거 없는 분포"다. 실제 주거환경은 남향/북향, 창과의 거리,
층수, 계절이 조합된 이산적인 상황들의 집합이고, 각 상황마다 광량·온도·습도가 함께 움직인다
(예: 남향 창가는 광량이 높으면서 여름 온도도 높다). 균등 샘플링은 이 상관을 못 만든다.

그래서 LLM으로 "현실적인 배치 상황" 목록을 만들고, 각 상황의 환경 통계량을 받아
그 프로파일에서 창을 샘플링한다.

## LLM 출력을 그대로 믿지 않는다
LLM 생성 데이터는 모델 고유의 편향 때문에 실제 분포에서 체계적으로 벗어나는 것이
보고되어 있다(distribution drift). 그래서 생성 -> 검증 -> 사용의 3단계로 나누고,
검증은 LLM 없이 기계적으로 한다(validate_profiles). 검증에 걸린 프로파일은 버린다.

## 재현성
생성은 1회성이다. 결과를 config.ENV_PROFILES_PATH(JSON)로 저장해 커밋하고,
4단계는 그 파일만 읽는다. 그래서 API 키 없이도 파이프라인 전체가 재현된다.
파일이 없으면 4단계는 기존 균등 샘플링으로 자동 폴백한다(비교 베이스라인 역할도 한다).

## 실행
    export ANTHROPIC_API_KEY=...        # 또는 `ant auth login`
    python -m src.env_profiles generate  # LLM 호출 -> JSON 저장 (비용 발생)
    python -m src.env_profiles validate  # 저장된 JSON 검증만 (API 불필요)
"""

import json
import sys

from src import config

# LLM에 요구할 프로파일 개수. 실내/실외 x 계절 x 배치를 덮으려면 이 정도는 필요하고,
# 너무 늘리면 프로파일별 샘플 수가 줄어 오히려 각 상황이 얇아진다.
N_PROFILES = 48

MODEL = "claude-opus-5"

# LLM 출력 스키마. additionalProperties=False로 두어 예상 밖 필드가 섞이지 않게 한다.
_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "cultivation_context": {"type": "string", "enum": [config.CULTIVATION_INDOOR, config.CULTIVATION_OUTDOOR]},
                    "season": {"type": "string", "enum": ["봄", "여름", "가을", "겨울"]},
                    "placement": {"type": "string"},
                    "light_lux_mean": {"type": "number"},
                    "light_log_spread": {"type": "number"},
                    "temp_mean": {"type": "number"},
                    "temp_spread": {"type": "number"},
                    "humidity_mean": {"type": "number"},
                    "humidity_spread": {"type": "number"},
                    "weight": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "name", "cultivation_context", "season", "placement",
                    "light_lux_mean", "light_log_spread",
                    "temp_mean", "temp_spread",
                    "humidity_mean", "humidity_spread",
                    "weight", "rationale",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["profiles"],
    "additionalProperties": False,
}

_PROMPT = f"""한국의 실내 화분 식물 추천 서비스를 위한 환경 프로파일 데이터를 만들어 주세요.

## 배경
사용자가 식물을 둘 장소의 광량/온도/습도를 {config.ENV_WINDOW_DAYS}일(약 3개월) 동안 측정한
값을 입력으로 받아, 그 환경에 맞는 식물을 추천하는 모델을 학습시키려 합니다.
학습용 합성 데이터를 만들 때 쓸 "현실적인 배치 상황" 목록이 필요합니다.

## 요청
서로 구별되는 배치 상황 {N_PROFILES}개를 만들어 주세요. 각 상황은 아래를 조합한 하나의
구체적인 장소·시기입니다.
- 실내: 창의 방향(남/동/서/북), 창과의 거리(창가 밀착 / 1~2m / 3m 이상 / 창 없는 구석),
  베란다·발코니 여부
- 실외: 마당, 노지, 옥상, 처마 아래, 큰 나무 그늘 등
- 계절: 봄/여름/가을/겨울 (한국 기후)

실내와 실외를 모두 포함하고, 네 계절이 모두 나타나야 합니다.

## 각 상황에 채울 값
- light_lux_mean: 그 기간의 평균 조도(lux). 낮 시간대 기준의 체감 조도로 잡아 주세요.
- light_log_spread: 일별 조도의 변동 정도를 자연로그 공간의 표준편차로. 날씨 변동이 큰
  실외는 크고, 조명이 일정한 실내 깊은 곳은 작습니다. 대략 0.1~1.0 범위입니다.
- temp_mean, temp_spread: 그 기간 평균기온(°C)과 일별 기온의 표준편차(°C).
- humidity_mean, humidity_spread: 그 기간 평균 상대습도(%)와 일별 표준편차(%p).
- weight: 실제 사용자 환경에서 이 상황이 나타날 상대적 빈도(양수). 전체 합은 자유입니다.
  아파트 거실·베란다처럼 흔한 상황은 크게, 옥상·노지처럼 드문 상황은 작게 주세요.
- rationale: 그 수치를 그렇게 잡은 근거를 한 문장으로. 검증에 쓰므로 구체적으로 써 주세요.

## 반드시 지킬 제약
- 광량 대소 관계가 물리적으로 맞아야 합니다: 실외 맑은 날 > 남향 창가 > 동/서향 창가 >
  북향 창가 > 창에서 먼 실내 > 창 없는 구석.
- 계절 기온 관계가 맞아야 합니다(실외): 겨울 < 봄·가을 < 여름.
- 실내는 냉난방 때문에 실외보다 기온 평균이 온화하고 변동(temp_spread)이 작습니다.
- 값은 한국 기후와 일반적인 주거 환경에 실제로 존재할 수 있는 범위여야 합니다.
  극단적으로 크거나 작은 값을 넣지 마세요."""


def generate_profiles(n_profiles: int = N_PROFILES, model: str = MODEL) -> list[dict]:
    """LLM으로 환경 프로파일을 생성한다. anthropic SDK와 인증이 필요하다.

    출력이 길어 스트리밍으로 받는다(비스트리밍은 HTTP 타임아웃에 걸릴 수 있다).
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - 생성 단계에서만 필요한 의존성
        raise SystemExit(
            "anthropic SDK가 필요합니다: pip install anthropic\n"
            "(생성 단계에서만 필요하고, 저장된 프로파일을 쓰는 4단계에는 필요 없습니다.)"
        ) from exc

    client = anthropic.Anthropic()
    prompt = _PROMPT if n_profiles == N_PROFILES else _PROMPT.replace(str(N_PROFILES), str(n_profiles))

    with client.messages.stream(
        model=model,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": _PROFILE_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise SystemExit(f"모델이 요청을 거절했습니다: {response.stop_details}")

    text = next(block.text for block in response.content if block.type == "text")
    profiles = json.loads(text)["profiles"]
    print(f"생성 완료: {len(profiles)}개 프로파일 (model={model}, "
          f"입력 {response.usage.input_tokens} / 출력 {response.usage.output_tokens} 토큰)")
    return profiles


# ---- 검증 (LLM 없이 기계적으로) ----

# 물리적으로 가능한 범위. config의 샘플링 범위보다 넉넉하게 잡되, 명백한 이상치는 걸러낸다.
_BOUNDS = {
    "light_lux_mean": (30.0, 120_000.0),
    "light_log_spread": (0.02, 1.5),
    "temp_mean": (-20.0, 40.0),
    "temp_spread": (0.1, 15.0),
    "humidity_mean": (10.0, 95.0),
    "humidity_spread": (0.5, 25.0),
    "weight": (0.0, 1000.0),
}

# 실내 배치의 광량 서열. 이름/배치 문자열에 아래 키워드가 있으면 그 등급으로 보고,
# 등급이 높은 배치가 낮은 배치보다 어두우면 모순으로 잡는다.
_LIGHT_ORDER_KEYWORDS = [
    (("창 없", "창문 없", "무창", "구석", "복도", "욕실"), 0),
    (("북향",), 1),
    (("동향", "서향"), 2),
    (("남향",), 3),
]


def _light_rank(profile: dict) -> int | None:
    text = f"{profile.get('name', '')} {profile.get('placement', '')}"
    for keywords, rank in _LIGHT_ORDER_KEYWORDS:
        if any(k in text for k in keywords):
            return rank
    return None


def validate_profiles(profiles: list[dict]) -> tuple[list[dict], list[str]]:
    """프로파일을 기계적으로 검증해 (통과한 프로파일, 문제 목록)을 반환한다.

    LLM 출력이라 값이 그럴듯해 보여도 실제 분포와 어긋날 수 있으므로, 사람 눈에 의존하지 않고
    아래를 코드로 검사한다. 걸린 프로파일은 버리고(사용하지 않고) 문제를 남긴다.
      1. 스키마/타입/필수 필드
      2. 물리적 범위 (_BOUNDS)
      3. 실내 광량 서열 (남향 > 동서향 > 북향 > 무창)
      4. 실외 계절 기온 서열 (겨울 < 봄·가을 < 여름)
      5. 실내는 실외보다 기온 변동이 작아야 함
      6. 커버리지 (실내/실외, 4계절이 모두 있어야 함)
    """
    issues: list[str] = []
    valid: list[dict] = []

    required = _PROFILE_SCHEMA["properties"]["profiles"]["items"]["required"]
    for index, profile in enumerate(profiles):
        label = profile.get("name", f"#{index}")
        missing = [key for key in required if key not in profile]
        if missing:
            issues.append(f"[{label}] 필드 누락: {missing}")
            continue

        out_of_range = []
        for key, (low, high) in _BOUNDS.items():
            value = profile[key]
            if not isinstance(value, (int, float)) or not (low <= value <= high):
                out_of_range.append(f"{key}={value} (허용 {low}~{high})")
        if out_of_range:
            issues.append(f"[{label}] 범위 이탈: {', '.join(out_of_range)}")
            continue

        valid.append(profile)

    indoor = [p for p in valid if p["cultivation_context"] == config.CULTIVATION_INDOOR]
    outdoor = [p for p in valid if p["cultivation_context"] == config.CULTIVATION_OUTDOOR]

    # 3. 실내 광량 서열
    ranked = [(r, p) for p in indoor if (r := _light_rank(p)) is not None]
    for low_rank, low in ranked:
        for high_rank, high in ranked:
            if low_rank < high_rank and low["light_lux_mean"] > high["light_lux_mean"] * 1.2:
                issues.append(
                    f"광량 서열 모순: '{low['name']}'({low['light_lux_mean']:.0f}lux)가 "
                    f"'{high['name']}'({high['light_lux_mean']:.0f}lux)보다 밝음"
                )

    # 4. 실외 계절 기온 서열
    if outdoor:
        by_season = {}
        for profile in outdoor:
            by_season.setdefault(profile["season"], []).append(profile["temp_mean"])
        means = {s: sum(v) / len(v) for s, v in by_season.items()}
        if "겨울" in means and "여름" in means and not means["겨울"] < means["여름"]:
            issues.append(f"계절 기온 모순: 실외 겨울 {means['겨울']:.1f}℃ >= 여름 {means['여름']:.1f}℃")
        for mid in ("봄", "가을"):
            if mid in means and "겨울" in means and means[mid] <= means["겨울"]:
                issues.append(f"계절 기온 모순: 실외 {mid} {means[mid]:.1f}℃ <= 겨울 {means['겨울']:.1f}℃")
            if mid in means and "여름" in means and means[mid] >= means["여름"]:
                issues.append(f"계절 기온 모순: 실외 {mid} {means[mid]:.1f}℃ >= 여름 {means['여름']:.1f}℃")

    # 5. 실내가 실외보다 기온 변동이 작아야 함
    if indoor and outdoor:
        indoor_spread = sum(p["temp_spread"] for p in indoor) / len(indoor)
        outdoor_spread = sum(p["temp_spread"] for p in outdoor) / len(outdoor)
        if indoor_spread >= outdoor_spread:
            issues.append(
                f"기온 변동 모순: 실내 평균 변동 {indoor_spread:.1f}℃ >= 실외 {outdoor_spread:.1f}℃"
            )

    # 6. 커버리지
    if not indoor:
        issues.append("커버리지 부족: 실내 프로파일이 없음")
    if not outdoor:
        issues.append("커버리지 부족: 실외 프로파일이 없음")
    seasons = {p["season"] for p in valid}
    missing_seasons = {"봄", "여름", "가을", "겨울"} - seasons
    if missing_seasons:
        issues.append(f"커버리지 부족: 누락된 계절 {sorted(missing_seasons)}")

    return valid, issues


def load_profiles(path=None) -> list[dict] | None:
    """저장된 프로파일을 읽는다. 파일이 없으면 None(4단계는 균등 샘플링으로 폴백)."""
    path = path or config.ENV_PROFILES_PATH
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["profiles"]


def save_profiles(profiles: list[dict], path=None) -> None:
    path = path or config.ENV_PROFILES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "profiles": profiles}, f, ensure_ascii=False, indent=2)


def _report(profiles: list[dict]) -> None:
    valid, issues = validate_profiles(profiles)
    print(f"검증: {len(valid)}/{len(profiles)}개 통과")
    if issues:
        print(f"문제 {len(issues)}건:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("문제 없음")

    indoor = [p for p in valid if p["cultivation_context"] == config.CULTIVATION_INDOOR]
    outdoor = [p for p in valid if p["cultivation_context"] == config.CULTIVATION_OUTDOOR]
    print(f"\n구성: 실내 {len(indoor)} / 실외 {len(outdoor)}")
    if valid:
        lux = sorted(p["light_lux_mean"] for p in valid)
        print(f"광량 평균 분포: 최소 {lux[0]:,.0f} / 중앙 {lux[len(lux) // 2]:,.0f} / 최대 {lux[-1]:,.0f} lux")
        # 기존 균등 샘플링이 놓쳤던 구간이 실제로 덮이는지 확인한다.
        dark = sum(1 for v in lux if v < 500)
        print(f"  500lux 미만(창에서 먼 실내) 프로파일: {dark}개 — 기존 균등 샘플링에서 가장 얇았던 구간")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "validate"

    if command == "generate":
        profiles = generate_profiles()
        valid, issues = validate_profiles(profiles)
        if issues:
            print(f"경고: 검증 문제 {len(issues)}건 — 통과한 {len(valid)}개만 저장합니다.")
        save_profiles(valid)
        print(f"저장 완료 -> {config.ENV_PROFILES_PATH}")
        _report(valid)
    elif command == "validate":
        profiles = load_profiles()
        if profiles is None:
            raise SystemExit(
                f"프로파일 파일이 없습니다: {config.ENV_PROFILES_PATH}\n"
                "먼저 `python -m src.env_profiles generate`를 실행하세요 (ANTHROPIC_API_KEY 필요)."
            )
        _report(profiles)
    else:
        raise SystemExit("사용법: python -m src.env_profiles [generate|validate]")


if __name__ == "__main__":
    main()
