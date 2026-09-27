"""1단계: 원본 엑셀의 텍스트 범위/카운트 컬럼을 min/max 숫자 컬럼으로 정규화.

파싱에 실패하거나 값이 애매한 케이스는 예외로 감추지 않고
data/processed/normalization_issues.csv 에 그대로 남겨서 사람이 직접 확인하도록 한다.
원본 데이터(data/raw)는 읽기 전용으로만 취급하며 값을 추정해서 채우지 않는다.
"""

import re

import pandas as pd

from src import config

_RANGE_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)(?:-(-?\d+(?:\.\d+)?))?$")
# "≥14"처럼 하한만 있고 상한이 없는 표기 — 해당 값 이상이면 전부 적정이라는 뜻으로,
# 임의로 상한을 추정해 채우지 않고 (min, None)으로 남긴다.
_OPEN_LOWER_PATTERN = re.compile(r"^[≥>]\s*(-?\d+(?:\.\d+)?)$")

# 학명 등 텍스트 컬럼에 섞여 들어오는 특수 공백(non-breaking space) — 일반 공백으로 정리.
_NBSP = "\xa0"


def load_raw_data(path=config.RAW_DATA_PATH, sheet_name=config.RAW_SHEET_NAME) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=config.RAW_HEADER_SKIPROWS)
    df[config.COL_SCIENTIFIC_NAME] = df[config.COL_SCIENTIFIC_NAME].str.replace(
        _NBSP, " ", regex=False
    )
    return df


# 검토본에서 덮어쓸 광량 컬럼. 검토본 헤더가 3차 원본 컬럼명과 같다.
_LIGHT_REVIEW_COLUMNS = (
    config.COL_LIGHT_SATURATION,
    config.COL_LIGHT_COMPENSATION,
    config.COL_LIGHT_LUX,
)


def apply_light_review(raw_df: pd.DataFrame, path=config.LIGHT_REVIEW_PATH) -> pd.DataFrame:
    """원예팀 광량 검토본의 값으로 해당 종의 광량 3개 컬럼(원본 텍스트)을 덮어쓴다.

    원본 파일은 고치지 않고 메모리에서만 덮어쓴다. 파싱 전에 텍스트를 바꾸므로
    이후 min/max 정규화와 충돌 검사가 모두 검토본 값을 기준으로 돈다.
    검토본의 식물명이 원본에 없거나 중복이면 조용히 건너뛰지 않고 에러로 멈춘다 —
    한 종이라도 빠지면 그 종만 옛 값으로 채점되는데 겉으로는 드러나지 않기 때문이다.
    """
    review = pd.read_excel(path)
    names = review[config.LIGHT_REVIEW_NAME_COLUMN].astype(str).str.strip()
    raw_names = raw_df[config.COL_NAME].astype(str).str.strip()

    missing = sorted(set(names) - set(raw_names))
    duplicated = sorted(names[names.duplicated()].unique())
    if missing or duplicated:
        raise ValueError(
            f"광량 검토본 식물명 매칭 실패 — 원본에 없음: {missing}, 검토본 내 중복: {duplicated}"
        )

    result = raw_df.copy()
    changed = 0
    for name, (_, review_row) in zip(names, review.iterrows()):
        mask = raw_names == name
        for col in _LIGHT_REVIEW_COLUMNS:
            new_value = review_row[col]
            old_value = result.loc[mask, col].iloc[0]
            if parse_numeric_range(new_value) != parse_numeric_range(old_value):
                changed += 1
            result.loc[mask, col] = str(new_value)
    print(f"광량 검토본 반영: {len(review)}종, 값이 바뀐 칸 {changed}개 ({path.name})")
    return result


# 기본 변환기
def parse_numeric_range(value) -> tuple[float | None, float | None]:
    """"15-25" -> (15,25), "14" -> (14,14), "10,000-20,000" -> (10000,20000).
    "≥14" -> (14, None) — 하한만 있고 상한이 없는 개방형 범위(해당 값 이상은 전부 적정).
    파싱할 수 없으면 (None, None)을 반환한다 (예외를 던지지 않음 — 호출부에서 로그로 남김).
    """
    if pd.isna(value):
        return None, None
    text = str(value).strip().replace(",", "")
    open_match = _OPEN_LOWER_PATTERN.match(text)
    if open_match:
        return float(open_match.group(1)), None
    match = _RANGE_PATTERN.match(text)
    if not match:
        return None, None
    lo = float(match.group(1))
    hi = float(match.group(2)) if match.group(2) is not None else lo
    return min(lo, hi), max(lo, hi)

# 숫자 범위 또는 "수시"(as-needed) 전용 변환기
def parse_as_needed_range(value) -> tuple[float | None, float | None, str | None]:
    """"1" -> (1,1,"fixed"). "1-2" -> (1,2,"fixed"). "수시" -> (None,None,"as_needed").
    셋 다 아니면 (None,None,None) — 호출부에서 issue로 남긴다.
    """
    text = str(value).strip()
    if any(k in text for k in config.AS_NEEDED_KEYWORDS):
        return None, None, "as_needed"
    lo, hi = parse_numeric_range(text)
    if lo is None:
        return None, None, None
    return lo, hi, "fixed"

# 재배구분 판독기
def classify_cultivation_type(value) -> tuple[str, str | None]:
    """'실내'/'실외' 키워드 포함 여부로 재배구분을 분류하고 원본 텍스트는 note로 보존한다.

    표준 3분류 밖의 서술형 텍스트(예: "실외, 실내 재배 제한적 가능...")도 '실내'와 '실외'가 함께 언급되면 혼합(실내·실외)로 분류한다.
    결측이거나 두 키워드 모두 없으면 '미상'으로 남기고 자동으로 포함/제외하지 않는다.
    """
    if pd.isna(value):
        return config.CULTIVATION_UNKNOWN, None
    text = str(value).strip().lstrip("*").strip()
    has_indoor = config.CULTIVATION_INDOOR in text
    has_outdoor = config.CULTIVATION_OUTDOOR in text
    if has_indoor and has_outdoor:
        category = config.CULTIVATION_MIXED
    elif has_indoor:
        category = config.CULTIVATION_INDOOR
    elif has_outdoor:
        category = config.CULTIVATION_OUTDOOR
    else:
        category = config.CULTIVATION_UNKNOWN
    return category, text


# 생육형 그룹 판독기
def classify_plant_group(value) -> str | None:
    """원예팀이 채워준 생육형 그룹을 표준 그룹명으로 통일한다.

    raw data 시트('관엽', '목본' 등)와 '그룹별 가중치' 시트('관엽식물', '목본류' 등)의 표기가
    달라서 config.PLANT_GROUP_ALIASES로 매핑한다. 매핑에 없는 값은 임의로 추측하지 않고
    None으로 남긴다(채점 시 동일 비율 기본 가중치로 처리).
    """
    if pd.isna(value):
        return None
    return config.PLANT_GROUP_ALIASES.get(str(value).strip())


def parse_pest_disease_list(value) -> str | None:
    """"응애, 진딧물" -> "응애|진딧물". 인코딩(멀티핫 등)은 이후 피처엔지니어링 단계에서 처리."""
    if pd.isna(value):
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return "|".join(parts)


def _add_issue(issues: list[dict], name, column, raw_value, reason: str) -> None:
    issues.append({"식물명": name, "컬럼": column, "원본값": raw_value, "사유": reason})


def _apply_range_columns(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for raw_col, prefix in config.RANGE_COLUMNS.items():
        mins, maxs = [], []
        for name, value in zip(df[config.COL_NAME], df[raw_col]):
            lo, hi = parse_numeric_range(value)
            if lo is None:
                reason = "결측" if pd.isna(value) else "숫자 범위 패턴 파싱 실패"
                _add_issue(issues, name, raw_col, value, reason)
            mins.append(lo)
            maxs.append(hi)
        result[f"{prefix}_min"] = mins
        result[f"{prefix}_max"] = maxs
    return result


def _apply_as_needed_range_columns(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for raw_col, prefix in config.AS_NEEDED_RANGE_COLUMNS.items():
        mins, maxs, freq_types = [], [], []
        for name, value in zip(df[config.COL_NAME], df[raw_col]):
            if pd.isna(value):
                _add_issue(issues, name, raw_col, value, "결측")
                mins.append(None)
                maxs.append(None)
                freq_types.append(None)
                continue

            lo, hi, freq_type = parse_as_needed_range(value)
            if freq_type is None:
                _add_issue(issues, name, raw_col, value, "숫자 범위/수시 패턴 파싱 실패 - 수동 확인 필요")

            mins.append(lo)
            maxs.append(hi)
            freq_types.append(freq_type)

        result[f"{prefix}_min"] = mins
        result[f"{prefix}_max"] = maxs
        result[f"{prefix}_type"] = freq_types
    return result


def _apply_cultivation_type(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    categories, notes = [], []
    for name, value in zip(df[config.COL_NAME], df[config.COL_CULTIVATION_TYPE]):
        category, note = classify_cultivation_type(value)
        if category == config.CULTIVATION_UNKNOWN:
            _add_issue(issues, name, config.COL_CULTIVATION_TYPE, value, "재배구분 미상 - 수동 확인 필요")
        categories.append(category)
        notes.append(note)
    return pd.DataFrame(
        {"cultivation_type": categories, "cultivation_note": notes}, index=df.index
    )


def _apply_plant_group(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    groups = []
    for name, value in zip(df[config.COL_NAME], df[config.COL_PLANT_GROUP]):
        group = classify_plant_group(value)
        if group is None:
            _add_issue(issues, name, config.COL_PLANT_GROUP, value, "생육형 그룹 미상 - 수동 확인 필요")
        groups.append(group)
    return pd.DataFrame({config.NORM_PLANT_GROUP: groups}, index=df.index)


def detect_light_conflicts(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """적정 광량과 광보상점/광포화점이 서로 모순되는 종을 찾는다.

    광보상점(광합성량이 호흡량을 넘어서는 최소 광량)보다 적정 광량 하한이 낮거나
    광포화점보다 적정 광량 상한이 높으면, 그 종은 "적정범위인데 동시에 생육 불가/피해 구간"이 된다.
    우리가 임의로 고치지 않고 원예팀 확인 요청용 목록으로만 남긴다.
    채점 단계(teacher_scoring)는 종별 적정 광량을 우선하도록 처리한다.

    범위로 기재된 값(예: "500-2,000")은 원예팀 확정 기준(2026-09)대로
    광보상점은 하한, 광포화점은 상한을 판정 경계로 쓴다.
    """
    rows = []
    for _, row in normalized_df.iterrows():
        lux_min = row[config.NORM_LIGHT_LUX_MIN]
        lux_max = row[config.NORM_LIGHT_LUX_MAX]
        compensation = row[config.NORM_LIGHT_COMPENSATION_MIN]
        saturation = row[config.NORM_LIGHT_SATURATION_MAX]

        reasons = []
        if pd.notna(lux_max) and pd.notna(saturation) and lux_max > saturation:
            reasons.append("적정 광량 상한 > 광포화점")
        if pd.notna(lux_min) and pd.notna(compensation) and lux_min < compensation:
            reasons.append("적정 광량 하한 < 광보상점")

        if reasons:
            rows.append({
                "식물명": row[config.COL_NAME],
                "생육형그룹": row[config.NORM_PLANT_GROUP],
                "적정 광량(lux)": row[config.COL_LIGHT_LUX],
                "광보상점(lux)": row[config.COL_LIGHT_COMPENSATION],
                "광포화점(lux)": row[config.COL_LIGHT_SATURATION],
                "사유": " / ".join(reasons),
            })
    return pd.DataFrame(
        rows,
        columns=["식물명", "생육형그룹", "적정 광량(lux)", "광보상점(lux)", "광포화점(lux)", "사유"],
    )


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict] = []

    for name, value in zip(df[config.COL_NAME], df[config.COL_APPRECIATION_POINT]):
        if pd.isna(value):
            _add_issue(issues, name, config.COL_APPRECIATION_POINT, value, "결측")

    parts = [
        df,
        _apply_range_columns(df, issues),
        _apply_as_needed_range_columns(df, issues),
        _apply_cultivation_type(df, issues),
        _apply_plant_group(df, issues),
    ]
    result = pd.concat(parts, axis=1)
    result["pest_disease_list"] = df[config.COL_PEST_DISEASE].apply(parse_pest_disease_list)

    issues_df = pd.DataFrame(issues, columns=["식물명", "컬럼", "원본값", "사유"])
    return result, issues_df


def save_outputs(
    normalized_df: pd.DataFrame, issues_df: pd.DataFrame, conflicts_df: pd.DataFrame
) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(config.NORMALIZED_OUTPUT_PATH, index=False)
    issues_df.to_csv(config.NORMALIZATION_ISSUES_PATH, index=False)
    conflicts_df.to_csv(config.LIGHT_CONFLICT_PATH, index=False)


def main() -> None:
    raw_df = apply_light_review(load_raw_data())
    normalized_df, issues_df = normalize_dataframe(raw_df)
    conflicts_df = detect_light_conflicts(normalized_df)
    save_outputs(normalized_df, issues_df, conflicts_df)
    print(f"정규화 완료: {len(normalized_df)}개 species -> {config.NORMALIZED_OUTPUT_PATH}")
    print(f"확인 필요 항목: {len(issues_df)}건 -> {config.NORMALIZATION_ISSUES_PATH}")
    print(f"광량 기준 충돌: {len(conflicts_df)}종 -> {config.LIGHT_CONFLICT_PATH}")


if __name__ == "__main__":
    main()
