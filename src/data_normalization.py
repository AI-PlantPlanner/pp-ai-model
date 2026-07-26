"""1단계: 원본 CSV의 텍스트 범위/카운트 컬럼을 min/max 숫자 컬럼으로 정규화.

파싱에 실패하거나 값이 애매한 케이스는 예외로 감추지 않고
data/processed/normalization_issues.csv 에 그대로 남겨서 사람이 직접 확인하도록 한다.
원본 데이터(data/raw)는 읽기 전용으로만 취급하며 값을 추정해서 채우지 않는다.
"""

import re

import pandas as pd

from src import config

_RANGE_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)(?:-(-?\d+(?:\.\d+)?))?$")
_COUNT_WITH_NOTE_PATTERN = re.compile(
    r"^(?P<num>-?\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s*(?:\((?P<note>[^)]*)\))?$"
)


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=config.RAW_HEADER_SKIPROWS)

# 기본 변환기
def parse_numeric_range(value) -> tuple[float | None, float | None]:
    """"15-25" -> (15,25), "14" -> (14,14), "10,000-20,000" -> (10000,20000).
    파싱할 수 없으면 (None, None)을 반환한다 (예외를 던지지 않음 — 호출부에서 로그로 남김).
    """
    if pd.isna(value):
        return None, None
    text = str(value).strip().replace(",", "")
    match = _RANGE_PATTERN.match(text)
    if not match:
        return None, None
    lo = float(match.group(1))
    hi = float(match.group(2)) if match.group(2) is not None else lo
    return min(lo, hi), max(lo, hi)

# 숫자 전용 변환기
def parse_count_with_note(value) -> tuple[float | None, float | None, str | None, bool]:
    """"1(봄-여름)" -> (1,1,"봄-여름",True). "수시(수형관리)" -> (None,None,"수시(수형관리)",False).
    반환값의 마지막 bool은 숫자 파싱 성공 여부. 실패 시 note에는 원본 텍스트 전체를 담는다.
    """
    text = str(value).strip()
    match = _COUNT_WITH_NOTE_PATTERN.match(text)
    if not match:
        return None, None, text, False
    lo, hi = parse_numeric_range(match.group("num"))
    return lo, hi, match.group("note"), True

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


def _apply_count_with_note_columns(df: pd.DataFrame, issues: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for raw_col, prefix in config.COUNT_WITH_NOTE_COLUMNS.items():
        is_pruning = raw_col == config.COL_PRUNING_FREQ
        mins, maxs, notes, freq_types = [], [], [], []
        for name, value in zip(df[config.COL_NAME], df[raw_col]):
            if pd.isna(value):
                _add_issue(issues, name, raw_col, value, "결측")
                mins.append(None)
                maxs.append(None)
                notes.append(None)
                freq_types.append(None)
                continue

            lo, hi, note, parsed_ok = parse_count_with_note(value)
            if parsed_ok:
                freq_type = "fixed" if is_pruning else None
            elif is_pruning and any(k in note for k in config.AS_NEEDED_KEYWORDS):
                freq_type = "as_needed"
            else:
                _add_issue(issues, name, raw_col, value, "숫자(텍스트) 패턴 파싱 실패 - 수동 확인 필요")
                freq_type = None

            mins.append(lo)
            maxs.append(hi)
            notes.append(note)
            freq_types.append(freq_type)

        result[f"{prefix}_min"] = mins
        result[f"{prefix}_max"] = maxs
        result[f"{prefix}_note"] = notes
        if is_pruning:
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


def normalize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict] = []

    for name, value in zip(df[config.COL_NAME], df[config.COL_APPRECIATION_POINT]):
        if pd.isna(value):
            _add_issue(issues, name, config.COL_APPRECIATION_POINT, value, "결측")

    parts = [
        df,
        _apply_range_columns(df, issues),
        _apply_count_with_note_columns(df, issues),
        _apply_cultivation_type(df, issues),
    ]
    result = pd.concat(parts, axis=1)
    result["pest_disease_list"] = df[config.COL_PEST_DISEASE].apply(parse_pest_disease_list)

    issues_df = pd.DataFrame(issues, columns=["식물명", "컬럼", "원본값", "사유"])
    return result, issues_df


def save_outputs(normalized_df: pd.DataFrame, issues_df: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(config.NORMALIZED_OUTPUT_PATH, index=False)
    issues_df.to_csv(config.NORMALIZATION_ISSUES_PATH, index=False)


def main() -> None:
    raw_df = load_raw_data()
    normalized_df, issues_df = normalize_dataframe(raw_df)
    save_outputs(normalized_df, issues_df)
    print(f"정규화 완료: {len(normalized_df)}개 species -> {config.NORMALIZED_OUTPUT_PATH}")
    print(f"확인 필요 항목: {len(issues_df)}건 -> {config.NORMALIZATION_ISSUES_PATH}")


if __name__ == "__main__":
    main()
