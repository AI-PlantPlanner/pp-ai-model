"""추천 점수 (고친 A, 11-12 사용자 결정).

구조: 기존 원예 규칙 채점(src/teacher_scoring.py)을 그대로 쓰고, 모델은 보조 정보만 더한다.
  - 탈락(0점) 기준: 원예팀 확정 두 가지(한계온도 이탈, 재배구분 불일치)만 — teacher_scoring.is_gated 그대로.
    근거: 원예팀 확정(2026-09) "0점 기준은 두 가지뿐", 원예팀 회신(10-28) "GBIF는 보조 자료로만",
    유의성 검정(11-12)에서 모델 추위 추정이 원예값보다 정확하다는 근거 없음.
  - 감점: 원예 적정범위 기준 그대로(광량·온도·습도, 그룹 가중치).
  - 모델 보조 정보(점수에 넣지 않음): 방의 건조한 날(일 평균습도 하위 5%)이 모델 건조 한계(dry_lim)보다 낮으면 경고.
    원예 자료의 건조 기준은 모든 식물 공통(10~20%)이라, 식물별 값은 모델이 유일한 출처.
    점수 반영은 원예팀 동의 후(원예 확정 규칙을 바꾸는 일이라).
입력(놓는 위치마다 365일 배열): t_min, t_max, rh(일 평균), light(일 평균 lux — 4번 DLI 전환 전까지 기존 규칙 단위),
  t_mean(없으면 (t_min+t_max)/2), cultivation_context(실내/실외).
  기간 요약은 teacher_scoring 정의를 따른다: 게이트 = 기간 극값(최저기온의 최소, 최고기온의 최대), 감점 = 기간 평균.
"""

import numpy as np
import pandas as pd

from src import config
from src.teacher_scoring import EnvironmentWindow, score_plant_detail
from src.gee_annual_extremes import ROOT

TOL = ROOT / "results" / "tolerance" / "plant_tolerance.csv"


def load_plants() -> pd.DataFrame:
    p = pd.read_csv(config.NORMALIZED_OUTPUT_PATH)
    tol = pd.read_csv(TOL)[["식물명", "dry_lim", "cold_cal", "hot_lim"]].rename(columns={"식물명": config.COL_NAME})
    p = p.merge(tol, on=config.COL_NAME, how="left")
    # 표시 학명만 확인된 학명으로(점수 계산에는 학명을 쓰지 않음). 원자료는 수정하지 않음.
    from src.name_overrides import DISPLAY_NAMES
    fix = p[config.COL_NAME].map(DISPLAY_NAMES)
    p[config.COL_SCIENTIFIC_NAME] = fix.fillna(p[config.COL_SCIENTIFIC_NAME])
    return p


def summarize(t_min, t_max, rh, light, cultivation_context: str, t_mean=None) -> tuple[EnvironmentWindow, dict]:
    t_min, t_max, rh, light = (np.asarray(a, float) for a in (t_min, t_max, rh, light))
    t_mean = (t_min + t_max) / 2 if t_mean is None else np.asarray(t_mean, float)
    env = EnvironmentWindow(light_mean=float(light.mean()), temp_mean=float(t_mean.mean()),
                            humidity_mean=float(rh.mean()), temp_min=float(t_min.min()), temp_max=float(t_max.max()),
                            cultivation_context=cultivation_context, n_samples=len(t_min))
    extra = {"rh_p05": float(np.percentile(rh, 5))}
    return env, extra


def recommend(plants: pd.DataFrame, env: EnvironmentWindow, extra: dict) -> pd.DataFrame:
    rows = []
    for _, r in plants.iterrows():
        d = score_plant_detail(r, env)
        if d is None:
            continue
        warn = None
        if pd.notna(r["dry_lim"]) and extra["rh_p05"] < r["dry_lim"]:
            warn = (f"건조한 시기 습도 {extra['rh_p05']:.0f}%가 이 식물 자생지의 건조 한계 {r['dry_lim']:.0f}%보다 낮음 "
                    f"(모델 추정, 참고)")
        rows.append({"식물명": r[config.COL_NAME], "점수": d["score"], "광량": d["light"],
                     "온도": d["temp"], "습도": d["humidity"], "탈락_사유": d["gate_reason"],
                     "모델_건조_경고": warn})
    return pd.DataFrame(rows).sort_values("점수", ascending=False).reset_index(drop=True)


SUITABLE_MIN = 40.0  # 09-30 원예 미팅 결정: 40점 이하 부적합


def recommend_json(placement_id: str, daily: dict, cultivation_context: str, plants: pd.DataFrame | None = None) -> dict:
    """입출력 명세(results/tolerance/for_game/io_spec_v1_draft.md)의 응답 형식. daily: 365일 배열 dict."""
    plants = load_plants() if plants is None else plants
    env, extra = summarize(daily["t_min"], daily["t_max"], daily["rh_mean"], daily["light_lux_mean"],
                           cultivation_context, daily.get("t_mean"))
    r = recommend(plants, env, extra)
    sci = plants.set_index(config.COL_NAME)[config.COL_SCIENTIFIC_NAME]
    dry = plants.set_index(config.COL_NAME)["dry_lim"]
    out = []
    for _, x in r.iterrows():
        warns = []
        if x["모델_건조_경고"] and x["탈락_사유"] is None:  # 탈락한 식물에는 경고를 붙이지 않음(사유가 이미 있음)
            warns.append({"type": "model_dry", "basis": "model_estimate", "room_value": round(extra["rh_p05"], 1),
                          "plant_limit": round(float(dry[x["식물명"]]), 1), "message": x["모델_건조_경고"]})
        out.append({"name": x["식물명"], "scientific_name": str(sci[x["식물명"]]).replace("?", " "),
                    "score": round(x["점수"], 1), "suitable": bool(x["탈락_사유"] is None and x["점수"] > SUITABLE_MIN),
                    "factor_scores": {"light": round(x["광량"], 1), "temp": round(x["온도"], 1), "humidity": round(x["습도"], 1)},
                    "gate_reason": x["탈락_사유"], "warnings": warns})
    return {"placement_id": placement_id, "cultivation_context": cultivation_context,
            "room_summary": {"t_min_year": round(env.temp_min, 1), "t_max_year": round(env.temp_max, 1),
                             "t_mean": round(env.temp_mean, 1), "rh_mean": round(env.humidity_mean, 1),
                             "rh_dry_p05": round(extra["rh_p05"], 1), "light_lux_mean": round(env.light_mean, 1)},
                             "plants": out, "spec_version": "v1-draft"}
