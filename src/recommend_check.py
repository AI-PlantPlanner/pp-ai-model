"""고친 A 점검: 서울 시나리오 5개(src/indoor_check.scenarios)에서 원예 규칙 판정과 모델 건조 경고를 나란히 본다.
광량은 이 점검의 대상이 아니라 총점 대신 온도·습도 요인 점수, 탈락, 경고만 집계한다.
실행: python -m src.recommend_check [범위표 경로]
"""
import sys

import numpy as np
import pandas as pd

from src import recommend
from src.gee_annual_extremes import ROOT
from src.indoor_check import SEOUL, scenarios
from src.model_data import Climate

OUT = ROOT / "results" / "tolerance" / "recommend_check.csv"


def main() -> None:
    if len(sys.argv) > 1:
        recommend.TOL = __import__("pathlib").Path(sys.argv[1])
    plants = recommend.load_plants()
    raw = np.asarray(Climate().clim[:, SEOUL]).astype(np.float32)
    rows = []
    for name, v in scenarios(raw).items():
        ctx = "실외" if name.startswith("S0") else "실내"
        for y in range(len(v)):  # 연도마다 1년 그래프 하나 = 게임 입력 하나
            env, extra = recommend.summarize(v[y, :, 0], v[y, :, 2], v[y, :, 3], np.full(365, 1000.0), ctx, v[y, :, 1])
            r = recommend.recommend(plants, env, extra)
            rows.append({"시나리오": name, "연도": 2016 + y, "평가_식물": len(r),
                         "탈락_재배구분": int((r["탈락_사유"] == "재배구분 불일치").sum()),
                         "탈락_추위": int(r["탈락_사유"].fillna("").str.contains("하한").sum()),
                         "탈락_더위": int(r["탈락_사유"].fillna("").str.contains("상한").sum()),
                         "습도점수_평균(통과식물)": round(r.loc[r["탈락_사유"].isna(), "습도"].mean(), 1),
                         "모델_건조경고(통과식물)": int(r.loc[r["탈락_사유"].isna(), "모델_건조_경고"].notna().sum()),
                         "방_최저": round(env.temp_min, 1), "방_건조5%": round(extra["rh_p05"], 1)})
    d = pd.DataFrame(rows)
    s = d.groupby("시나리오", sort=False).mean(numeric_only=True).round(1).drop(columns="연도")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT, encoding="utf-8-sig")
    print(s.to_string())


if __name__ == "__main__":
    main()
