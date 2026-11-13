"""손실·패치 후보 선택 — 결과를 보기 전(11-09, 겹 1–4 실행 시작 시점)에 고정한 규칙.

지표: 후보마다 겹별 "미세조정 구간(1,250~4,000단계) 내부 AUC 평균" → 5겹 평균과 표준오차(표준편차/√5).
  최고값 하나 대신 여러 평가의 평균을 쓰는 이유: 겹 0에서 평가 간 흔들림(표준편차 0.007~0.016)이 후보 간 차이와 비슷했고,
  최고값 선택은 흔들림의 최대값을 고르는 낙관적 선택이 됨.
규칙: 1-표준오차 규칙 — 평균이 가장 높은 후보의 (평균 − 1 표준오차) 이상인 후보 중 가장 단순한 것.
  근거: James, Witten, Hastie & Tibshirani, An Introduction to Statistical Learning(7쇄) 6.1.3절 p.214
  "select the smallest model for which the estimated test error is within one standard error of the lowest point on the curve.
   ... if a set of models appear to be more or less equally good, then we might as well choose the simplest model"(원문 확인).
단순함 순서(미리 정함): ① 조절값 수가 적은 것 — A(λ 없음) < B(λ 있음) ② 같으면 학습 시간이 짧은 것 — 패치 16 < 패치 8
  ③ B끼리는 λ1(가중 없음) < λR−1.
실행: python -m src.align_select → results/align/selection.json
"""

import json

import numpy as np

from src.gee_annual_extremes import ROOT

SEL = ROOT / "results" / "align" / "select"
ORDER = ["A_p16", "A_p8", "B_p16_lam1", "B_p16_lamR-1", "B_p8_lam1", "B_p8_lamR-1"]  # 단순한 순
FT_FROM = 1250
FOLDS = range(5)


def ft_mean(path) -> float:
    recs = [json.loads(l) for l in open(path / "log.jsonl")]
    return float(np.mean([r["inner_auc"] for r in recs if r["step"] >= FT_FROM]))


def main() -> None:
    table = {}
    for tag in ORDER:
        vals = [ft_mean(SEL / f"fold{f}" / tag) for f in FOLDS]
        table[tag] = {"per_fold": vals, "mean": float(np.mean(vals)), "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals)))}
    best = max(table, key=lambda t: table[t]["mean"])
    bar = table[best]["mean"] - table[best]["se"]
    chosen = next(t for t in ORDER if table[t]["mean"] >= bar)
    out = {"rule": "1-SE, 단순한 순 " + " < ".join(ORDER), "best_mean": best, "threshold": bar, "chosen": chosen,
           "table": table}
    (SEL.parent / "selection.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for t in ORDER:
        print(f"{t:14s} {table[t]['mean']:.3f} ± {table[t]['se']:.3f}  {np.round(table[t]['per_fold'], 3)}")
    print("최고 평균", best, "기준", round(bar, 3), "→ 선택", chosen)


if __name__ == "__main__":
    main()
