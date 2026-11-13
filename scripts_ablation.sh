#!/bin/sh
# 부품 제거 실험(설계 3단계 6) — 시드 0 하나, 겹 0–4(설정 1) + 겹 -1(설정 2), 본 모델과 같은 조합·단계(B p8 λ1, 1,250단계).
# 순서: 빠르고 정보 큰 것부터. 끝난 것은 건너뜀(이어 하기). 결과: results/eval/summary_abl-*.json
# 판정 규칙(결과 전 고정, docs/model_design.md 3-5): 본 모델 시드 0 대비 차이가 '측정한 시드 표준편차 × 2'보다 크면 의미 있는 차이.
cd "$(dirname "$0")" || exit 1
for AB in stats_mlp no_ctype no_pretrain unbalanced mean_input; do
  for F in 0 1 2 3 4 -1; do
    d="results/align/final/fold$F/B_p8_lam1_abl-$AB"
    if [ ! -f "$d/summary.json" ]; then
      echo "학습 fold$F $AB $(date '+%m-%d %H:%M')"
      .venv/bin/python -u -m src.align --final --fold "$F" --loss B --patch 8 --lam 1 --lp_steps 1000 --ft_steps 250 --seed 0 --ablate "$AB" || exit 1
    fi
    .venv/bin/python -u -m src.evaluate score --fold "$F" --seed 0 --ablate "$AB" || exit 1
  done
  .venv/bin/python -u -m src.evaluate report --ablate "$AB" > "results/eval/report_abl-$AB.txt" 2>&1
  echo "완료 $AB $(date '+%m-%d %H:%M')"
done
echo "전체 완료 $(date '+%m-%d %H:%M')"
