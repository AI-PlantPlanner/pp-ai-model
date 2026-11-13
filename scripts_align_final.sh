#!/bin/sh
# 최종 학습: 선택된 조합(손실 B, 패치 8, λ1 — results/align/selection.json)으로 바깥 겹마다, 시드 5개.
# 단계 수 1,250(고정 1,000 + 미세조정 250): 내부 검증 5겹 평균 AUC 곡선이 최고(0.848)인 단계.
#   근거: Goodfellow·Bengio·Courville, Deep Learning 7.8절 알고리즘 7.2 — 조기 종료로 찾은 단계 수만큼 학습 자료 전체로 다시 학습.
# 사용: sh scripts_align_final.sh  (끝난 실행은 summary.json 있으면 건너뜀)
cd "$(dirname "$0")" || exit 1
for S in 0 1 2 3 4; do
  for F in 0 1 2 3 4; do
    if [ "$S" = 0 ]; then tag="B_p8_lam1"; else tag="B_p8_lam1_s$S"; fi
    if [ -f "results/align/final/fold$F/$tag/summary.json" ]; then echo "건너뜀 fold$F $tag"; continue; fi
    echo "시작 fold$F $tag $(date '+%m-%d %H:%M')"
    .venv/bin/python -u -m src.align --final --fold "$F" --loss B --patch 8 --lam 1 \
      --lp_steps 1000 --ft_steps 250 --seed "$S" || exit 1
  done
done
echo "전체 완료 $(date '+%m-%d %H:%M')"
