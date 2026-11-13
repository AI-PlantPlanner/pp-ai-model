#!/bin/sh
# 설정 2(학습에 있던 식물)용: 전 종(76식물)으로 학습, 선택된 조합·단계 수·시드는 최종 학습과 같음. 겹 -1 = 평가 겹 없음.
cd "$(dirname "$0")" || exit 1
for S in 0 1 2 3 4; do
  if [ "$S" = 0 ]; then tag="B_p8_lam1"; else tag="B_p8_lam1_s$S"; fi
  if [ -f "results/align/final/fold-1/$tag/summary.json" ]; then echo "건너뜀 fold-1 $tag"; continue; fi
  echo "시작 fold-1 $tag $(date '+%m-%d %H:%M')"
  .venv/bin/python -u -m src.align --final --fold -1 --loss B --patch 8 --lam 1 --lp_steps 1000 --ft_steps 250 --seed "$S" || exit 1
done
echo "전체 완료 $(date '+%m-%d %H:%M')"
