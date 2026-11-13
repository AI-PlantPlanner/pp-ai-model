#!/bin/sh
# 손실 A/B × 패치 × λ 후보를 바깥 겹마다 내부 검증 (src/align.py).
# 사용: sh scripts_align_grid.sh 1 2 3 4
# 이어 하기: 끝난 후보(summary.json 있음)는 건너뛴다. 중간에 끊긴 후보는 처음부터 다시 돈다.
# 노트북 덮개를 닫으면 잠자기 동안 멈췄다가 깨면 이어서 진행(프로세스는 유지).
cd "$(dirname "$0")" || exit 1
for F in "$@"; do
  for c in "A 16 -" "B 16 1" "B 16 R-1" "A 8 -" "B 8 1" "B 8 R-1"; do
    set -- $c
    if [ "$1" = A ]; then tag="A_p$2"; extra=""; else tag="B_p$2_lam$3"; extra="--lam $3"; fi
    if [ -f "results/align/select/fold$F/$tag/summary.json" ]; then echo "건너뜀 fold$F $tag"; continue; fi
    echo "시작 fold$F $tag $(date '+%m-%d %H:%M')"
    .venv/bin/python -u -m src.align --fold "$F" --loss "$1" --patch "$2" $extra || exit 1
  done
done
echo "전체 완료 $(date '+%m-%d %H:%M')"
