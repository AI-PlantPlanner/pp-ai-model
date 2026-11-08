#!/bin/sh
# 손실 A/B × 패치 × λ 후보를 한 바깥 겹에서 내부 검증 (src/align.py). 사용: sh scripts_align_grid.sh 0
F=$1
for args in "--loss A --patch 16" "--loss B --patch 16 --lam 1" "--loss B --patch 16 --lam R-1" \
            "--loss A --patch 8" "--loss B --patch 8 --lam 1" "--loss B --patch 8 --lam R-1"; do
  .venv/bin/python -u -m src.align --fold $F $args || exit 1
done
