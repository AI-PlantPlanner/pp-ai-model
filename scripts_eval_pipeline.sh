#!/bin/sh
# 최종 학습(scripts_align_final.sh)이 끝나길 기다렸다가 → 설정 2 학습 → 전 모델 점수 → 지표.
# GPU를 학습과 나눠 쓰면 점수 계산이 크게 느려져(11-10 측정: 2,048칸 한 묶음 26초) 순서대로 실행.
cd "$(dirname "$0")" || exit 1
while pgrep -f scripts_align_final.sh >/dev/null; do sleep 60; done
grep -q "전체 완료" results/align/run_final.log || { echo "최종 학습이 완료되지 않음"; exit 1; }
sh scripts_align_seen.sh || exit 1
echo "점수 시작 $(date '+%m-%d %H:%M')"
.venv/bin/python -u -m src.evaluate score-all || exit 1
echo "점수 완료 $(date '+%m-%d %H:%M')"
.venv/bin/python -u -m src.evaluate report
echo "평가 완료 $(date '+%m-%d %H:%M')"
