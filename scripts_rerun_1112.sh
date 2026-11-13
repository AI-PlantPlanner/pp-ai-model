#!/bin/sh
# 11-12 학명 재확인(구즈베리 → R. uva-crispa, 마지나타 → D. reflexa 원종 근사) 반영 후 전부 다시 돌리기.
# 설정(손실 B, 패치 8, λ1, 1,250단계, 시드 5개)·판정 규칙은 그대로. 이전 결과는 results/_pre_addendum_1112/.
cd "$(dirname "$0")" || exit 1
echo "시작 $(date '+%m-%d %H:%M')"
.venv/bin/python -u -m src.baseline_stats > results/baseline_rerun.log 2>&1 || { echo "베이스라인 실패"; exit 1; }
echo "베이스라인 완료 $(date '+%m-%d %H:%M')"
sh scripts_align_final.sh > results/align/run_final.log 2>&1 || { echo "최종 학습 실패"; exit 1; }
echo "최종 학습 완료 $(date '+%m-%d %H:%M')"
sh scripts_align_seen.sh || exit 1
.venv/bin/python -u -m src.evaluate score-all || exit 1
.venv/bin/python -u -m src.evaluate report || exit 1
echo "평가 완료 $(date '+%m-%d %H:%M')"
.venv/bin/python -u -m src.significance || exit 1
.venv/bin/python -u -m src.tolerance_table || exit 1
.venv/bin/python -u -m src.indoor_check || exit 1
echo "검정·범위표·실내 점검 완료 $(date '+%m-%d %H:%M')"
sh scripts_ablation.sh > results/ablation.log 2>&1 || { echo "부품 제거 실패"; exit 1; }
echo "전체 완료 $(date '+%m-%d %H:%M')"
