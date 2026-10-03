#!/usr/bin/env bash
# 절제 실험 — 규칙 하나씩 끄고 10판씩. 한 줄씩 복사해 실행한다.
# 각 조건은 세트 1~10 을 같은 순서로 쓴다. 판정은 기준과 동일(s / f + 단계 + 원인).
set -u
P=/home/wonseok/miniforge3/envs/grounded_sam2_thor/bin/python
R=~/fr5cap/bench_runner_v4.py
COMMON="--from 1 --to 10 --force-cycle 70"

echo "조건을 하나 고른 뒤 그 줄만 실행하라. 한 조건 10판에 20~30분."

# ── 기준 ────────────────────────────────────────────────────────────────────
# 규칙 여섯 개를 모두 켠 상태. 이미 30판을 측정했으므로 보통은 건너뛴다.
run_base() {
  LATCH_MODE=full GRASP_DZ=0.008 $P $R --tag base $COMMON
}

# ── A 끄기 — 파지 구간 하강 없음 (완료: 9/10) ───────────────────────────────
run_noA() {
  LATCH_MODE=full GRASP_DZ=0 $P $R --tag noA_dz $COMMON
}

# ── C 끄기 — 파지 후 잠금 없음 ──────────────────────────────────────────────
# 쥔 뒤 정책이 내는 폭을 그대로 따른다. 이송 중 벌어져 떨어뜨리는 판이 늘 것이다.
run_noC() {
  LATCH_MODE=full GRASP_DZ=0 GRIP_LATCH=0 $P $R --tag noC_latch $COMMON
}

# ── E 끄기 — 중간 폭을 그대로 보냄 ──────────────────────────────────────────
# 접근 중 손가락이 떨린다. 그릇을 밀어내면 그 판은 무효(x)로 찍는다.
run_noE() {
  LATCH_MODE=full GRASP_DZ=0 GRIP_BINARY=0 $P $R --tag noE_binary $COMMON
}

# ── B 끄기 — 폐쇄 높이 제한 없음 ────────────────────────────────────────────
# 공중에서 닫는 판이 생긴다. 위험은 낮고 헛잡음이 늘어난다.
run_noB() {
  LATCH_MODE=full GRASP_DZ=0 GRIP_Z=9 $P $R --tag noB_gripz $COMMON
}

# ── D 끄기 — 해제 높이 제한 없음 ────────────────────────────────────────────
# 높은 곳에서 떨어뜨리는 판이 생긴다. 접시 주변을 비우고 진행한다.
run_noD() {
  LATCH_MODE=full GRASP_DZ=0 REL_Z=9 $P $R --tag noD_relz $COMMON
}

# ── F 끄기 — 정책의 회전 출력을 사용 ────────────────────────────────────────
# 목표 자세가 크게 달라진다. MAXJUMP 를 15 로 낮추고 비상정지에 손을 올린 채 진행한다.
run_noF() {
  LATCH_MODE=full GRASP_DZ=0 ORIENT=policy MAXJUMP=15 $P $R --tag noF_orient $COMMON
}

case "${1:-}" in
  base) run_base ;;  A) run_noA ;;  B) run_noB ;;  C) run_noC ;;
  D) run_noD ;;      E) run_noE ;;  F) run_noF ;;
  *) echo "사용법: bash run_ablations.sh [base|A|B|C|D|E|F]" ;;
esac
