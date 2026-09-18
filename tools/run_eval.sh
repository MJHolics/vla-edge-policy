#!/usr/bin/env bash
# LIBERO 평가 실행 래퍼 — 카메라 이름·개수 불일치를 여기서 흡수한다.
#
# 왜 래퍼가 필요한가 (2026-08-18에 물린 것):
#   `lerobot/smolvla_libero` 정책은 카메라 3대(`camera1/2/3`)를 기대하는데
#   LIBERO env는 2대(`image`, `image2`)를 준다. 그대로 돌리면 make_policy에서
#   "Feature mismatch"로 죽는다.
#
#   해결은 `--rename_map` 하나다. image→camera1, image2→camera2 로 이름을 맞추면
#   검증(`validate_visual_features_consistency`)이 **한쪽이 부분집합이면 통과**시키므로
#   {camera1,camera2} ⊆ {camera1,camera2,camera3} 가 되어 지나간다.
#
# 2026-08-19 정정 — 여기엔 원래 `--policy.empty_cameras=1` 도 같이 있었다. 처음 크래시를 고칠 때
#   rename_map 과 **함께** 넣었고, 에러가 사라졌으므로 둘 다 필요하다고 믿었다. 아니었다.
#   `train_config.json` 의 `empty_cameras=0` 이고 학습 데이터셋 `lerobot/libero` 의 카메라는
#   **2대뿐**이다. `modeling_smolvla.prepare_images` 는 그 값이 0이면 없는 카메라를
#   패딩하지 않고 건너뛴다 → **학습은 이미지 2장으로 이뤄졌다.**
#   1로 두면 학습에 없던 -1 패딩 이미지가 매 추론에 들어간다(A/B: 64% vs 60%, 구간 겹침 —
#   해롭다고 말할 근거는 없지만 학습 조건과 다르고 비전 토큰만 늘린다).
#   → 기본을 **0**(학습 조건)으로 되돌린다.
#
# 사용: bash tools/run_eval.sh <policy_path> <suite> <task_ids> <n_episodes> <batch_size> <out_dir> [추가인자...]
set -u

POLICY="${1:-lerobot/smolvla_libero}"
SUITE="${2:-libero_spatial}"
TASK_IDS="${3:-[0]}"
N_EPISODES="${4:-2}"
BATCH="${5:-2}"
OUT="${6:-/root/vla/results/eval}"
shift 6 2>/dev/null || true

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "정책=$POLICY suite=$SUITE tasks=$TASK_IDS episodes=$N_EPISODES batch=$BATCH"
echo "출력=$OUT"
START=$(date +%s)

lerobot-eval \
  --policy.path="$POLICY" \
  --policy.device=cuda \
  --policy.empty_cameras=0 \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size="$BATCH" \
  --eval.n_episodes="$N_EPISODES" \
  --output_dir="$OUT" \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  "$@"
RC=$?

END=$(date +%s)
echo "종료코드=$RC · 소요 $((END-START))초"
exit $RC
