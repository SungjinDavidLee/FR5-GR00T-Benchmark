# 변환과 학습

## 1. 변환

```bash
$VENV /path/to/to_lerobot_groot.py \
  /path/to/episodes_groot  local/fr5_task6_abs \
  --fps 10 --robot-type fairino_fr5
```

액션은 **다음 프레임의 절대 state 앞 7개**로 만든다. 변환 뒤 `action[t] == state[t+1][:7]`
을 검사해 오차 0 을 확인한다 (`tools/rot_diag.py` 가 같이 찍는다).

## 2. 학습

```bash
cd ~/lerobot
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/lerobot-train \
  --dataset.repo_id=local/fr5_task6_abs \
  --policy.type=groot \
  --policy.pretrained_path=nvidia/gr00t17-lerobot-libero_spatial-640 \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.embodiment_tag=new_embodiment \
  --policy.push_to_hub=false \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.chunk_size=32 --policy.n_action_steps=32 \
  --policy.tune_llm=false --policy.tune_visual=false \
  --policy.tune_projector=true --policy.tune_diffusion_model=true --policy.tune_vlln=true \
  --policy.use_bf16=true \
  --batch_size=16 --steps=6000 --save_freq=6000 --log_freq=50 \
  --num_workers=4 --seed=1000 --wandb.enable=false \
  --output_dir=outputs/fr5_task6_abs --job_name=fr5_task6_abs
```

```
6,000 스텝    37분,  9.16 에폭,  손실 1.911 → 0.006,  메모리 36 GB
12,000 스텝   75분, 18.32 에폭,  손실 0.004          (steps/save_freq/output_dir 만 변경)
학습 파라미터 1.62 B / 전체 3.14 B
```

`--policy.push_to_hub=false` 가 없으면 `repo_id` 를 요구하며 시작 전에 멈춘다.

## 3. 6,000 대 12,000 — 더 학습하면 나빠진다

12,000 스텝 체크포인트는 손실이 더 낮은데도 **물체 위치를 따라가지 못한다.**

```
6,000 스텝    그릇이 10 cm 이동 → 파지점 3.5 cm 이동   (추종 0.35)
12,000 스텝   그릇이  7 cm 이동 → 파지점 0.8 cm 이동   (추종 ≈ 0.1, 4판 표본)
```

12,000 스텝은 세트가 바뀌어도 거의 같은 자리로 간다. 205판에 18 에폭은 과학습이며,
본 평가에는 6,000 스텝을 썼다.

## 4. 추론 서버

```bash
cd ~/lerobot
CKPT=~/lerobot/outputs/fr5_task6_abs/checkpoints/006000/pretrained_model \
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python fr5_policy_server.py
```

`POST /act` 로 관측(256×256 두 장 + 8차원 상태)을 받아 **[32, 7] 청크**를 돌려준다.
denoise 16, 1회 약 400 ms. 학습과 같은 정규화 통계를 쓰기 위해 `dataset_meta` 를 넘기고
영상에는 ImageNet 통계를 주입한다 — 이걸 빠뜨리면 재현 오차가 크게 벌어진다.

학습 프레임을 되먹여 재현 오차를 재면 **위치 중앙 30.6 mm** 다 (`tools/probe_policy.py`).
