# Kplanner model registry (X2 planner: vqvae / root / pose)

Why this exists: checkpoints accumulate across training eras in
differently-named roots with no index, and a warm-start is easily restarted
from the wrong root. Keep ONE registry per training host, regenerated from
the checkpoint tree, and archive superseded roots with an
`ARCHIVED_DO_NOT_USE.md` marker.

## Layout the runtime expects (bring-your-own, see MODELS.md)

| File | Identity |
|---|---|
| `kplanner_torch/vqvae/x2_kplanner_vqvae.ckpt` | vqvae_x2 checkpoint the pose / root models were trained against |
| `kplanner_torch/pose/x2_kplanner_pose.ckpt` | pose_x2 checkpoint |
| `kplanner_torch/root/x2_kplanner_root.ckpt` | root_x2 checkpoint (full Lightning payload incl. optimizer states, so a true resume is possible) |
| `kplanner_onnx/x2_kplanner_{template,velocity}.onnx` | exported from the trio above (`scripts/export_x2_planner_onnx.py`) |

The launchers resolve these under `$SONIC_HOME/x2/` (default `~/.cache/sonic/x2`);
`gear_sonic/scripts/download_from_hf.py` fills that layout from a Hugging Face repo.

## Rules

- Every new training run lands in an existing CURRENT root or adds its row to
  the host registry in the same commit that launches it.
- Record the md5 of every shipped checkpoint next to the training step it
  came from; a "459k export" that is really the 300k checkpoint is exactly
  the confusion this file prevents.
- Mirrors (rsync targets, object storage) are NOT authoritative and can lag.

## Training chain template

| Run | State | Step | Host |
|---|---|---|---|
| `vqvae_<corpus>_<steps>` | finished | | |
| `pose_<corpus>_<steps>` | | | |
| `root_<corpus>_<steps>` | | | |

`scripts/train_vqvae_x2.py` -> `scripts/train_root_x2.py` / `scripts/train_pose_x2.py`
(both consume the vqvae checkpoint) -> `scripts/export_x2_planner_onnx.py`.
