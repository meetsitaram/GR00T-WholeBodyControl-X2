# X2 Pico chores reference set

Operator Pico full-body teleop session on the AgiBot X2 Ultra (house-chores
style upper-body work with locomotion), recorded on the robot and shipped here
as the reference motion set for every smoke and training-run test in this
repository. It is first-party data: the operator's own body motion and the
robot's executed track. No third-party mocap corpus is involved.

| File | What | Format |
|---|---|---|
| `x2_pico_chores_0917.pkl` | 14 retargeted robot clips, 878 s total, 50 fps | motion-lib pkl: `dof (T,31)`, `root_trans_offset`, `root_rot` (xyzw), `pose_aa`, `fps` |
| `smpl_sidecars/<key>.pkl` | one SMPL sidecar per clip (100% coverage) | `pose_aa (T,72)`, `transl (T,3)`, `smpl_joints (T,24,3)`, `fps`; corpus frame (y-up, pelvis rest offset) |
| `tapes/session_*.npz` | two raw Pico session tapes (seg01, seg02) | `gear_sonic/utils/teleop/pico_tape.py` schema; `run_pico_replay.sh` input |

How it was built (all shipped tools): tapes -> `gear_sonic/data_process/build_pico_finetune_bundle.py`
(retargeted clips, despiked per joint: 438 wrap-flip frames of 43,903 were
interpolated, max joint step 256 -> 27 rad/s; + `pico_tape_to_smpl_obs.py` sidecars), then gated with
`gear_sonic/scripts/check_smpl_sidecars.py` (14 PASS / 0 FAIL) and
`gear_sonic/scripts/cloud/preflight_train.py` (coverage 14/14, no timebase
mismatches). Six clips of the session whose SMPL conversion failed the elbow
round-trip gate are not included, so every shipped clip is fully paired.

Used by default in: `run_smoke_8gpu.sh` (`MOTION_FILE`, `SMPL_MOTION_DIR`),
the X2 training configs, `preflight_train.py`, and as a real-tape input for
`run_pico_replay.sh` (`tapes/`). Override with your own corpus via the same
variables (see MODELS.md).
