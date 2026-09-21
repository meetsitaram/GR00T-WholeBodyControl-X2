# X2 MC Gesture Recordings

In-repo (LFS-tracked) corpus of mobile-app-triggered Motion Controller
(MC) gestures captured from the real X2 Ultra robot. These exist to
extend the SOMA-retargeted training corpus with style + intent the
operator can demo on a physical robot at any time.

## Layout

```
gear_sonic/data/motions/x2_recorded/
├── balanced_stand.pkl                    # legacy single-clip stand
└── mc_gestures/                          # <-- you are here
    ├── README.md                         # this file
    ├── hug_001.pkl                       # SOMA-schema motion_lib PKL
    ├── hug_002.pkl                       # (single key per file:
    ├── bow_001.pkl                       #  the bare gesture name +
    ├── …                                 #  three-digit take number)
    └── waist_right_002.pkl

gear_sonic/data/motions/x2_recorded/mc_gestures_npz/
└── mc_gestures_<UTC YYYYMMDD>/           # session = capture date
    ├── hug_001.npz                       # raw HAL recording from PC2
    ├── hug_002.npz                       # (source of truth; PKL is
    ├── bow_001.npz                       #  re-derivable from this via
    └── …                                 #  --convert-only)
```

## Capture workflow

See [`mc_gesture_capture_commands.md`](../../../../../mc_gesture_capture_commands.md)
at the repo root for the full operator cheatsheet. Short form:

```bash
# Start the recorder on PC2, trigger the gesture from the mobile app,
# Ctrl-C when the robot returns to rest. Wrapper handles ssh + rsync +
# converter automatically.
./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh hug 001 \
    --pc2-host <PC2_IP> --view
```

## PKL schema

Each PKL is `joblib`-loadable as `dict[motion_key -> entry]` with a
single key matching the filename stem (`hug_001`, `bow_002`, …). The
entry is byte-compatible with the SOMA-schema training corpus
corpus -- drop into the training motion library the same way:

| Field                | Shape       | Notes                                       |
|----------------------|-------------|---------------------------------------------|
| `root_trans_offset`  | `(T, 3)`    | Pelvis world XYZ. With `--anchor-xy` the anchor foot's frame-0 XY is held constant. |
| `root_rot`           | `(T, 4)`    | Pelvis world quaternion `(x, y, z, w)`. With `--root-rot foot-flat` the anchor foot's frame-0 orientation is held constant; pelvis pitch is whatever the leg chain requires for balance. |
| `pose_aa`            | `(T, 32, 3)`| Per-body axis-angle. Body 0 = root; bodies 1..31 = `DOF_AXIS * dof_value`. |
| `dof`                | `(T, 31)`   | 31-DOF body joint trajectory.               |
| `smpl_joints`        | `(T, 24, 3)`| Zeros (X2 has no SMPL skeleton).            |
| `fps`                | int         | 30 (matches the bones-seed corpus).         |

Plus provenance metadata: `x2_record_source_npz`,
`x2_record_dof_source`, `x2_record_root_rot_mode`,
`x2_record_floor_anchor`, `x2_record_anchor_xy`,
`x2_record_root_pose_source`, `x2_record_window_s`.

## Provenance: reproducibility from NPZ

PKLs in this directory are **deterministic outputs** of
`gear_sonic/data_process/convert_x2_record_to_motion_lib.py` applied to
the matching NPZ in `../mc_gestures_npz/<session>/`. If you change the
converter (new `--root-rot` mode, bug fix, schema bump), reconvert ALL
takes and commit the refreshed PKLs alongside the converter patch in
the same change:

```bash
for npz in gear_sonic/data/motions/x2_recorded/mc_gestures_npz/*/*.npz; do
    take=$(basename "$npz" .npz)
    gesture="${take%_*}"; num="${take##*_}"
    ./gear_sonic_deploy/scripts/record_x2_mc_gesture.sh \
        --convert-only --override "$gesture" "$num"
done
```

The wrapper is idempotent: same NPZ + same converter version → same PKL
bytes. Re-running on already-converted takes is safe (it overwrites the
PKL with identical content).

## Default conversion settings

Captured + converted via `record_x2_mc_gesture.sh` with MC-gesture-tuned
defaults:

- `--source state` -- use the encoder positions (what physically happened) rather than MC's commanded targets.
- `--root-rot foot-flat` -- derive pelvis rotation from leg-chain FK so the anchor foot stays at its frame-0 (idle, flat) orientation. Produces physically-correct counter-balance pitch (e.g. pelvis tilts back when arms reach forward); the torso IMU is ignored.
- `--floor-anchor lower-foot --anchor-xy` -- anchor foot is fully pinned in world (orientation + position); pelvis floats to satisfy leg kinematics.
- `--fps 30 --trim 0.5/0.5` -- 30 Hz output (matches bones-seed), 0.5 s shaved off each end.

Override any flag per-take on the wrapper command line if a gesture
needs different treatment (e.g. `--root-rot torso-imu` for a walking
take, `--floor-anchor left-foot` for one-legged balance).

## Why these live in main repo (not a separate dataset repo)

- **Small.** 51 takes + 52 PKLs = ~38 MB through LFS today. Even at 100× the corpus size (~5,000 takes) we'd be at ~4 GB of LFS, which is still trivial.
- **Co-located with code.** The converter, playback wrappers (`play_gesture`, `deploy_x2.sh --motion`), and consumer training configs all live here -- splitting captures into a separate repo would force submodule-SHA bumps on every recording session.
- **Atomic provenance.** NPZ + converter version + PKL all land in the same commit, so `git log` answers "which PKL bytes were produced by which converter from which raw capture" without cross-repo archaeology.
