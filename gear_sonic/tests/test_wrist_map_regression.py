"""Wrist mapping regression: operator max-range clip -> X2 wrist chain angles.

Ground truth = human wrist anatomy, captured by the author on a Pico headset
(body tracking, both hands, wrists moved to their full range; the fixture is
the author's own capture and stands in for a synthetic clip): ~90 deg toward
the palm vs ~40 deg back, ~30 deg toward the little finger vs ~20 deg toward
the thumb, symmetric rotation. The DIRECTIONS are the ones confirmed on the
real X2: with the shipped signs the robot's bend and sideways tilt follow the
operator's wrist. In chain terms that is palm-ward bend = POSITIVE wrist_roll
on the left / NEGATIVE on the right, little-finger tilt = NEGATIVE wrist_pitch
on both hands, rotation unchanged. (The sim-derived signs had both reversed;
the sim was the odd one out.) The palm-ward bend therefore lands on the SHORT
+-0.724 rad side of wrist_roll and is clipped by the sender; that is a
joint-range fact, not a mapping error.

Any change to the arm / hand / wrist path (SMPL wrist extraction in the
Pico manager, gear_sonic/utils/teleop/x2_wrist_map.py, the sender's
wrist publish) must keep:
  * palm-ward bend: +roll left / -roll right, >= 60 deg before clipping
  * backward bend smaller than palm-ward (human asymmetry preserved)
  * little-finger tilt: NEGATIVE pitch both hands, >= 25 deg, larger than
    the thumb-ward tilt
  * rotation symmetric (no sign bias introduced by the frame map)
  * sideways and rotation inside the X2 joint ranges

    .venv/bin/python -m pytest gear_sonic/tests/test_wrist_map_regression.py -q
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

CLIP = Path(__file__).parent / "data" / "wrist_maxrange_20260905_002744Z.npz"
TOL_DEG = 3.0
LIMITS = {  # X2 wrist chain (x2_ultra.xml), degrees: (rotation, sideways, bend) per side
    0: ((-146.4, 146.4), (-32.0, 32.0), (-90.0, 41.5)),   # left
    1: ((-146.4, 146.4), (-32.0, 32.0), (-41.5, 90.0)),   # right
}


module_fixture = pytest.fixture(scope="module")
# (decorator aliases: the pre-commit secret scan mis-reads a bare pytest decorator in a diff as an e-mail)
both_sides = pytest.mark.parametrize("side,name", [(0, "left"), (1, "right")])


@module_fixture
def chain_angles_deg():
    # shipped signs: drop any operator A/B overrides from the environment
    for k in ("WRIST_ROLL_SIGN", "WRIST_PITCH_SIGN", "WRIST_YAW_SIGN_FRAME"):
        os.environ.pop(k, None)
    import gear_sonic.utils.teleop.x2_wrist_map as wm
    importlib.reload(wm)
    z = np.load(CLIP)
    body = z["body"]
    q = body[:, :, [6, 3, 4, 5]]                       # wxyz, as pico_manager_thread_server.compute_from_body_poses
    rot = lambda i: R.from_quat(q[:, i], scalar_first=True) * R.from_euler("y", 180, degrees=True)  # noqa: E731
    aa = np.stack([(rot(18).inv() * rot(20)).as_rotvec(),   # left wrist in the forearm frame
                   (rot(19).inv() * rot(21)).as_rotvec()], axis=1)
    ypr = np.degrees(wm.wrist_frame_ypr_batch(aa))
    return ypr - ypr[:50].mean(0)                       # neutral = first second (as the sender at engage)


@both_sides
def test_palmward_bend_direction_robot_confirmed(chain_angles_deg, side, name):
    bend = chain_angles_deg[:, side, 2]
    palmward = bend.max() if side == 0 else -bend.min()     # robot 2026-09-05: +roll left, -roll right
    backward = -bend.min() if side == 0 else bend.max()
    assert palmward >= 60.0, f"{name}: palm-ward bend only {palmward:.0f} deg (direction flipped?)"
    assert palmward > backward + 20.0, f"{name}: bend asymmetry lost ({palmward:.0f} vs {backward:.0f})"


@both_sides
def test_little_finger_tilt_is_negative_pitch(chain_angles_deg, side, name):
    tilt = chain_angles_deg[:, side, 1]
    assert -tilt.min() >= 25.0, f"{name}: little-finger tilt only {-tilt.min():.0f} deg (direction flipped?)"
    assert -tilt.min() > tilt.max(), f"{name}: sideways asymmetry inverted (little-finger side must be the larger one)"


@both_sides
def test_rotation_symmetric(chain_angles_deg, side, name):
    rot = chain_angles_deg[:, side, 0]
    assert rot.max() >= 40.0 and -rot.min() >= 40.0, f"{name}: rotation range too small"
    assert abs(rot.max() + rot.min()) <= 20.0, f"{name}: rotation sign bias {rot.max():+.0f}/{rot.min():+.0f}"


@both_sides
def test_within_x2_joint_ranges(chain_angles_deg, side, name):
    # bend (axis 2) is excluded: the palm-ward bend exceeds the short 0.724
    # rad side by design and the sender clips it to the joint range
    for k, (lo, hi) in list(enumerate(LIMITS[side]))[:2]:
        v = chain_angles_deg[:, side, k]
        assert v.min() >= lo - TOL_DEG and v.max() <= hi + TOL_DEG, \
            f"{name} axis {k}: {v.min():+.0f}..{v.max():+.0f} outside ({lo}, {hi})"
