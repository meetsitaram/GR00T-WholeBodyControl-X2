#!/usr/bin/env python3
"""MuJoCo <-> ROS 2 bridge for the AgiBot X2 Ultra.

This is the X2 sibling of ``gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py``
(the G1 bridge). It lets the C++ deploy node ``agi_x2_deploy_onnx_ref`` run
in **closed-loop** against MuJoCo without any real robot:

    [MuJoCo physics @ 1 kHz]
        |---- joint state ---->  /aima/hal/joint/{leg,waist,arm,head}/state  ----+
        |---- IMU ------------>  /aima/hal/imu/torso/state                       |
        |                                                                        |
        +<---- joint command --  /aima/hal/joint/{leg,waist,arm,head}/command <--+
                                                                                 |
                                                                       [C++ deploy]

Topology (matches `aimdk_io.{hpp,cpp}`):
  - 4 publishers, one per joint group (leg=12, waist=3, arm=14, head=2),
    `aimdk_msgs/JointStateArray`, joint **names** + position + velocity
    in **MuJoCo joint order** (per group). The deploy will FATAL if the
    names don't match `mujoco_joint_names[start..start+len)`.
  - 1 publisher for `/aima/hal/imu/torso/state` (`sensor_msgs/Imu`).
    Despite the topic name we publish PELVIS-frame quantities, because
    the MJCF only has live IMU sensors at the pelvis (`imu_0` site) and
    that's what training implicitly trained against. Override with
    `--imu-from torso` to publish from torso_link if you want to mimic the
    real sensor mounting (sim2real residual).
  - 4 subscribers for `/aima/hal/joint/{leg,waist,arm,head}/command`
    (`aimdk_msgs/JointCommandArray`). Each callback updates a per-joint
    PD setpoint table; the physics loop applies
        tau = stiffness * (target_pos - q) + damping * (target_vel - dq) + effort_ff
    every step, clipped to the actuator's `actuatorfrcrange`.

QoS: `rclpy.qos.qos_profile_sensor_data` (BEST_EFFORT + VOLATILE +
KEEP_LAST(10)) — exactly what `aimdk_io.cpp` uses for both subs and pubs,
so the handshake matches with no warnings.

Defaults to ``ROS_LOCALHOST_ONLY=1`` and a quirky ``ROS_DOMAIN_ID`` so a
sim run on the SDK ethernet can't fight a real robot. Override via env.

The constants (joint names, IL/MJ remap, KP/KD, default standing pose)
are imported from ``gear_sonic/scripts/eval_x2_mujoco.py`` -- the same
module that's the source of truth for ``policy_parameters.hpp``. This
makes the bridge auto-update whenever `eval_x2_mujoco` changes; if the
two ever drift, the deploy's name-mismatch FATAL is the loud failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pathlib
import sys
import threading
import time
from typing import Optional

import numpy as np
import scipy.spatial.transform

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class _NullCtx:
    """No-op context manager used when a viewer.lock() isn't available."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


class ElasticBand:
    """Virtual spring that hangs the robot's base body from world ``[0,0,1]``.

    Verbatim port of ``gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:ElasticBand``
    (the G1 sim's anti-faceplant device). We do NOT import that module directly
    because it pulls in ``unitree_sdk2py`` at top level, which is not installed
    in the X2 deploy container. Keep the two copies in lockstep -- if you tune
    one, tune the other.

    Usage matches ``base_sim.py``:
      - viewer mode: register ``MujuocoKeyCallback`` so key 9 toggles, 7/8 raise
        / lower the suspension height
      - per sim step: write ``mj_data.xfrc_applied[band_attached_link] =
        band.Advance(pose)`` when ``band.enable`` is True (else zero it out)

    The pose layout passed to ``Advance`` is the one ``base_sim.py`` builds:
        [pos(3), quat_wxyz(4), lin_vel_world(3), ang_vel_world(3)]
    where the velocities come out of ``mj_objectVelocity(... flg_local=0)`` and
    the angular/linear halves are SWAPPED to put linear first (MuJoCo's
    objectVelocity returns angular-first).
    """

    def __init__(self):
        self.kp_pos = 10000.0
        self.kd_pos = 1000.0
        self.kp_ang = 1000.0
        self.kd_ang = 10.0
        self.point = np.array([0.0, 0.0, 1.0])
        self.length = 0.0
        self.enable = True

    def Advance(self, pose: np.ndarray) -> np.ndarray:
        pos     = pose[0:3]
        quat    = pose[3:7]   # wxyz
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        dx = self.point - pos
        f = (self.kp_pos * (dx + np.array([0.0, 0.0, self.length]))
             + self.kd_pos * (0.0 - lin_vel))

        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
        rotvec = rot.as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        # Only import glfw if the viewer actually drives a key event;
        # headless runs never trigger this path.
        import glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        elif key == glfw.KEY_8:
            self.length += 0.1
        elif key == glfw.KEY_9:
            self.enable = not self.enable
            print(f"[bridge] ElasticBand enable: {self.enable}", flush=True)


def _load_eval_x2():
    """Import ``eval_x2_mujoco`` for the canonical X2 constants (no side effects).

    Mirrors the trick used by ``codegen_x2_policy_parameters.py`` -- importing
    by file path so we don't need ``gear_sonic`` on PYTHONPATH.
    """
    p = REPO_ROOT / "gear_sonic" / "scripts" / "eval_x2_mujoco.py"
    if not p.is_file():
        raise FileNotFoundError(f"eval_x2_mujoco.py not found at {p}")
    spec = importlib.util.spec_from_file_location("eval_x2_mujoco", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_x2_mujoco"] = mod
    spec.loader.exec_module(mod)
    return mod


# Per-group ranges in MUJOCO_JOINT_NAMES order; match
# `kLegStart/kWaistStart/kArmStart/kHeadStart` in `aimdk_io.hpp`.
GROUPS = [
    ("leg",   "/aima/hal/joint/leg",    0, 12),
    ("waist", "/aima/hal/joint/waist", 12,  3),
    ("arm",   "/aima/hal/joint/arm",   15, 14),
    ("head",  "/aima/hal/joint/head",  29,  2),
]


# ----------------------------------------------------------------------------
# Init-pose YAML loader
# ----------------------------------------------------------------------------
# The bridge supports a set of named "init poses" -- the joint-space and
# floating-base configuration the simulated robot starts in before the
# deploy node connects. ``--init-pose default`` is hard-coded inside the
# bridge (DEFAULT_DOF + pelvis_z=0.85 with the band engaged); every other
# named pose is loaded from
#
#   gear_sonic_deploy/configs/sim_init_poses.yaml
#
# That YAML maps a name (e.g. ``gantry_hang``, ``gantry_dangle``) to a
# pelvis_z, an ElasticBand suspension length, and a per-joint offset
# table from DEFAULT_DOF. Operators edit the YAML to add or tune poses
# without touching this file. See ``sim_init_poses.yaml`` itself for the
# schema.
#
# Numbers in the shipped YAML were captured live from the real robot via
# ``gear_sonic_deploy/scripts/x2_capture_pose.py`` and post-processed
# (symmetrize L/R, drop noise) so sim mirrors a defensible operating
# point. The original captures are kept under ``scripts/_capture_*.json``.
_INIT_POSES_YAML_PATH = pathlib.Path(__file__).resolve().parents[1] / \
    "configs" / "sim_init_poses.yaml"


def _load_named_init_poses(path: pathlib.Path = _INIT_POSES_YAML_PATH) -> dict:
    """Read the init-poses YAML; return ``{}`` if it's missing.

    The bridge falls back to the hard-coded ``default`` pose when the
    YAML is absent or empty, so a bad / missing config never bricks
    sim startup. Bad entries (missing required keys, joint name typos)
    raise loudly at first use, since silently dropping them would mask
    real bugs.
    """
    if not path.is_file():
        print(f"[bridge] init-pose YAML not found at {path}; only "
              f"--init-pose=default will be available.", file=sys.stderr)
        return {}
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            f"PyYAML is required to load {path}. Install python3-yaml or "
            "use --init-pose=default."
        ) from e
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping of pose-name -> entry."
        )
    return raw


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run with viewer, robot held in default standing pose until deploy connects:
  python gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py --viewer

  # Headless, RSI from a motion frame:
  python gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py \\
      --motion gear_sonic/data/motions/x2_ultra_standing_only.pkl \\
      --init-frame 0
""",
    )
    p.add_argument("--mjcf", type=pathlib.Path, default=None,
                   help="MJCF path (default: gear_sonic/data/assets/.../x2_ultra.xml). "
                        "Mutually exclusive with --with-omnihand (which programmatically "
                        "composes a different model).")
    p.add_argument("--with-omnihand", action="store_true",
                   help="Compose the X2 + OmniHand augmented MJCF in-memory via "
                        "gear_sonic.scripts.compose_x2_with_omnihand instead of "
                        "loading the bare x2_ultra.xml. Replaces the dummy wrist "
                        "stubs with the 10-active-DOF (per side) OmniHand "
                        "fingers. The body joints and actuator names are "
                        "unchanged so the deploy harness sees the exact same "
                        "31-DoF interface; the hand qpos slots are driven by "
                        "an optional ZMQ subscriber (see --hand-zmq-host / "
                        "--hand-zmq-port).")
    p.add_argument("--hand-zmq-host", type=str, default="localhost",
                   help="Host the bridge SUBs on for OmniHand finger commands "
                        "when --with-omnihand is set. Default 'localhost' "
                        "(matches the live VLA bridge's PUB).")
    p.add_argument("--hand-zmq-port", type=int, default=5556,
                   help="Port for the OmniHand SUB (default 5556 -- the same "
                        "port the SONIC C++ harness already subscribes to). "
                        "ZMQ PUB/SUB is multi-subscriber-safe so the bridge "
                        "can ride alongside the harness.")
    p.add_argument("--hand-zmq-topic", type=str, default="pose",
                   help="ZMQ topic prefix for the OmniHand SUB. Must match "
                        "the live VLA bridge's --pub-topic (default 'pose').")
    p.add_argument("--no-hand-zmq", action="store_true",
                   help="Skip the OmniHand ZMQ subscriber even when "
                        "--with-omnihand is set. Useful for static viewer "
                        "screenshots: fingers stay at their MJCF rest pose.")
    # ── G1 / robocasa scene plumbing ───────────────────────────────────
    # When --mjcf points at a static scene XML built by
    # gear_sonic/scripts/build_x2_robocasa_scene_xml.py, the bridge:
    #   (a) auto-discovers the scene metadata sidecar (same stem, .json),
    #   (b) PUBs cube/bowl/etc. freejoint qpos at state-rate so the
    #       recorder's RobocasaTaskMirror can compute success/reward,
    #   (c) SUBs to scene_reset on a separate port so the recorder can
    #       inject fresh per-episode object poses (placement_initializer
    #       output) into the bridge's MuJoCo at episode start.
    # All three behaviours are silent no-ops on a "no scene metadata
    # found" mjcf so existing flat-floor sims keep working unchanged.
    p.add_argument("--scene-metadata", type=pathlib.Path, default=None,
                   help="Override scene metadata JSON path (defaults to "
                        "<mjcf>.json next to the MJCF when --mjcf is set).")
    p.add_argument("--scene-state-pub-port", type=int, default=5559,
                   help="ZMQ PUB port for the scene_state topic when scene "
                        "metadata is detected. The recorder's "
                        "RobocasaTaskMirror SUBs here to receive cube / "
                        "bowl freejoint qpos at state-rate.")
    p.add_argument("--scene-state-pub-host", default="*",
                   help="Bind iface for the scene_state PUB (default '*').")
    p.add_argument("--scene-reset-sub-port", type=int, default=5560,
                   help="ZMQ SUB port for the scene_reset topic. The "
                        "recorder publishes fresh object poses here at "
                        "episode start; the bridge writes them into "
                        "mj_data and re-runs mj_forward.")
    p.add_argument("--scene-reset-sub-host", default="localhost",
                   help="Connect iface for the scene_reset SUB.")
    p.add_argument("--no-scene-zmq", action="store_true",
                   help="Disable the scene_state PUB / scene_reset SUB even "
                        "when scene metadata is detected. Useful when running "
                        "the bridge against a scene XML purely for viewing.")
    p.add_argument("--robot-pose-pub-port", type=int, default=5570,
                   help="ZMQ PUB port for the ground-truth pelvis pose "
                        "telemetry (sim-only; the real robot has no equivalent "
                        "since it has no ground-truth XY). Host-side tools "
                        "(browse_x2_planner_primitives.py --with-sonic) "
                        "subscribe here to capture per-primitive distance "
                        "and yaw deltas. Set to 0 to disable.")
    p.add_argument("--robot-pose-pub-host", default="*",
                   help="Bind iface for the robot_pose PUB (default '*').")
    p.add_argument("--motion", type=pathlib.Path, default=None,
                   help="Optional motion source for Reference State Initialization (RSI). "
                        "Accepts a motion-lib .pkl (same loader eval_x2_mujoco_onnx.py uses). "
                        "Mutually exclusive with --init-pose=gantry-hang. "
                        "If omitted, the robot starts in --init-pose (default: 'default').")
    p.add_argument("--init-frame", type=int, default=0,
                   help="Motion frame to RSI from (default 0).")
    p.add_argument("--init-pose", default="default",
                   help="Initial pose when --motion is NOT set. 'default' is "
                        "the legacy hard-coded behaviour (DEFAULT_DOF at "
                        "pelvis_z=0.85m, band fully engaged). Any other name "
                        "is looked up in configs/sim_init_poses.yaml -- ship a "
                        "new pose by editing that file, no code changes "
                        "needed. The shipped YAML defines 'gantry_hang' "
                        "(MC-stand handoff pose, captured from the real X2) "
                        "and 'gantry_dangle' (zero-torque hanging pose, "
                        "captured from the real X2). Hyphens and underscores "
                        "are equivalent in pose names.")
    p.add_argument("--band-length", type=float, default=0.0,
                   help="ElasticBand suspension length (m). The band's anchor "
                        "is at world [0,0,1.0]; --band-length offsets the pull "
                        "target downward to [0,0,1.0-LENGTH]. Default 0.0 "
                        "(target at z=1.0, lifts pelvis above standing). For "
                        "the gantry-hang profile use ~0.22 to put the pull "
                        "target ~3 cm above the bent pelvis (~88 %% body "
                        "weight supported, ~12 %% on feet). Viewer keys 7/8 "
                        "still tweak at runtime.")
    p.add_argument("--band-kp-mult", type=float, default=1.0,
                   help="Multiplier on the ElasticBand's kp_pos / kd_pos. "
                        "1.0 = current 10000 / 1000 (stiff, near critically "
                        "damped at default mass). Lower values (0.3-0.5) make "
                        "the band softer / more forgiving so the body bobs "
                        "naturally on the legs instead of being yanked rigidly "
                        "around its anchor. kd_pos is scaled by sqrt(mult) to "
                        "preserve the damping ratio.")
    p.add_argument("--nudge-force", type=float, default=0.0,
                   help="E23 disturbance probe: periodic lateral force (N) applied "
                        "at chest level to excite the waist. 0 = off (default). "
                        "Written to xfrc_applied on --nudge-body, which is a "
                        "DIFFERENT body from the elastic band's attach link, so "
                        "the two never fight.")
    p.add_argument("--nudge-body", default="torso_link",
                   help="Body the nudge is applied to (default torso_link = chest).")
    p.add_argument("--nudge-period", type=float, default=4.0,
                   help="Seconds between nudge onsets.")
    p.add_argument("--nudge-duration", type=float, default=0.15,
                   help="Seconds each nudge is held.")
    p.add_argument("--nudge-alternate", action="store_true", default=True,
                   help="Flip direction each nudge so the robot is not walked "
                        "off the mat by a one-sided push.")
    # HAND-TILT probe (2026-09-07): reproduce the operator's manual torso tilt — a torque about the
    # torso's pitch (or roll) axis ramped up slowly, held, then released quickly, repeating with
    # alternating sign. Logged joint_pos/target_pos CSVs bin exactly like the robot session
    # (bigrun_teleop_stability.md "WAIST SNAP-BACK: MECHANISM FOUND ON THE ROBOT").
    p.add_argument("--tilt-torque", type=float, default=0.0,
                   help="Peak torque (N*m) applied at --nudge-body about its pitch/roll axis. 0 = off.")
    p.add_argument("--tilt-axis", choices=("pitch", "roll"), default="pitch")
    p.add_argument("--tilt-ramp", type=float, default=3.0, help="seconds to ramp 0 -> peak (the slow push)")
    p.add_argument("--tilt-hold", type=float, default=1.5, help="seconds held at peak")
    p.add_argument("--tilt-release", type=float, default=0.1, help="seconds to ramp peak -> 0 (the hand letting go)")
    p.add_argument("--tilt-rest", type=float, default=4.0, help="seconds at zero before the next push")
    p.add_argument("--tilt-start", type=float, default=10.0, help="sim seconds before the first push (let the deploy settle)")
    p.add_argument("--tilt-alternate", action="store_true", default=True,
                   help="flip the sign every push (forward then backward lean)")
    p.add_argument("--tilt-follow-margin", type=float, default=0.0,
                   help="HAND-TILT follow mode (operator 2026-09-08: 'use what torque is being applied by the "
                        "model and just set your force a bit higher'): during ramp/hold the applied torque = "
                        "|waist actuator torque about the tilt axis| + this margin (N.m), capped at --tilt-torque, "
                        "so the torso yields at the rate the policy gives way, like a slow manual push. 0 = off.")
    # WAIST RETURN BUDGET prototype (2026-09-15, snap-back on the chores tape; SIM-ONLY diagnostic, default off).
    # Mechanism measured in a snap-back capture: with the arms out front the torso is 16-28 N*m heavy about
    # the waist pitch axis, the policy pins its target on the far rail (-25.8 deg), the joint parks on the +20 deg
    # stop with ~20 N*m of anti-gravity demand, and when the load drops the spring wins by ~10 N*m and whips the torso
    # back at 60-200 deg/s. The budget caps the waist-pitch actuator, ONLY while it works against gravity, at the
    # pose's exact gravity torque (MuJoCo subtree CoM) plus this margin, so the net restoring torque is at most the
    # margin at every arm pose. Torque toward gravity (a commanded lean) is untouched. 0 = off.
    p.add_argument("--waist-return-budget", type=float, default=0.0,
                   help="N*m margin above the pose's gravity torque allowed to the waist-pitch actuator when it opposes gravity. 0 = off.")
    p.add_argument("--waist-return-kd-mult", type=float, default=1.0,
                   help="extra damping on waist pitch while the joint moves AGAINST gravity (the return): kd * (mult-1) * dq. 1 = off.")
    # --- hip target rate gate prototype (sim-only, 2026-09-16, operator: "limit hip velocities when the waist is close to the
    # forward 20 deg limit"). The whip is triggered by hip-pitch TARGET jumps (> 5 deg/tick) while the waist sits near the stop.
    # While q_waist_pitch > --hip-gate-waist-deg, the hip-pitch targets may move at most --hip-rate-gate-deg-s; below the
    # threshold the targets pass through untouched. 0 = off.
    p.add_argument("--hip-rate-gate-deg-s", type=float, default=0.0, help="max hip-pitch target rate (deg/s) while the waist is past the gate angle. 0 = off.")
    p.add_argument("--hip-gate-waist-deg", type=float, default=16.0, help="waist pitch (deg) above which the hip target rate gate is active.")
    # --- waist-pitch target RATE limit prototype (sim-only, 2026-09-17, operator: "nothing limits how fast a waist target may move -
    # then let's put a limit to it"). The waist-pitch target may move at most --waist-rate-limit-deg-s, always. 0 = off.
    p.add_argument("--waist-rate-limit-deg-s", type=float, default=0.0, help="max waist-pitch target rate (deg/s), sim-only prototype. 0 = off.")
    p.add_argument("--tau-log", type=pathlib.Path, default=None,
                   help="CSV (container path) of applied waist/hip torques + gravity torque at 50 Hz, for the snap-back analysis.")
    p.add_argument("--viewer", action="store_true",
                   help="Open the MuJoCo passive viewer window.")
    p.add_argument("--cam-track-body", type=str, default=None,
                   help="Body name to follow with the passive viewer's camera "
                        "(e.g. 'pelvis'). Default: free camera (no tracking). "
                        "Combine with --cam-distance/elevation/azimuth to frame.")
    p.add_argument("--cam-distance", type=float, default=3.5,
                   help="Tracking-camera distance (m) from the tracked body. "
                        "Default 3.5; ignored unless --cam-track-body is set.")
    p.add_argument("--cam-elevation", type=float, default=-12.0,
                   help="Tracking-camera elevation (deg, negative = looking down). "
                        "Default -12; ignored unless --cam-track-body is set.")
    p.add_argument("--cam-azimuth", type=float, default=135.0,
                   help="Tracking-camera azimuth (deg, 0=+X, 90=+Y, 180=-X, 270=-Y). "
                        "Default 135 (3/4 view from front-right). Ignored unless "
                        "--cam-track-body is set.")
    p.add_argument("--sim-dt", type=float, default=0.001,
                   help="Physics step (s). Default 0.001 (1 kHz).")
    p.add_argument("--state-rate-hz", type=float, default=200.0,
                   help="Joint-state publish rate (default 200 Hz, matches firmware).")
    p.add_argument("--imu-rate-hz", type=float, default=500.0,
                   help="IMU publish rate (default 500 Hz, matches firmware).")
    p.add_argument("--imu-from", choices=("pelvis", "torso"), default="pelvis",
                   help="Which body to read IMU quantities from. 'pelvis' matches "
                        "what training reads off `imu_0`; 'torso' uses torso_link "
                        "(closer to the real sensor mounting, but introduces a "
                        "kinematic offset the policy never saw).")
    p.add_argument("--fixed-base", action="store_true",
                   help="Pin the floating base (skips gravity / falls). Useful for "
                        "verifying joint command paths without dynamics. Note: many "
                        "X2 MJCFs ship with a floating base joint and pinning at "
                        "runtime is not supported -- this flag will warn if so.")
    p.add_argument("--hold-stiffness-mult", type=float, default=1.0,
                   help="Multiplier on policy_parameters.kps used to hold the default "
                        "pose BEFORE the first deploy command arrives. 1.0 = same as "
                        "policy gains; >1 makes the standby hold stiffer.")
    p.add_argument("--no-elastic-band", action="store_true",
                   help="Disable the virtual ElasticBand (hangs the robot's pelvis "
                        "from world [0,0,1] with stiff PD). The band is on by default "
                        "so the robot stays upright while the deploy warms up; press "
                        "key 9 in the viewer to drop, or use --band-release-after-s.")
    p.add_argument("--band-release-after-s", type=float, default=1.0,
                   help="When the band is on AND we're headless (no viewer to press "
                        "key 9), auto-disable the band this many seconds AFTER the "
                        "first deploy command arrives. Default 1.0s -- gives the "
                        "deploy's PD/control loop one tick to settle before gravity "
                        "kicks in. Use a negative value to keep the band on forever "
                        "(unsafe headless; the robot will never touch the ground).")
    p.add_argument("--no-localhost-only", action="store_true",
                   help="Don't force ROS_LOCALHOST_ONLY=1. Use only if you intentionally "
                        "want the bridge to be reachable across the SDK ethernet.")
    p.add_argument("--ros-domain-id", type=int, default=73,
                   help="ROS_DOMAIN_ID to use unless already set in env (default 73).")
    p.add_argument("--print-scene", action="store_true",
                   help="Dump the MJCF body/joint/actuator/sensor table on startup.")
    p.add_argument("--verbose", action="store_true", help="Per-tick debug logging.")
    return p


class X2MujocoRosBridge:
    """Owns the MuJoCo model + a single rclpy node + a sim thread."""

    def __init__(self, args: argparse.Namespace):
        # Lazy / late-bound imports: keep --help cheap, and let the user see
        # a clean error if rclpy or the MuJoCo wheels aren't installed yet.
        global mujoco, mujoco_viewer, rclpy, JointStateArray, JointCommandArray, ImuMsg
        import mujoco  # noqa: F811
        import mujoco.viewer as mujoco_viewer  # noqa: F811
        import rclpy  # noqa: F811
        from rclpy.qos import qos_profile_sensor_data  # noqa: F401  (used in build)
        from aimdk_msgs.msg import JointCommandArray, JointStateArray  # noqa: F811
        from sensor_msgs.msg import Imu as ImuMsg  # noqa: F811

        self.args = args
        self.eval_x2 = _load_eval_x2()

        # --- DDS env isolation (do this BEFORE rclpy.init in main) ---
        # main() handles the env vars; we just stash them for the banner.
        self._ros_localhost_only = os.environ.get("ROS_LOCALHOST_ONLY", "0")
        self._ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")

        # --- MuJoCo ---
        # Two paths:
        #   * default (--mjcf or eval_x2.MJCF_PATH): bare X2 (dummy wrist
        #     stubs, no OmniHand) -- the canonical 31-DoF model the
        #     SONIC ONNX was trained against.
        #   * --with-omnihand: programmatic compose via
        #     gear_sonic.scripts.compose_x2_with_omnihand. Adds 10
        #     active + 4 passive finger DOFs per side. The body joint
        #     names + actuators are unchanged so the deploy harness's
        #     joint-name handshake still passes.
        if args.with_omnihand and args.mjcf is None:
            # In-memory compose: bare canonical X2 + OmniHand, no scene.
            # Lazy import: compose_x2_with_omnihand pulls additional
            # urdfpy / mesh-IO machinery that we don't need on the
            # default code path.
            sys.path.insert(0, str(REPO_ROOT))  # so `gear_sonic.*` resolves
            from gear_sonic.scripts.compose_x2_with_omnihand import (
                build_x2_with_omnihand_spec,
            )

            spec, model, hand_layout = build_x2_with_omnihand_spec()
            self.mj_model = model
            self.mjcf_path = pathlib.Path("<composed: x2_ultra + omnihand>")
            self._omnihand_layout = hand_layout
        elif args.mjcf is not None:
            mjcf_path = args.mjcf
            if not mjcf_path.is_file():
                raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
            self.mjcf_path = mjcf_path
            self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            # If the loaded MJCF already contains the canonical OmniHand
            # active joints (it does when --mjcf points at a robocasa
            # scene XML built by build_x2_robocasa_scene_xml.py), build
            # the hand layout from the COMPILED model so OmniHand qpos
            # ZMQ commands still flow through. The layout helper resolves
            # everything by joint name -- it doesn't care that the bodies
            # came from disk vs. an in-memory MjSpec.attach.
            self._omnihand_layout = None
            if args.with_omnihand:
                sys.path.insert(0, str(REPO_ROOT))
                from gear_sonic.scripts.compose_x2_with_omnihand import (
                    _build_layout,
                    _default_side_configs,
                )
                try:
                    self._omnihand_layout = _build_layout(
                        self.mj_model, _default_side_configs()
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"--with-omnihand was set but {mjcf_path} does not "
                        f"contain the canonical OmniHand joints: {exc}"
                    ) from exc
        else:
            mjcf_path = pathlib.Path(self.eval_x2.MJCF_PATH)
            if not mjcf_path.is_file():
                raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
            self.mjcf_path = mjcf_path
            self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
            self._omnihand_layout = None

        # Auto-detect a scene metadata sidecar so the bridge knows which
        # freejoints / welded bodies to publish + accept resets for. The
        # sidecar lives next to the MJCF with a .json suffix (the build
        # script's default layout). Override via --scene-metadata.
        self._scene_metadata: dict | None = None
        self._scene_freejoint_qadr: dict[str, int] = {}
        self._scene_freejoint_dofadr: dict[str, int] = {}
        self._scene_welded_bid: dict[str, int] = {}
        # Per-object contact geom IDs (filled in below from the metadata's
        # object_contact_geoms table). Used by ``_collect_grasp_contacts``
        # to classify each ``mj_data.contact`` entry.
        self._scene_object_geom_ids: dict[str, set[int]] = {}
        # Per-side hand geom ID sets, built by walking from the wrist root
        # body listed in the metadata. Used to classify a contact entry as
        # left-hand vs right-hand vs neither.
        self._scene_hand_geom_ids: dict[str, set[int]] = {"left": set(), "right": set()}
        # Per-side fingertip body IDs (the *_dip distal phalanges). Their
        # world positions feed the shaped-reward "approach" phase.
        self._scene_fingertip_bids: dict[str, list[int]] = {"left": [], "right": []}
        meta_path: pathlib.Path | None = None
        if args.scene_metadata is not None:
            meta_path = args.scene_metadata
        elif args.mjcf is not None:
            cand = args.mjcf.with_suffix(".json")
            if cand.is_file():
                meta_path = cand
        if meta_path is not None and meta_path.is_file():
            import json
            try:
                self._scene_metadata = json.loads(meta_path.read_text())
                print(f"[bridge] loaded scene metadata: {meta_path}")
            except Exception as exc:
                print(f"[bridge] WARNING: could not parse scene metadata "
                      f"{meta_path}: {exc}", file=sys.stderr)
                self._scene_metadata = None
        if self._scene_metadata is not None:
            for jname in self._scene_metadata.get(
                "object_freejoint_map", {}
            ).values():
                jid = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname
                )
                if jid < 0:
                    print(f"[bridge] WARNING: scene freejoint {jname!r} "
                          f"not found in MJCF; skipping.", file=sys.stderr)
                    continue
                self._scene_freejoint_qadr[jname] = int(
                    self.mj_model.jnt_qposadr[jid]
                )
                self._scene_freejoint_dofadr[jname] = int(
                    self.mj_model.jnt_dofadr[jid]
                )
            for bname in self._scene_metadata.get(
                "object_welded_map", {}
            ).values():
                bid = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bname
                )
                if bid < 0:
                    print(f"[bridge] WARNING: scene welded body {bname!r} "
                          f"not found in MJCF; skipping.", file=sys.stderr)
                    continue
                self._scene_welded_bid[bname] = int(bid)

            # Resolve per-object contact-geom IDs.
            for logical, geom_names in self._scene_metadata.get(
                "object_contact_geoms", {}
            ).items():
                ids: set[int] = set()
                for gname in geom_names:
                    gid = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname
                    )
                    if gid < 0:
                        print(
                            f"[bridge] WARNING: scene contact geom {gname!r} "
                            f"(logical={logical!r}) not in MJCF; skipping.",
                            file=sys.stderr,
                        )
                        continue
                    ids.add(int(gid))
                if ids:
                    self._scene_object_geom_ids[logical] = ids

            # Resolve per-side hand geom IDs by walking the body subtree
            # rooted at each wrist_roll_link. This catches every fingertip
            # cylinder + thumb collision box without having to enumerate
            # them in the metadata.
            for side, root_bname in self._scene_metadata.get(
                "hand_root_bodies", {}
            ).items():
                root_bid = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, root_bname
                )
                if root_bid < 0:
                    print(
                        f"[bridge] WARNING: hand root {root_bname!r} not "
                        f"in MJCF; side {side!r} contact reporting disabled.",
                        file=sys.stderr,
                    )
                    continue
                # Walk every body whose ancestor chain passes through
                # root_bid; for each such body, collect its geom IDs.
                # MuJoCo's body_parentid lets us check ancestry in O(depth).
                hand_gids: set[int] = set()
                for bid in range(self.mj_model.nbody):
                    cur = int(bid)
                    while cur > 0:
                        if cur == root_bid:
                            geomadr = int(self.mj_model.body_geomadr[bid])
                            geomnum = int(self.mj_model.body_geomnum[bid])
                            for k in range(geomnum):
                                hand_gids.add(geomadr + k)
                            break
                        cur = int(self.mj_model.body_parentid[cur])
                self._scene_hand_geom_ids[side] = hand_gids

            # Resolve per-side fingertip body IDs.
            for side, body_names in self._scene_metadata.get(
                "fingertip_bodies", {}
            ).items():
                tip_bids: list[int] = []
                for bname in body_names:
                    bid = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bname
                    )
                    if bid < 0:
                        print(
                            f"[bridge] WARNING: fingertip body {bname!r} "
                            f"not in MJCF; skipping (side={side}).",
                            file=sys.stderr,
                        )
                        continue
                    tip_bids.append(int(bid))
                self._scene_fingertip_bids[side] = tip_bids

            print(
                f"[bridge] scene plumbing: "
                f"{len(self._scene_freejoint_qadr)} freejoints, "
                f"{len(self._scene_welded_bid)} welded bodies, "
                f"{sum(len(v) for v in self._scene_object_geom_ids.values())} "
                f"object collision geoms, "
                f"L:{len(self._scene_hand_geom_ids.get('left', set()))} "
                f"R:{len(self._scene_hand_geom_ids.get('right', set()))} "
                f"hand geoms, "
                f"L:{len(self._scene_fingertip_bids.get('left', []))} "
                f"R:{len(self._scene_fingertip_bids.get('right', []))} "
                f"fingertips"
            )

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = float(args.sim_dt)

        # When the OmniHand-augmented MJCF is loaded, park active hand
        # qpos AND ctrl at the canonical open-hand rest pose *before*
        # the first ``mj_step``. Without this, both default to 0, which
        # (a) lands several active joints on a joint-limit edge, and
        # (b) makes the position actuators -- ``inheritrange=1`` so the
        # ctrl envelope matches joint range -- perpetually drive
        # toward qpos=0. The combined effect is a slamming motion vs.
        # the joint stops the moment dynamics start, observed by the
        # operator as random finger jitter even when no VR commands
        # are flowing. The retargeter's ``ratio=0`` open-hand pose is
        # the natural target for a "rest" / "no input yet" state.
        if self._omnihand_layout is not None:
            from gear_sonic.scripts.compose_x2_with_omnihand import (
                apply_active_hand_rest_pose,
            )
            apply_active_hand_rest_pose(
                self.mj_model, self.mj_data, self._omnihand_layout
            )

        # Resolve per-joint qpos / qvel / actuator indices from joint NAMES,
        # so we don't have to trust the MJCF actuator order. Falls back to
        # eval_x2's JOINT_TO_ACTUATOR if name lookup fails (older MJCFs).
        self.qpos_adr = np.zeros(self.eval_x2.NUM_DOFS, dtype=np.int64)
        self.qvel_adr = np.zeros(self.eval_x2.NUM_DOFS, dtype=np.int64)
        self.act_idx  = np.zeros(self.eval_x2.NUM_DOFS, dtype=np.int64)
        for i, jname in enumerate(self.eval_x2.MUJOCO_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"MJCF missing joint '{jname}'")
            self.qpos_adr[i] = self.mj_model.jnt_qposadr[jid]
            self.qvel_adr[i] = self.mj_model.jnt_dofadr[jid]
            aname = f"motor_{jname}"
            aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self.act_idx[i] = aid
            else:
                # Fall back to eval_x2's hand-rolled map (warn once).
                self.act_idx[i] = self.eval_x2.JOINT_TO_ACTUATOR[i]

        # Effort limits from the MJCF actuator force range (per actuator).
        # Use the wider of |min|/|max| in case it's asymmetric.
        force_range = self.mj_model.actuator_forcerange[self.act_idx]
        self.tau_limit = np.maximum(np.abs(force_range[:, 0]), np.abs(force_range[:, 1]))
        # Some MJCFs publish 0/0 if the actuator has no force range; guard.
        self.tau_limit = np.where(self.tau_limit > 0.0, self.tau_limit, 1e6)

        # --- waist return budget prototype (see --waist-return-budget) ---
        self._wrb_delta = float(getattr(args, "waist_return_budget", 0.0) or 0.0)
        self._wrb_kd_mult = float(getattr(args, "waist_return_kd_mult", 1.0) or 1.0)
        self._wrb_on = self._wrb_delta > 0.0 or self._wrb_kd_mult > 1.0
        self._tau_log_path = getattr(args, "tau_log", None)
        self._wrb_i = list(self.eval_x2.MUJOCO_JOINT_NAMES).index("waist_pitch_joint")
        _wjid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
        self._wrb_jid = _wjid
        self._wrb_body = int(self.mj_model.jnt_bodyid[_wjid])
        self._wrb_mass = float(self.mj_model.body_subtreemass[self._wrb_body])
        self._wrb_hip = [list(self.eval_x2.MUJOCO_JOINT_NAMES).index(n) for n in ("left_hip_pitch_joint", "right_hip_pitch_joint")]
        self._hg_rate = math.radians(float(getattr(args, "hip_rate_gate_deg_s", 0.0) or 0.0)) / 1000.0   # rad per 1 kHz PD step
        self._hg_thr = math.radians(float(getattr(args, "hip_gate_waist_deg", 16.0)))
        self._hg_last = None; self._hg_gated = 0
        self._wl_rate = math.radians(float(getattr(args, "waist_rate_limit_deg_s", 0.0) or 0.0)) / 1000.0   # rad per 1 kHz PD step
        self._wl_last = None; self._wl_limited = 0
        if self._wl_rate > 0: print("[bridge] waist target rate limit ON: %.0f deg/s" % getattr(args, "waist_rate_limit_deg_s"), flush=True)
        if self._hg_rate > 0: print("[bridge] hip target rate gate ON: %.0f deg/s while waist > %.0f deg" % (getattr(args, "hip_rate_gate_deg_s"), getattr(args, "hip_gate_waist_deg")), flush=True)
        self._wrb_stats = dict(ticks=0, capped=0, max_cut=0.0, kd_ticks=0)
        self._tau_log_f = None; self._tau_log_n = 0
        if self._tau_log_path is not None:
            self._tau_log_f = open(str(self._tau_log_path), "w")
            self._tau_log_f.write("t,tau_waist_p,tau_waist_p_raw,tau_g_waist,capped,tau_hip_l,tau_hip_r,q_waist_p,dq_waist_p\n")
        if self._wrb_on:
            print(f"[bridge] WAIST RETURN BUDGET prototype ON: margin {self._wrb_delta:.1f} N*m above gravity, "
                  f"return kd mult {self._wrb_kd_mult:.1f}; torso subtree mass {self._wrb_mass:.1f} kg", flush=True)

        # --- Floating base detection (matches base_sim.py logic) ---
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self.has_floating_base = any(
            self.mj_model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE
            for i in range(self.mj_model.njnt)
        )
        if args.fixed_base and self.has_floating_base:
            print("[bridge] WARNING: --fixed-base requested but MJCF has a free joint; "
                  "ignoring (would need MJCF surgery to honour).", file=sys.stderr)

        # Resolve the body that owns the free joint (i.e. the floating base).
        # We need this to decide whether ``_publish_imu`` can short-circuit
        # to ``qvel[3:6]`` for bit-exact ang-vel parity with Python eval --
        # see the comment in ``_publish_imu``.
        self._free_base_body_id = -1
        if self.has_floating_base:
            for jid in range(self.mj_model.njnt):
                if self.mj_model.joint(jid).type == mujoco.mjtJoint.mjJNT_FREE:
                    self._free_base_body_id = int(self.mj_model.jnt_bodyid[jid])
                    break

        # --- IMU body ---
        if args.imu_from == "pelvis":
            self.imu_body = "pelvis"
        else:
            self.imu_body = "torso_link"
        try:
            self.imu_body_id = self.mj_model.body(self.imu_body).id
        except KeyError as e:
            raise RuntimeError(f"MJCF missing body '{self.imu_body}' for IMU readout") from e

        # --- Standby PD setpoint table (per joint, MJ order) ---
        # Before any deploy command arrives, hold the default standing pose
        # with policy_parameters' kps/kds * hold_stiffness_mult so the robot
        # doesn't immediately crumple under gravity.
        n = self.eval_x2.NUM_DOFS
        self.cmd_lock = threading.Lock()
        self._target_pos = self.eval_x2.DEFAULT_DOF.astype(np.float64).copy()
        self._target_vel = np.zeros(n, dtype=np.float64)
        self._effort_ff  = np.zeros(n, dtype=np.float64)
        self._kp = self.eval_x2.KP.astype(np.float64).copy() * float(args.hold_stiffness_mult)
        self._kd = self.eval_x2.KD.astype(np.float64).copy() * float(args.hold_stiffness_mult)
        self._first_command_received = False
        self._first_command_mono = 0.0  # set on first deploy command arrival

        # --- ElasticBand (G1-style "hang the robot" suspension) ---
        # Without this the robot tips over while the deploy spins up its
        # control loop and the C++ tilt watchdog instantly trips.
        # Disabled when (a) user passed --no-elastic-band or (b) the MJCF has
        # no floating base to push against (forces would be silently lost).
        self.elastic_band: Optional[ElasticBand] = None
        self.band_attached_link: int = -1
        if not args.no_elastic_band and self.has_floating_base:
            self.elastic_band = ElasticBand()
            # Initial value from the CLI; ``_reset_to_named_pose`` may
            # override this from the YAML entry's ``band_length`` if the
            # operator left ``--band-length`` at its default (0). With
            # length=0 and a low pelvis (e.g. z=0.6), the band would yank
            # the body upward with hundreds of N -- exactly the surprise
            # the YAML override engineers around.
            self.elastic_band.length = float(args.band_length)
            if args.band_kp_mult != 1.0:
                self.elastic_band.kp_pos *= float(args.band_kp_mult)
                # Scale damping by sqrt(mult) to preserve the damping ratio
                # ``zeta = kd / (2 * sqrt(kp * m))`` as kp scales.
                self.elastic_band.kd_pos *= float(np.sqrt(args.band_kp_mult))
            # X2's pelvis is the natural attach point (matches base_sim.py's
            # ``elif "g1" ... waist`` branch -- in our MJCF the waist body IS
            # the pelvis-mounted free body).
            try:
                self.band_attached_link = int(self.mj_model.body("pelvis").id)
            except KeyError as e:
                raise RuntimeError(
                    "ElasticBand: MJCF has no body named 'pelvis' -- pass "
                    "--no-elastic-band or extend this code to pick another."
                ) from e
        elif not args.no_elastic_band and not self.has_floating_base:
            print("[bridge] ElasticBand requested but MJCF has no floating "
                  "base; band disabled.", file=sys.stderr)

        # --- ROS 2 ---
        self.node = rclpy.create_node("x2_mujoco_ros_bridge")
        from rclpy.qos import qos_profile_sensor_data  # local; module-level alias may be missing
        self.qos = qos_profile_sensor_data

        # Pre-build per-group joint name lists (avoid per-publish python work).
        self._group_names = []
        for grp_name, topic_prefix, start, length in GROUPS:
            names = list(self.eval_x2.MUJOCO_JOINT_NAMES[start : start + length])
            self._group_names.append((grp_name, topic_prefix, start, length, names))

        # Joint-state publishers
        self._state_pubs = []
        for grp_name, topic_prefix, _start, _length, _names in self._group_names:
            self._state_pubs.append(
                self.node.create_publisher(JointStateArray, f"{topic_prefix}/state", self.qos)
            )

        # IMU publisher
        self._imu_pub = self.node.create_publisher(
            ImuMsg, "/aima/hal/imu/torso/state", self.qos
        )

        # Joint-command subscribers (one per group)
        self._cmd_subs = []
        for grp_name, topic_prefix, start, length, names in self._group_names:
            cb = self._make_cmd_callback(grp_name, start, length, names)
            self._cmd_subs.append(
                self.node.create_subscription(
                    JointCommandArray, f"{topic_prefix}/command", cb, self.qos
                )
            )

        # --- Initial pose ---
        # Hyphens and underscores are equivalent (so --init-pose=gantry-hang
        # and --init-pose=gantry_hang both resolve the same way).
        normalized_init_pose = args.init_pose.replace("-", "_")
        if args.motion and normalized_init_pose != "default":
            raise ValueError(
                f"Cannot combine --motion (RSI from motion frame) with "
                f"--init-pose={args.init_pose!r}. RSI puts the robot at the "
                "motion's frame 0 pose; named init poses put it in a fixed "
                "configuration. Pick one. (For sim profiles built around a "
                "named pose, no --motion is needed; the deploy node still "
                "loads its --motion arg as the *reference* the policy "
                "tracks; the bridge's init pose is independent.)"
            )
        if args.motion:
            self._rsi_from_motion(args.motion, args.init_frame)
        elif normalized_init_pose == "default":
            self._reset_to_default_pose()
        else:
            self._reset_to_named_pose(normalized_init_pose)

        if args.print_scene:
            self._print_scene()

        # --- Counters ---
        self._sim_step_count = 0
        # Real-time-factor telemetry: the pacing loops silently reset their
        # cadence when a 1 kHz tick overruns ("Falling behind" branch), so a
        # slow machine plays the whole sim uniformly slower than wall time
        # with NO log evidence -- clips then look damped/sluggish in the
        # viewer while the wall-clocked planner reference races ahead
        # (2026-08-10 operator report: dances AND boxing felt slow in the
        # stack, LPFs were off). Report achieved RTF every 5 sim-seconds.
        self._rtf_mark_step = 0
        self._rtf_mark_wall = None
        self._cmd_count = 0
        self._state_pub_count = 0
        self._imu_pub_count = 0

        self._stop = threading.Event()
        self.viewer = None

        # --- OmniHand ZMQ subscriber state (filled in run() if enabled) ---
        # ``_hand_qpos_targets`` is the most recent (10-D-per-side)
        # active-finger setpoint vector; ``_apply_omnihand_qpos`` writes
        # it into ``mj_data.qpos`` every sim step. Held under a lock so
        # the SUB thread and the sim thread can race without tearing.
        self._hand_lock = threading.Lock()
        # Seed the silent-stream cache with the canonical open-hand
        # rest pose (matches what we just wrote into ``mj_data.qpos``
        # / ``mj_data.ctrl`` above). With this seed, the recurring
        # ``_apply_omnihand_qpos`` -> ``apply_active_hand_ctrl`` write
        # at sim cadence keeps the actuator targets parked at rest
        # until a real ZMQ packet arrives -- not at qpos=0, which
        # would otherwise unwind everything we did at init.
        if self._omnihand_layout is not None:
            from gear_sonic.scripts.compose_x2_with_omnihand import (
                get_active_hand_rest_pose,
            )
            self._hand_left_active = get_active_hand_rest_pose("left")
            self._hand_right_active = get_active_hand_rest_pose("right")
        else:
            self._hand_left_active = np.zeros(10, dtype=np.float64)
            self._hand_right_active = np.zeros(10, dtype=np.float64)
        self._hand_first_msg_logged = False
        self._hand_msg_count = 0

        # OmniHand diagnostic: every 5 s, log (a) the per-finger ctrl
        # setpoint we just wrote and (b) the resulting qpos after the
        # most recent ``mj_step``. Used to diagnose "fingers visually
        # don't close in robocasa scenes but do close in bare scenes"
        # bugs by separating recorder/bridge issues from MuJoCo
        # physics issues:
        #   * if target == qpos at all steady-state values → bridge
        #     and physics agree, so any visual mismatch is a render
        #     or perception issue.
        #   * if target moves (e.g. up to 1.54) but qpos stays low
        #     (e.g. ~0.1) → the integrator is unable to reach the
        #     target. Most common cause is contact (hand fingertips
        #     touching the table/cube at episode start so the PD
        #     can't push past), but it could also be solver clamping
        #     or a kp/forcerange mismatch in the merged scene.
        #   * if target stays at the rest pose (~0.09) regardless of
        #     ZMQ input → the SUB thread never updated
        #     ``self._hand_left_active`` / ``_hand_right_active``,
        #     i.e. the recorder isn't pushing or the message format
        #     mismatches.
        self._hand_diag_last_log_t: float = 0.0
        # Throttle in seconds; keep aligned with the recorder's
        # status_log_period_s so the two log streams interleave at
        # the same cadence and operators can read them as pairs.
        self._hand_diag_period_s: float = 5.0

        # --- Scene reset queue (set by _scene_reset_zmq_thread, drained
        #     by the sim thread). The SUB thread previously wrote
        #     ``mj_data.qpos`` / ``mj_model.body_pos`` directly without
        #     holding ``viewer.lock()``; that races the sim thread's
        #     mid-step constraint solver and silently corrupts state
        #     once the scene has built up enough contacts (typical
        #     symptom: the second B press on a long session leaves the
        #     viewer rendering a frozen / NaN'd pose while the deploy
        #     keeps publishing CONTROL ticks for ~20 s before the
        #     operator notices and Ctrl-Cs). The queue + sim-thread
        #     drain pattern mirrors the OmniHand SUB and applies the
        #     reset at a known-safe sim-cycle boundary instead.
        #     ``_pending_scene_reset`` holds the latest queued payload
        #     (later messages overwrite earlier unprocessed ones --
        #     drop-old semantics, identical to ZMQ_CONFLATE on the SUB
        #     side); ``_pending_scene_reset_lock`` serialises the
        #     stash + drain.
        self._pending_scene_reset_lock = threading.Lock()
        self._pending_scene_reset: Optional[dict] = None
        self._pending_scene_reset_count = 0

    # ----------------------------------------------------------------
    # Command ingestion
    # ----------------------------------------------------------------
    def _make_cmd_callback(self, grp_name, start, length, expected_names):
        """Return a closure that ingests one group's JointCommandArray."""
        validated = [False]
        def cb(msg):
            if not validated[0]:
                if len(msg.joints) != length:
                    self.node.get_logger().fatal(
                        f"[bridge:{grp_name}] command has {len(msg.joints)} "
                        f"joints, expected {length}; refusing to ingest."
                    )
                    return
                for i, j in enumerate(msg.joints):
                    if j.name and j.name != expected_names[i]:
                        self.node.get_logger().fatal(
                            f"[bridge:{grp_name}] command slot {i} is "
                            f"'{j.name}', expected '{expected_names[i]}'."
                        )
                        return
                validated[0] = True
                self.node.get_logger().info(
                    f"[bridge:{grp_name}] joint name validation OK "
                    f"({length} joints)."
                )
            with self.cmd_lock:
                for i, j in enumerate(msg.joints):
                    mj = start + i
                    self._target_pos[mj] = float(j.position)
                    self._target_vel[mj] = float(j.velocity)
                    self._effort_ff[mj]  = float(j.effort)
                    self._kp[mj]         = float(j.stiffness)
                    self._kd[mj]         = float(j.damping)
                if not self._first_command_received:
                    self._first_command_received = True
                    self._first_command_mono = time.monotonic()
                    self.node.get_logger().info(
                        f"[bridge] first deploy command received on group '{grp_name}' "
                        "-- handing PD over to the deploy."
                    )
                self._cmd_count += 1
        return cb

    # ----------------------------------------------------------------
    # State publication
    # ----------------------------------------------------------------
    def _publish_joint_states(self):
        from aimdk_msgs.msg import JointState, JointStateArray
        now = self.node.get_clock().now().to_msg()
        for (grp_name, topic_prefix, start, length, names), pub in zip(
            self._group_names, self._state_pubs
        ):
            msg = JointStateArray()
            msg.header.stamp = now
            msg.header.frame_id = grp_name
            msg.header.sequence = self._state_pub_count & 0xFFFFFFFF
            msg.header.meas_stamp = now
            for i in range(length):
                mj = start + i
                js = JointState()
                js.name = names[i]
                js.position = float(self.mj_data.qpos[self.qpos_adr[mj]])
                js.velocity = float(self.mj_data.qvel[self.qvel_adr[mj]])
                js.effort = float(self.mj_data.actuator_force[self.act_idx[mj]])
                js.coil_temp = 0
                js.motor_temp = 0
                js.motor_vol = 0
                msg.joints.append(js)
            pub.publish(msg)
        self._state_pub_count += 1

    def _publish_imu(self):
        body = self.mj_data.body(self.imu_body_id)
        # MuJoCo body xquat is wxyz; sensor_msgs/Imu wants xyzw
        wxyz = body.xquat
        msg = ImuMsg()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.imu_body
        msg.orientation.w = float(wxyz[0])
        msg.orientation.x = float(wxyz[1])
        msg.orientation.y = float(wxyz[2])
        msg.orientation.z = float(wxyz[3])
        # ---- Angular velocity (body-local frame) ----------------------------
        # Parity gotcha: ``mj_objectVelocity(..., flg_local=1)`` for a body
        # returns the angular component of ``mj_data.cvel``, which is the
        # spatial velocity computed at the body's CoM expressed in body
        # frame. After a manual ``qvel`` write + ``mj_forward`` it does NOT
        # round-trip exactly to ``qvel[3:6]`` -- we observed a 0.02 rad/s
        # drift on the parity gate vs ``eval_x2_mujoco{,_onnx}.py``, and the
        # policy reacts visibly to a 0.02 rad/s OOD ang-vel input. Both the
        # eval scripts AND IsaacLab's training observation read
        # ``qvel[3:6]`` directly (it is the BODY-LOCAL angular velocity of
        # the free-joint root body by MuJoCo convention -- see Gotcha G1 in
        # docs/source/user_guide/sim2sim_mujoco.md). To stay bit-exact with
        # Python eval and the trained policy, publish ``qvel[3:6]`` instead
        # of ``mj_objectVelocity`` when the IMU is on the floating-base
        # body (the "pelvis" default). For ``--imu-from torso`` we don't
        # have a clean qvel slice -- the policy isn't trained on a torso
        # IMU anyway, so the legacy mj_objectVelocity path is fine there.
        if self.imu_body_id == self._free_base_body_id:
            qvel_ang = self.mj_data.qvel[3:6]
            msg.angular_velocity.x = float(qvel_ang[0])
            msg.angular_velocity.y = float(qvel_ang[1])
            msg.angular_velocity.z = float(qvel_ang[2])
        else:
            ang = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                self.imu_body_id, ang, 1,  # 1 = body local frame
            )
            msg.angular_velocity.x = float(ang[0])
            msg.angular_velocity.y = float(ang[1])
            msg.angular_velocity.z = float(ang[2])
        # Linear acceleration: derive from qacc projected to body frame would be
        # ideal; for now we send root linear acceleration in world frame which
        # is what an idealised IMU after gravity removal would not match. The
        # deploy doesn't read linear_acceleration today, so leave at zero (the
        # field is required to be present but the deploy doesn't consume it).
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 0.0
        # Covariances unknown -> set first element to -1.0 per REP-145 to mark
        # the orientation/ang vel/lin accel covariance as unknown.
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance[0] = 0.0
        msg.linear_acceleration_covariance[0] = -1.0
        self._imu_pub.publish(msg)
        self._imu_pub_count += 1

    # ----------------------------------------------------------------
    # PD step
    # ----------------------------------------------------------------
    def _apply_pd(self):
        # Snapshot setpoints under the lock; release before the np math.
        with self.cmd_lock:
            target_pos = self._target_pos.copy()
            target_vel = self._target_vel.copy()
            effort_ff  = self._effort_ff.copy()
            kp         = self._kp.copy()
            kd         = self._kd.copy()

        q  = self.mj_data.qpos[self.qpos_adr]
        dq = self.mj_data.qvel[self.qvel_adr]
        # --- hip target rate gate prototype (sim-only; see --hip-rate-gate-deg-s) ---
        if self._hg_rate > 0:
            if self._hg_last is None: self._hg_last = target_pos[self._wrb_hip].copy()
            if q[self._wrb_i] > self._hg_thr:
                step = np.clip(target_pos[self._wrb_hip] - self._hg_last, -self._hg_rate, self._hg_rate)
                target_pos[self._wrb_hip] = self._hg_last + step; self._hg_gated += 1
            self._hg_last = target_pos[self._wrb_hip].copy()
        # --- waist-pitch target rate limit prototype (sim-only; see --waist-rate-limit-deg-s) ---
        if self._wl_rate > 0:
            i = self._wrb_i
            if self._wl_last is None: self._wl_last = float(target_pos[i])
            step = float(np.clip(target_pos[i] - self._wl_last, -self._wl_rate, self._wl_rate))
            if abs(target_pos[i] - self._wl_last) > self._wl_rate: self._wl_limited += 1
            target_pos[i] = self._wl_last + step; self._wl_last = float(target_pos[i])
        tau = effort_ff + kp * (target_pos - q) + kd * (target_vel - dq)
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        # --- waist return budget prototype (sim-only; see --waist-return-budget) ---
        if self._wrb_on or self._tau_log_f is not None:
            i = self._wrb_i
            # gravity generalized force along +q of waist pitch: axis . ((subtree_com - anchor) x (0,0,-m g))
            r = self.mj_data.subtree_com[self._wrb_body] - self.mj_data.xanchor[self._wrb_jid]
            F = np.array([0.0, 0.0, -9.81 * self._wrb_mass])
            tau_g = float(np.dot(self.mj_data.xaxis[self._wrb_jid], np.cross(r, F)))
            raw = float(tau[i]); ti = raw; capped = 0
            if self._wrb_kd_mult > 1.0 and (dq[i] * tau_g) < 0.0:
                # moving against gravity (the return): extra damping
                ti -= (self._wrb_kd_mult - 1.0) * float(kd[i]) * float(dq[i]); self._wrb_stats["kd_ticks"] += 1
            if self._wrb_delta > 0.0 and (ti * tau_g) <= 0.0:
                cap = abs(tau_g) + self._wrb_delta
                if abs(ti) > cap:
                    self._wrb_stats["max_cut"] = max(self._wrb_stats["max_cut"], abs(ti) - cap)
                    ti = np.sign(ti) * cap; capped = 1
            tau[i] = np.clip(ti, -self.tau_limit[i], self.tau_limit[i])
            self._wrb_stats["ticks"] += 1; self._wrb_stats["capped"] += capped
            if self._tau_log_f is not None:
                self._tau_log_n += 1
                if self._tau_log_n % 20 == 0:   # 1 kHz PD -> 50 Hz log
                    hl, hr = self._wrb_hip
                    self._tau_log_f.write("%.4f,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%.5f,%.4f\n" % (
                        self.mj_data.time, tau[i], raw, tau_g, capped, tau[hl], tau[hr], q[i], dq[i]))
                    if self._tau_log_n % 1000 == 0: self._tau_log_f.flush()
            if self._wrb_on and self._wrb_stats["ticks"] % 20000 == 0:
                s = self._wrb_stats
                print(f"[bridge] waist budget: capped {100.0 * s['capped'] / max(s['ticks'], 1):.1f}% of ticks, "
                      f"max cut {s['max_cut']:.1f} N*m, return-kd ticks {s['kd_ticks']}", flush=True)

        # Write into ctrl[] using the per-joint actuator index.
        self.mj_data.ctrl[self.act_idx] = tau

    # ----------------------------------------------------------------
    # OmniHand finger drive
    # ----------------------------------------------------------------
    def _apply_omnihand_qpos(self):
        """Write the latest hand setpoints into ``mj_data.ctrl``.

        Despite the legacy method name (kept for callsite stability),
        this drives the augmented spec's position actuators rather than
        stamping qpos directly. Position actuators implement
        ``tau = kp*(ctrl - q) - kv*dq``; the integrator handles finger
        dynamics smoothly during ``mj_step`` and the equality constraints
        enforce mimic relationships continuously. The earlier qpos-stamp
        approach used by the offline renderer
        (:func:`gear_sonic.scripts.compose_x2_with_omnihand.apply_active_hand_qpos`)
        triggered QACC NaN within 0.5 s of sim time because each tick's
        qpos jump differentiated into infinite implicit qvel that the
        equality solver couldn't reconcile. The ctrl-driven path is
        also semantically equivalent to how the real OmniHand motors
        operate (PD on torque), closing a sim-to-hardware gap.

        No-op when ``--with-omnihand`` is off (``_omnihand_layout`` is
        None) -- the bare X2 has no hand actuators to drive.
        """
        if self._omnihand_layout is None:
            return
        with self._hand_lock:
            left = self._hand_left_active
            right = self._hand_right_active
        # Lazy import to keep this module importable on the bare-X2
        # path (where compose_x2_with_omnihand is never needed).
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            apply_active_hand_ctrl,
        )
        apply_active_hand_ctrl(
            self.mj_data,
            self._omnihand_layout,
            left_active=left,
            right_active=right,
        )
        self._maybe_log_hand_diag(left=left, right=right)

    def _maybe_log_hand_diag(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
    ) -> None:
        """Throttled bridge-side diagnostic for OmniHand command tracking.

        Prints, once every ``_hand_diag_period_s``, the pair
        ``(ctrl_target, qpos_settled)`` for each of the 10 active
        fingers per side. The raw values come from the same
        ``HandQposLayout`` the writer just used, so any mismatch is
        guaranteed to be physics-side (PD couldn't reach setpoint)
        and not a layout-routing bug.

        See the constructor for the failure-mode taxonomy this diag
        is designed to discriminate.
        """
        now = time.monotonic()
        if (now - self._hand_diag_last_log_t) < self._hand_diag_period_s:
            return
        self._hand_diag_last_log_t = now

        layout = self._omnihand_layout
        if layout is None:
            return

        def _fmt_pair(target: float, settled: float) -> str:
            return f"({target:+.2f}->{settled:+.2f})"

        for side, active in (("left", left), ("right", right)):
            qadrs = layout.active_qposadr.get(side) or []
            actadrs = layout.active_actadr.get(side) or []
            if not qadrs or not actadrs:
                # Layout was built without the position actuators
                # (e.g. the offline qpos-stamp renderer path that
                # doesn't add ``pos_<joint>``). Diag is meaningless
                # there because no PD loop is running.
                continue
            # Read settled qpos (post mj_step) and the actuator's
            # last-written ctrl. We could read ``active`` too, but
            # going through ``mj_data.ctrl[actid]`` proves the value
            # the integrator actually saw -- if the bridge dropped
            # the write somewhere, the diag will surface it.
            try:
                ctrls = [float(self.mj_data.ctrl[a]) for a in actadrs]
                qpos = [float(self.mj_data.qpos[q]) for q in qadrs]
            except (IndexError, TypeError):
                # Defensive: skip the diag if anything is mid-shape-
                # change (e.g. the model just got replaced by a
                # scene_reset). The next tick will retry.
                return
            pairs = " ".join(_fmt_pair(c, q) for c, q in zip(ctrls, qpos))
            self.node.get_logger().info(
                f"[hand-bridge] {side.upper():<5s} "
                f"thumb=(roll/abad/mcp) fingers=(idx_mcp/idx_pip"
                f"/mid_mcp/mid_pip/ring_mcp/ring_pip/pinky_mcp) "
                f"target->settled: {pairs}"
            )

    def _ensure_pyzmq_in_container(self, *, purpose: str) -> bool:
        """Lazy-install ``pyzmq`` if the container's Python is missing it.

        Returns True if ``import zmq`` succeeds (either was already present
        or just installed). Pre-2026-05-10 docker_x2 base images shipped
        without pyzmq; the OmniHand SUB has had a one-shot pip recovery
        forever, but the scene_state PUB and scene_reset SUB initially
        skipped that step which left them silently dead until the next
        ``docker compose build x2sim`` rebake. This helper centralises
        the recovery so every ZMQ entry point gets the same self-heal.

        ``purpose`` is logged so the operator can tell which feature
        triggered the install attempt.
        """
        try:
            import zmq  # noqa: F401
            return True
        except ImportError:
            pass

        import subprocess  # local: only when pyzmq is missing
        self.node.get_logger().warn(
            f"[bridge] {purpose}: pyzmq not present in this container; "
            "attempting one-shot 'pip3 install pyzmq'. Rebuild the "
            "docker_x2 image (``cd gear_sonic_deploy/docker_x2 && "
            "docker compose build x2sim``) to bake this dependency in "
            "permanently."
        )
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 "--quiet", "--no-input", "--disable-pip-version-check",
                 "pyzmq"],
                timeout=60,
            )
            import zmq  # noqa: F401, F811  -- side-effect after install
            self.node.get_logger().info(
                f"[bridge] pyzmq installed at runtime; {purpose} online."
            )
            return True
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                ImportError) as exc:
            self.node.get_logger().warn(
                f"[bridge] runtime pip install pyzmq failed for {purpose} "
                f"({exc})."
            )
            return False

    def _omnihand_zmq_thread(self):
        """SUB on the live VLA bridge's pose topic and refresh hand setpoints.

        The pose message we expect (defined in
        :mod:`gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder`) carries
        ``left_hand_joints`` / ``right_hand_joints`` 10-D fields per tick at
        the publisher's rate (50 Hz for the live VLA bridge). We unpack each
        message, copy the two hand vectors under ``_hand_lock``, and let
        ``_apply_omnihand_qpos`` write them at the sim cadence. ZMQ PUB/SUB
        is multi-subscriber-safe so this can ride alongside the SONIC C++
        harness reading the same port.

        Connection setup is best-effort: if pyzmq isn't installed (the
        pre-2026-05-10 Docker base image didn't carry it) we attempt
        a one-shot ``pip3 install pyzmq`` into the running container,
        then re-import. If the auto-install fails we log once and exit
        the thread; the viewer then shows OmniHand fingers at their
        MJCF rest pose -- still better than the dummy stubs. The
        Dockerfile now bakes pyzmq in permanently so on the next
        ``docker compose build`` of ``x2sim`` this fallback becomes a
        no-op cached import.
        """
        if not self._ensure_pyzmq_in_container(
            purpose="OmniHand ZMQ subscriber"
        ):
            self.node.get_logger().warn(
                "[bridge] fingers will stay at MJCF rest pose. Install "
                "pyzmq manually in the bridge env to drive the fingers."
            )
            return
        import zmq
        # The decoder lives under gear_sonic which is already on sys.path
        # (we inserted REPO_ROOT earlier when --with-omnihand resolved
        # compose_x2_with_omnihand). Be defensive in case of an exotic
        # docker mount.
        try:
            from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
                unpack_message,
            )
        except ImportError as exc:
            self.node.get_logger().warn(
                f"[bridge] OmniHand ZMQ subscriber: cannot import "
                f"unpack_message ({exc}); fingers stay at rest pose."
            )
            return

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        # CONFLATE keeps only the latest message; we don't need a queue
        # and it bounds memory if the SUB thread ever falls behind.
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 200)  # ms: 5x our 50 Hz period
        sock.setsockopt(zmq.SUBSCRIBE, self.args.hand_zmq_topic.encode())
        endpoint = f"tcp://{self.args.hand_zmq_host}:{self.args.hand_zmq_port}"
        try:
            sock.connect(endpoint)
        except Exception as exc:
            self.node.get_logger().warn(
                f"[bridge] OmniHand ZMQ connect to {endpoint} failed: {exc}; "
                "fingers stay at rest pose."
            )
            sock.close(linger=0)
            return

        self.node.get_logger().info(
            f"[bridge] OmniHand ZMQ SUB connected at {endpoint} "
            f"(topic='{self.args.hand_zmq_topic}'); waiting for poses."
        )
        while not self._stop.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            except Exception as exc:
                self.node.get_logger().warn(
                    f"[bridge] OmniHand ZMQ recv error: {exc}"
                )
                continue
            try:
                msg = unpack_message(raw, expected_topic=self.args.hand_zmq_topic)
            except ValueError as exc:
                # Topic may legitimately drift if the publisher restarts
                # mid-run; keep going.
                self.node.get_logger().warn(
                    f"[bridge] OmniHand ZMQ decode error: {exc}"
                )
                continue
            left = msg.fields.get("left_hand_joints")
            right = msg.fields.get("right_hand_joints")
            if left is None or right is None:
                continue
            left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
            right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
            if left_arr.size != 10 or right_arr.size != 10:
                # Wrong shape -- log once and stop trying. Layouts that
                # send 16-D or 20-D hand vectors must remap upstream.
                self.node.get_logger().warn(
                    f"[bridge] OmniHand ZMQ unexpected shape "
                    f"left={left_arr.shape} right={right_arr.shape}; "
                    "expected (10,) each. Dropping."
                )
                continue
            with self._hand_lock:
                self._hand_left_active = left_arr
                self._hand_right_active = right_arr
                self._hand_msg_count += 1
            if not self._hand_first_msg_logged:
                self._hand_first_msg_logged = True
                self.node.get_logger().info(
                    f"[bridge] OmniHand ZMQ: first finger setpoint received "
                    f"(|L|={float(np.linalg.norm(left_arr)):.3f} "
                    f"|R|={float(np.linalg.norm(right_arr)):.3f})."
                )

        try:
            sock.close(linger=0)
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Scene state PUB / scene reset SUB (G1 robocasa plumbing)
    # ----------------------------------------------------------------
    def _publish_robot_pose(self):
        """PUB the bridge's ground-truth pelvis qpos for host-side eval tools.

        Always-on companion to ``_publish_scene_state``; does not depend on
        scene metadata. The deploy's ``x2_debug`` topic only carries IMU
        orientation (no XY/Z), so any tool that wants to measure how far
        the robot actually walked / how far it rotated needs this PUB.

        Wire format: see :mod:`gear_sonic.utils.teleop.zmq.robot_pose_zmq`.
        """
        sock = getattr(self, "_robot_pose_pub_sock", None)
        pack = getattr(self, "_robot_pose_pack", None)
        if sock is None or pack is None:
            return
        # Pelvis is the floating-base free joint; first 7 qpos entries are
        # [x, y, z, qw, qx, qy, qz] in MuJoCo convention.
        pelvis_qpos = self.mj_data.qpos[0:7].astype(float).tolist()
        try:
            sock.send(pack(float(self.mj_data.time), pelvis_qpos), flags=0)
        except Exception as exc:
            if not getattr(self, "_robot_pose_pub_warned", False):
                print(f"[bridge] robot_pose PUB send failed: {exc}", flush=True)
                self._robot_pose_pub_warned = True

    def _collect_grasp_contacts(self) -> dict:
        """Walk ``mj_data.contact[:ncon]`` and bucket each contact pair
        into ``{logical_object_name: {"left": bool, "right": bool, "any": bool}}``.

        A contact is attributed to side ``S`` if exactly one of its two
        geoms belongs to that side's hand-geom set AND the other belongs
        to the object's contact-geom set. ``any`` is the OR over sides --
        useful for non-laterality oracles (e.g. bowl-on-target).

        No-op (returns ``{}``) when scene metadata wasn't detected.
        Cost is O(ncon) with per-contact O(1) set lookups -- typically
        fewer than a hundred contacts per tick on the X2 + tabletop scene.
        """
        if not self._scene_object_geom_ids:
            return {}
        out: dict[str, dict[str, bool]] = {
            logical: {"left": False, "right": False, "any": False}
            for logical in self._scene_object_geom_ids
        }
        ncon = int(self.mj_data.ncon)
        if ncon == 0:
            return out
        # Cache for lookups inside the loop.
        left = self._scene_hand_geom_ids.get("left", set())
        right = self._scene_hand_geom_ids.get("right", set())
        for k in range(ncon):
            c = self.mj_data.contact[k]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            for logical, obj_gids in self._scene_object_geom_ids.items():
                if g1 in obj_gids and g2 in obj_gids:
                    continue  # self-contact (e.g. bowl wall vs bowl floor)
                if g1 in obj_gids:
                    other = g2
                elif g2 in obj_gids:
                    other = g1
                else:
                    continue
                if other in left:
                    out[logical]["left"] = True
                    out[logical]["any"] = True
                elif other in right:
                    out[logical]["right"] = True
                    out[logical]["any"] = True
                else:
                    # Object-vs-non-hand contact (table, floor, other
                    # object). Surface it under "any" so the oracle can
                    # know "the cube touched something" without splitting
                    # by side, but DON'T flag a side (otherwise grasp
                    # signals would fire on table contact).
                    out[logical]["any"] = True
        return out

    def _collect_fingertip_pos(self) -> dict:
        """Return ``{side: [[x, y, z], …]}`` for every fingertip body in
        the metadata, evaluated against the bridge's current ``data.xpos``.

        Five entries per side (thumb/index/middle/ring/pinky distal
        phalanx); empty list when the side wasn't resolved (e.g. on a
        flat-floor sim with no scene metadata). The mirror uses the
        minimum fingertip-to-cube distance for the shaped-reward
        ``approach`` phase.
        """
        out: dict[str, list[list[float]]] = {}
        for side, bids in self._scene_fingertip_bids.items():
            out[side] = [
                self.mj_data.xpos[bid].astype(float).tolist() for bid in bids
            ]
        return out

    def _publish_scene_state(self):
        """PUB the current scene-object freejoint qpos + welded body pos.

        Called from the sim loop at the same cadence as
        :meth:`_publish_joint_states`. No-op when scene metadata wasn't
        detected at startup, or when ``--no-scene-zmq`` is set, or when
        the SUB endpoint failed to bind.

        Wire format: see ``gear_sonic.utils.teleop.zmq.scene_state_zmq``.
        """
        sock = getattr(self, "_scene_state_pub_sock", None)
        if sock is None:
            return
        try:
            from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
                pack_json,
                SCENE_STATE_TOPIC,
            )
        except ImportError:
            return
        payload = dict(
            sim_time=float(self.mj_data.time),
            object_freejoint_qpos={
                jname: self.mj_data.qpos[qadr:qadr + 7].astype(float).tolist()
                for jname, qadr in self._scene_freejoint_qadr.items()
            },
            mutable_body_pos={
                bname: self.mj_model.body_pos[bid].astype(float).tolist()
                for bname, bid in self._scene_welded_bid.items()
            },
            grasp_contacts=self._collect_grasp_contacts(),
            fingertip_pos=self._collect_fingertip_pos(),
        )
        try:
            sock.send(pack_json(SCENE_STATE_TOPIC, payload), flags=0, copy=False)
        except Exception as exc:
            # The PUB is fire-and-forget; complain once but don't crash.
            if not getattr(self, "_scene_state_pub_warned", False):
                self.node.get_logger().warn(
                    f"[bridge] scene_state PUB send failed: {exc}"
                )
                self._scene_state_pub_warned = True

    def _scene_reset_zmq_thread(self):
        """SUB on scene_reset and *enqueue* fresh per-episode object poses.

        Mirror of :meth:`_omnihand_zmq_thread`: the SUB thread does
        **not** mutate ``mj_data`` / ``mj_model`` directly. Instead it
        stashes the latest ``ResetObjects`` payload under
        :attr:`_pending_scene_reset_lock` and the sim thread drains it
        from inside its existing ``viewer.lock()`` (or
        ``_NullCtx`` in headless mode) at the start of the next
        :meth:`_sim_step_once`. This eliminates a real data race that
        would otherwise corrupt the MuJoCo solver state on the second
        / third B press of a long session (the reset thread would
        write ``qpos`` mid-``mj_step`` and the constraint solver
        would silently NaN, leaving the viewer rendering a frozen
        pose while CONTROL ticks kept ticking for ~20 s).

        Drop-old semantics: if a second message arrives before the
        sim thread has drained the first, the first is overwritten.
        That matches the recorder's ad-hoc usage (one B press == one
        new episode) and also matches ``ZMQ_CONFLATE`` on the SUB
        side. Velocity zeroing + ``mj_forward`` happen inside the
        drain (see :meth:`_apply_scene_reset`) so the cube doesn't
        teleport with an instantaneous (q_new - q_old)/dt velocity.
        """
        # The SUB is always created (unless --no-scene-zmq): besides scene
        # objects, a scene_reset payload may carry sim-only test hooks
        # (``band_enable`` / ``band_length`` for the elastic band, used to
        # rehearse the deploy's SAFE_HOLD -> RECOVER assisted stand-up).
        if getattr(self.args, "no_scene_zmq", False):
            return
        if not self._ensure_pyzmq_in_container(purpose="scene_reset SUB"):
            self.node.get_logger().warn(
                "[bridge] scene_reset SUB: pyzmq install failed; recorder "
                "will not be able to push fresh per-episode object poses."
            )
            return
        import zmq
        try:
            from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
                unpack_json,
                SCENE_RESET_TOPIC,
            )
        except ImportError as exc:
            self.node.get_logger().warn(
                f"[bridge] scene_reset SUB: cannot import unpack_json ({exc}); "
                "scene resets disabled."
            )
            return

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.setsockopt(zmq.SUBSCRIBE, SCENE_RESET_TOPIC.encode())
        endpoint = (
            f"tcp://{self.args.scene_reset_sub_host}:{self.args.scene_reset_sub_port}"
        )
        try:
            sock.connect(endpoint)
        except Exception as exc:
            self.node.get_logger().warn(
                f"[bridge] scene_reset SUB connect to {endpoint} failed: {exc}"
            )
            sock.close(linger=0)
            return
        self.node.get_logger().info(
            f"[bridge] scene_reset SUB connected at {endpoint} "
            f"(topic='{SCENE_RESET_TOPIC}')."
        )

        while not self._stop.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            except Exception as exc:
                self.node.get_logger().warn(
                    f"[bridge] scene_reset SUB recv error: {exc}"
                )
                continue
            try:
                _, payload = unpack_json(raw, expected_topic=SCENE_RESET_TOPIC)
            except ValueError as exc:
                self.node.get_logger().warn(
                    f"[bridge] scene_reset decode error: {exc}"
                )
                continue
            with self._pending_scene_reset_lock:
                dropped = self._pending_scene_reset is not None
                self._pending_scene_reset = payload
                self._pending_scene_reset_count += 1
            if dropped:
                self.node.get_logger().info(
                    "[bridge] scene_reset queued (overwrote previous "
                    "un-applied payload; sim thread is behind)."
                )
            else:
                n_fj = len(payload.get("object_freejoint_qpos", {}))
                n_bd = len(payload.get("mutable_body_pos", {}))
                self.node.get_logger().info(
                    f"[bridge] scene_reset queued: "
                    f"{n_fj} freejoints, {n_bd} welded bodies"
                )

    def _drain_pending_scene_reset(self) -> Optional[dict]:
        """Pop the most recent queued ``scene_reset`` payload, if any.

        Called once per sim cycle from :meth:`_sim_step_once` while the
        sim thread already holds ``viewer.lock()`` (or runs headless).
        Returns ``None`` if no payload is waiting.
        """
        with self._pending_scene_reset_lock:
            payload = self._pending_scene_reset
            self._pending_scene_reset = None
        return payload

    def _apply_scene_reset(self, payload: dict) -> None:
        """Write a scene_reset payload into the bridge's MuJoCo state.

        Caller invariant: the sim thread already holds ``viewer.lock()``
        (or runs headless), so this method does **no** locking of its
        own. Direct mutation of ``mj_data`` / ``mj_model`` from any
        other thread will race the constraint solver -- always go
        through :meth:`_drain_pending_scene_reset`.
        """
        freejoint_qpos = payload.get("object_freejoint_qpos", {})
        body_pos = payload.get("mutable_body_pos", {})

        import mujoco as _mujoco

        # Sim-only test hooks: elastic band on/off and length ("someone
        # holds the robot up" for the deploy's RECOVER rehearsal).
        if self.elastic_band is not None:
            if "band_length" in payload:
                self.elastic_band.length = float(payload["band_length"])
                print(f"[bridge] scene_reset: band length -> {self.elastic_band.length:.2f} m", flush=True)
            if "band_enable" in payload:
                self.elastic_band.enable = bool(payload["band_enable"])
                if self.elastic_band.enable:
                    # an explicit re-engage wins over --band-release-after-s
                    self.args.band_release_after_s = -1.0
                print(f"[bridge] scene_reset: ElasticBand enable -> {self.elastic_band.enable}", flush=True)

        for jname, qpos in freejoint_qpos.items():
            qadr = self._scene_freejoint_qadr.get(jname)
            if qadr is None:
                continue
            if len(qpos) < 7:
                self.node.get_logger().warn(
                    f"[bridge] scene_reset {jname!r} qpos has "
                    f"len={len(qpos)} (need 7); skipping."
                )
                continue
            self.mj_data.qpos[qadr:qadr + 7] = np.asarray(
                qpos[:7], dtype=np.float64
            )
            vadr = self._scene_freejoint_dofadr.get(jname)
            if vadr is not None:
                self.mj_data.qvel[vadr:vadr + 6] = 0.0
        for bname, pos in body_pos.items():
            bid = self._scene_welded_bid.get(bname)
            if bid is None:
                continue
            if len(pos) < 3:
                continue
            self.mj_model.body_pos[bid] = np.asarray(pos[:3], dtype=np.float64)
        _mujoco.mj_forward(self.mj_model, self.mj_data)
        self.node.get_logger().info(
            f"[bridge] scene_reset applied: "
            f"{len(freejoint_qpos)} freejoints, {len(body_pos)} welded bodies"
        )

    # ----------------------------------------------------------------
    # ElasticBand
    # ----------------------------------------------------------------
    def _apply_elastic_band(self):
        """Apply the virtual suspension force to the pelvis (if armed).

        Mirrors the 6-line block in ``gear_sonic/utils/mujoco_sim/base_sim.py``
        verbatim: build the [pos, quat, lin_vel, ang_vel] pose, run it through
        ``ElasticBand.Advance``, and write to ``mj_data.xfrc_applied``. Auto-
        release: once the deploy has been driving for
        ``--band-release-after-s`` seconds we flip ``band.enable`` off so the
        robot drops onto its feet and the policy runs against gravity for real.
        Operators who want to hold the body in mid-air for camera work pass
        ``--band-release-after-s -1`` (never release); key 9 in the viewer
        also toggles the band on/off interactively.
        """
        if self.elastic_band is None:
            return
        # Auto-release applies in both headless and viewer modes -- when an
        # automated profile (e.g. --sim-profile handoff) schedules a release,
        # the viewer should faithfully show the body dropping when it would
        # in real life. Holding the band on forever is opt-in via -1.
        if (
            self.elastic_band.enable
            and self._first_command_received
            and self.args.band_release_after_s >= 0.0
            and (time.monotonic() - self._first_command_mono)
                > float(self.args.band_release_after_s)
        ):
            self.elastic_band.enable = False
            self.node.get_logger().info(
                f"[bridge] ElasticBand auto-released "
                f"({self.args.band_release_after_s:.2f}s after first deploy "
                "command) -- robot is now under gravity."
            )

        link = self.band_attached_link
        if self.elastic_band.enable:
            pose = np.concatenate([
                self.mj_data.xpos[link],
                self.mj_data.xquat[link],
                np.zeros(6),
            ])
            # Pull world-frame [angular(3), linear(3)] velocities of the body.
            mujoco.mj_objectVelocity(
                self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                link, pose[7:13], 0,
            )
            # objectVelocity returns angular-first; ElasticBand wants linear-first.
            pose[7:10], pose[10:13] = pose[10:13].copy(), pose[7:10].copy()
            self.mj_data.xfrc_applied[link] = self.elastic_band.Advance(pose)
            self._log_band_force(self.mj_data.xfrc_applied[link])
        else:
            self.mj_data.xfrc_applied[link] = np.zeros(6)

    def _log_band_force(self, f) -> None:
        """Sim-only diagnostic (2026-09-08): the band's assist force = how much a "strap" had to help. Appends
        ``t,fx,fy,fz,fmag`` every 10th step to $X2_BAND_FORCE_LOG_DIR/band_force.csv when that dir is set and exists
        (the simstack's log dir); silent otherwise."""
        n = getattr(self, "_band_log_n", 0) + 1; self._band_log_n = n
        if n % 10: return
        fh = getattr(self, "_band_log_fh", None)
        if fh is None:
            import os
            d = os.environ.get("X2_BAND_FORCE_LOG_DIR", "")
            if not os.path.isdir(d): self._band_log_fh = False; return
            fh = open(os.path.join(d, "band_force.csv"), "w"); fh.write("t,fx,fy,fz,fmag\n"); self._band_log_fh = fh
        if fh is False: return
        fh.write(f"{self.mj_data.time:.3f},{f[0]:.1f},{f[1]:.1f},{f[2]:.1f},{float(np.linalg.norm(f[:3])):.1f}\n")
        if n % 500 == 0: fh.flush()

    def _apply_tilt(self) -> None:
        """HAND-TILT probe: torque about the nudge body's pitch (body Y) or roll (body X) axis,
        expressed in world frame via the body's current rotation; profile ramp -> hold -> release
        -> rest, sign alternating. Shares xfrc_applied[nudge body] with --nudge-force (use one)."""
        tq = getattr(self.args, "tilt_torque", 0.0) or 0.0
        if tq <= 0.0:
            return
        if not hasattr(self, "_tilt_body_id"):
            try:
                self._tilt_body_id = self.mj_model.body(self.args.nudge_body).id
            except Exception:
                self._tilt_body_id = -1
            self._tilt_period = (self.args.tilt_ramp + self.args.tilt_hold
                                 + self.args.tilt_release + self.args.tilt_rest)
            self._tilt_last_idx = -1
            print(f"[bridge] HAND-TILT probe: {tq:.1f} N*m {self.args.tilt_axis} on "
                  f"{self.args.nudge_body}: ramp {self.args.tilt_ramp}s hold {self.args.tilt_hold}s "
                  f"release {self.args.tilt_release}s rest {self.args.tilt_rest}s "
                  f"(period {self._tilt_period:.1f}s, alternate={self.args.tilt_alternate})",
                  flush=True)
        if self._tilt_body_id < 0:
            return
        t = float(self.mj_data.time) - float(getattr(a0 := self.args, 'tilt_start', 0.0) or 0.0)
        if t < 0.0:
            self.mj_data.xfrc_applied[self._tilt_body_id] = np.zeros(6)
            return
        idx = int(t // self._tilt_period)
        phase = t - idx * self._tilt_period
        a = self.args
        if phase < a.tilt_ramp:
            mag = tq * phase / max(a.tilt_ramp, 1e-6)
        elif phase < a.tilt_ramp + a.tilt_hold:
            mag = tq
        elif phase < a.tilt_ramp + a.tilt_hold + a.tilt_release:
            mag = tq * (1.0 - (phase - a.tilt_ramp - a.tilt_hold) / max(a.tilt_release, 1e-6))
        else:
            mag = 0.0
        sign = -1.0 if (a.tilt_alternate and idx % 2) else 1.0
        fm = float(getattr(a, "tilt_follow_margin", 0.0) or 0.0)
        if fm > 0.0 and phase < a.tilt_ramp + a.tilt_hold:
            # follow mode: push just above what the policy's waist actuator is applying right now
            if not hasattr(self, "_tilt_follow_aid"):
                jn = "waist_pitch_joint" if a.tilt_axis == "pitch" else "waist_roll_joint"
                self._tilt_follow_aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{jn}")
                print(f"[bridge] HAND-TILT FOLLOW mode: |tau_{jn}| + {fm:.1f} N*m, capped at {tq:.1f}", flush=True)
            tau_w = abs(float(self.mj_data.actuator_force[self._tilt_follow_aid])) if self._tilt_follow_aid >= 0 else 0.0
            follow = min(tq, tau_w + fm)
            mag = follow * (phase / max(a.tilt_ramp, 1e-6)) if phase < a.tilt_ramp else follow
        if idx != self._tilt_last_idx:
            self._tilt_last_idx = idx
            print(f"[bridge] HAND-TILT push #{idx} sign {sign:+.0f} at sim t={t:.1f}s", flush=True)
        # body-frame axis -> world: R = xmat (row-major 3x3); pitch = body +Y, roll = body +X
        R = np.asarray(self.mj_data.xmat[self._tilt_body_id]).reshape(3, 3)
        axis_body = np.array([0.0, 1.0, 0.0]) if a.tilt_axis == "pitch" else np.array([1.0, 0.0, 0.0])
        vec = np.zeros(6)
        vec[3:6] = R @ axis_body * (sign * mag)
        self.mj_data.xfrc_applied[self._tilt_body_id] = vec

    def _apply_nudge(self) -> None:
        """E23: periodic chest-level shove to excite the waist.

        Off unless --nudge-force > 0. Applied to a body the elastic band does
        NOT touch (torso_link vs the band's pelvis), so the two force writers
        never overwrite each other. Direction alternates so the robot is not
        pushed steadily off the mat.
        """
        f = getattr(self.args, "nudge_force", 0.0) or 0.0
        if f <= 0.0:
            return
        if not hasattr(self, "_nudge_body_id"):
            try:
                self._nudge_body_id = self.mj_model.body(self.args.nudge_body).id
            except Exception:
                self._nudge_body_id = -1
            self._nudge_n = 0
            print(f"[bridge] NUDGE probe: {f:.1f} N on {self.args.nudge_body} "
                  f"every {self.args.nudge_period:.1f}s for "
                  f"{self.args.nudge_duration:.2f}s", flush=True)
        if self._nudge_body_id < 0:
            return
        t = float(self.mj_data.time)
        phase = t % max(self.args.nudge_period, 1e-6)
        on = phase < self.args.nudge_duration
        vec = np.zeros(6)
        if on:
            idx = int(t // max(self.args.nudge_period, 1e-6))
            sign = -1.0 if (self.args.nudge_alternate and idx % 2) else 1.0
            vec[1] = sign * f          # world-Y = lateral shove at the chest
        self.mj_data.xfrc_applied[self._nudge_body_id] = vec

    # ----------------------------------------------------------------
    # Sim loop
    # ----------------------------------------------------------------
    def _sim_step_once(self, state_period_steps: int, imu_period_steps: int,
                       viewer=None):
        """One physics tick + maybe-publish. Caller owns the cadence/loop.

        When ``viewer`` is non-None we hold ``viewer.lock()`` across the
        ``mj_step`` write so the viewer's render thread can't read a
        half-mutated ``mj_data`` (segfaults out of GLFW otherwise -- this is
        the documented MuJoCo passive-viewer pattern).

        Pre-handoff freeze
        ------------------
        Before the first deploy command arrives we DO NOT call ``mj_step``.
        State and IMU publishers still fire on schedule (the deploy needs
        fresh state to advance INIT -> WAIT_FOR_CONTROL -> CONTROL), but
        the body is held bit-exact at its RSI / init-pose configuration.

        Why: the deploy's INIT and (default 5 s) WAIT phases together can
        easily last hundreds of ms before the policy sends its first
        command. If the sim integrated the entire time, gravity, ground
        contact, and any non-zero joint vel from the RSI'd motion frame
        would silently evolve the body off its initial state. The deploy's
        first-tick observation would then NOT match Python eval's
        observation of the same frame -- making sim-to-sim parity a moving
        target. Freezing makes the bridge match ``eval_x2_mujoco_onnx.py``
        exactly: the policy's first inference sees the bit-exact RSI
        state.

        On hardware this is also what actually happens: HAL streams joint
        states from MC, but joints don't physically move until the deploy
        starts driving them. Mirroring that in sim is more honest than
        letting the bridge run free.

        The freeze breaks at the moment the bridge's command callback
        flips ``self._first_command_received`` (see ``_make_cmd_callback``).
        From that tick onward, mj_step runs at full rate, the elastic band
        auto-release schedule starts ticking against ``_first_command_mono``,
        and the sim is identical to its prior behaviour.
        """
        self._apply_nudge()
        self._apply_tilt()
        if not self._first_command_received:
            # Frozen-body publish: state stays at RSI values, the deploy
            # gets fresh timestamps so its freshness check still clears.
            # We still apply OmniHand qpos (if armed) so the viewer
            # shows finger motion as soon as the live VLA bridge starts
            # publishing -- which can be well before SONIC takes over,
            # and is a useful "pipeline alive" signal for the operator.
            self._apply_omnihand_qpos()
            # Even before the deploy has started commanding we honour
            # queued scene resets: the recorder may rerandomise the
            # cube while the operator is still in INIT/WAIT. No
            # mj_step is firing, so the only contender for the GIL on
            # mj_data is the viewer's render thread -- still hold
            # viewer.lock() to avoid a single bad frame.
            pending_reset = self._drain_pending_scene_reset()
            if pending_reset is not None:
                if viewer is not None:
                    with viewer.lock():
                        self._apply_scene_reset(pending_reset)
                else:
                    self._apply_scene_reset(pending_reset)
            self._sim_step_count += 1
            if self._sim_step_count % state_period_steps == 0:
                self._publish_joint_states()
                self._publish_scene_state()
                self._publish_robot_pose()
            if self._sim_step_count % imu_period_steps == 0:
                self._publish_imu()
            return
        self._apply_pd()
        self._apply_elastic_band()
        self._apply_omnihand_qpos()
        # Drain at most one queued scene_reset per sim cycle and apply
        # it immediately *before* mj_step, all under the same
        # viewer.lock() that brackets the step. This is the only place
        # in the bridge where mj_data.qpos / mj_model.body_pos are
        # written for scene objects -- the SUB thread merely enqueues.
        # Fixes the data race where ZMQ-side writes during mj_step
        # corrupted the constraint solver and froze the viewer after
        # ~1 reset (see _scene_reset_zmq_thread docstring).
        pending_reset = self._drain_pending_scene_reset()
        if viewer is not None:
            with viewer.lock():
                if pending_reset is not None:
                    self._apply_scene_reset(pending_reset)
                mujoco.mj_step(self.mj_model, self.mj_data)
        else:
            if pending_reset is not None:
                self._apply_scene_reset(pending_reset)
            mujoco.mj_step(self.mj_model, self.mj_data)
        self._sim_step_count += 1
        if self._sim_step_count % state_period_steps == 0:
            self._publish_joint_states()
            self._publish_scene_state()
            self._publish_robot_pose()
        if self._sim_step_count % imu_period_steps == 0:
            self._publish_imu()

    def _sim_loop_headless(self):
        """Background-thread sim loop used in headless (no viewer) mode."""
        sim_dt = float(self.args.sim_dt)
        state_period_steps = max(1, int(round(1.0 / float(self.args.state_rate_hz) / sim_dt)))
        imu_period_steps   = max(1, int(round(1.0 / float(self.args.imu_rate_hz)   / sim_dt)))

        next_t = time.monotonic()
        while not self._stop.is_set():
            self._sim_step_once(state_period_steps, imu_period_steps, viewer=None)
            self._report_rtf(sim_dt)

            # Real-time pacing.
            next_t += sim_dt
            now = time.monotonic()
            sleep_for = next_t - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Falling behind; reset cadence rather than spiral.
                next_t = now

    def _report_rtf(self, sim_dt: float) -> None:
        """Print achieved real-time factor every 5 sim-seconds.

        RTF < 1.0 means the pacing loop is overrunning its tick budget and
        the whole sim world (robot, clip playback, physics) runs slower
        than wall time -- which the viewer shows as uniformly sluggish
        motion. 1.00 = real time.
        """
        period_steps = max(1, int(round(5.0 / sim_dt)))
        if self._sim_step_count % period_steps:
            return
        now = time.monotonic()
        if self._rtf_mark_wall is not None:
            dsteps = self._sim_step_count - self._rtf_mark_step
            wall = now - self._rtf_mark_wall
            if wall > 0 and dsteps > 0:
                rtf = (dsteps * sim_dt) / wall
                print(f"[bridge] RTF {rtf:.2f} "
                      f"({dsteps * sim_dt:.0f} sim-s in {wall:.1f} wall-s)"
                      + ("  ** SLOWER THAN REAL TIME **" if rtf < 0.95 else ""),
                      flush=True)
        self._rtf_mark_step = self._sim_step_count
        self._rtf_mark_wall = now

    # ----------------------------------------------------------------
    # Initial state
    # ----------------------------------------------------------------
    def _reset_to_default_pose(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i in range(self.eval_x2.NUM_DOFS):
            self.mj_data.qpos[self.qpos_adr[i]] = float(self.eval_x2.DEFAULT_DOF[i])
            self.mj_data.qvel[self.qvel_adr[i]] = 0.0
        # Floating base at a sane height so the robot isn't immediately in the
        # ground plane. Pelvis-up height: eyeballed to ~0.85 m for X2 standing.
        if self.has_floating_base:
            self.mj_data.qpos[2] = 0.85
            self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _reset_to_named_pose(self, pose_name: str):
        """Initialize from a named entry in ``sim_init_poses.yaml``.

        Looks up ``pose_name`` (hyphens already normalized to underscores
        upstream), validates schema, applies per-joint offsets atop
        DEFAULT_DOF, sets the floating base at the YAML's ``pelvis_z``,
        and seeds the standby PD setpoint to the resulting joint config
        so the bridge holds the captured pose during autostart instead of
        fighting it by pulling toward DEFAULT_DOF.

        ``ElasticBand.length`` is also overridden from the YAML's
        ``band_length`` *only if the operator left ``--band-length`` at
        its default (0.0)*; an explicit ``--band-length`` always wins.
        """
        poses = _load_named_init_poses()
        if pose_name not in poses:
            available = sorted(poses.keys()) or ["<none -- YAML missing>"]
            raise ValueError(
                f"Unknown --init-pose {pose_name!r}. "
                f"Available poses in {_INIT_POSES_YAML_PATH}: {available}. "
                "Use --init-pose=default for the built-in legacy behaviour."
            )
        entry = poses[pose_name]
        if not isinstance(entry, dict):
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r} must be a mapping."
            )

        # Required fields.
        try:
            pelvis_z = float(entry["pelvis_z"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r} missing or "
                "invalid required field 'pelvis_z' (float, m)."
            ) from e
        offsets_rad = entry.get("offsets_rad") or {}
        if not isinstance(offsets_rad, dict):
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r}.offsets_rad "
                "must be a mapping of joint_name -> float."
            )

        # Build the qpos joint vector starting from DEFAULT_DOF.
        target_dof = self.eval_x2.DEFAULT_DOF.astype(np.float64).copy()
        name_to_idx = {n: i for i, n in
                        enumerate(self.eval_x2.MUJOCO_JOINT_NAMES)}
        unknown: list[str] = []
        for joint_name, offset in offsets_rad.items():
            try:
                mj_idx = name_to_idx[str(joint_name)]
            except KeyError:
                unknown.append(str(joint_name))
                continue
            target_dof[mj_idx] += float(offset)
        if unknown:
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r} references "
                f"unknown joint name(s) {unknown}. Valid names are in "
                "x2_mujoco_ros_bridge.MUJOCO_JOINT_NAMES."
            )

        # Apply to MuJoCo state.
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i in range(self.eval_x2.NUM_DOFS):
            self.mj_data.qpos[self.qpos_adr[i]] = float(target_dof[i])
            self.mj_data.qvel[self.qvel_adr[i]] = 0.0
        if self.has_floating_base:
            self.mj_data.qpos[2] = pelvis_z
            self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Override ElasticBand length from YAML iff operator left it at the
        # CLI default. Explicit --band-length always wins.
        band_length = entry.get("band_length")
        if band_length is not None and self.elastic_band is not None \
                and float(self.args.band_length) == 0.0:
            self.elastic_band.length = float(band_length)

        with self.cmd_lock:
            self._target_pos = target_dof.copy()
        self._init_pose_name = pose_name  # for the startup banner

    def _rsi_from_motion(self, motion_path: pathlib.Path, frame: int):
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion not found: {motion_path}")
        suffix = motion_path.suffix.lower()
        if suffix == ".pkl":
            data = self.eval_x2.load_motion_data(str(motion_path))
        else:
            raise ValueError(
                f"Unsupported motion source extension {suffix!r} for {motion_path}; "
                f"expected .pkl. (X2M2 is the deploy-binary "
                f"runtime format, not an RSI source -- pass the original PKL "
                f"the X2M2 was baked from.)"
            )
        fps = self.eval_x2.get_motion_fps(data)
        state = self.eval_x2.compute_motion_state(data, frame, fps)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.has_floating_base:
            self.mj_data.qpos[0:3] = state["root_pos_w"]
            self.mj_data.qpos[3:7] = state["root_quat_w_wxyz"]
            self.mj_data.qvel[0:3] = state["root_lin_vel_w"]
            # MuJoCo free-joint qvel convention: qvel[3:6] is BODY-LOCAL angular
            # velocity (NOT world frame). compute_motion_state returns
            # root_ang_vel_w in world frame, so we must rotate it into the body
            # frame here -- mirrors gear_sonic/scripts/eval_x2_mujoco.py:799-802
            # (_apply_init_state). Without this, the IMU's published
            # base_ang_vel disagrees with what the Python eval observes at the
            # same RSI state, which the policy reads as a fictitious tilt rate
            # and corrects against -- causing the parity-profile sim to fall
            # within ~2 s on iter-4000 even after the tokenizer layout fix.
            self.mj_data.qvel[3:6] = self.eval_x2.quat_rotate_inverse(
                state["root_quat_w_wxyz"], state["root_ang_vel_w"]
            )
        for i in range(self.eval_x2.NUM_DOFS):
            self.mj_data.qpos[self.qpos_adr[i]] = float(state["joint_pos_mj"][i])
            self.mj_data.qvel[self.qvel_adr[i]] = float(state["joint_vel_mj"][i])
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # Also seed the standby setpoint to the current joint pose (so the
        # robot doesn't snap back to the default pose between RSI and the
        # first deploy command).
        with self.cmd_lock:
            self._target_pos = state["joint_pos_mj"].astype(np.float64).copy()

    def _print_scene(self):
        print("=== bodies ===")
        for i in range(self.mj_model.nbody):
            print(f"  {i:3d}  {self.mj_model.body(i).name}")
        print("=== joints ===")
        for i in range(self.mj_model.njnt):
            j = self.mj_model.joint(i)
            print(f"  {i:3d}  {j.name}  type={j.type}  qposadr={j.qposadr[0]}  dofadr={j.dofadr[0]}")
        print("=== actuators ===")
        for i in range(self.mj_model.nu):
            print(f"  {i:3d}  {self.mj_model.actuator(i).name}")

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def run(self):
        """Run the bridge.

        Threading model differs by viewer mode:

        - **Headless** (``--viewer`` off): sim loop runs on a background
          thread, ``rclpy.spin_once`` runs on the main thread. This keeps
          the original lean structure for CI / smoke tests.
        - **Viewer on**: sim loop runs on the **main thread** (because
          ``mujoco.viewer.launch_passive`` requires its ``sync()`` to be
          called from the thread that opened it, and we want sim + render
          serialised through ``viewer.lock()`` to avoid the well-known
          GLFW segfault when ``mj_step`` and ``viewer.sync`` race on
          ``mj_data``). ROS spinning is pushed to a background thread.

        The OmniHand ZMQ subscriber, when armed (``--with-omnihand`` and
        not ``--no-hand-zmq``), is *always* a background thread regardless
        of viewer mode. It only refreshes a small numpy buffer under a
        lock; the actual qpos write happens on the sim thread inside
        ``_apply_omnihand_qpos``.
        """
        self._print_banner()

        # Spawn OmniHand ZMQ SUB once we know the bridge is past
        # construction. Only when --with-omnihand AND --no-hand-zmq is
        # off; the latter is useful for static screenshots.
        hand_zmq_thread = None
        if self.args.with_omnihand and not self.args.no_hand_zmq:
            hand_zmq_thread = threading.Thread(
                target=self._omnihand_zmq_thread,
                name="x2-omnihand-zmq",
                daemon=True,
            )
            hand_zmq_thread.start()

        # Always-on robot_pose PUB (sim-only ground-truth pelvis qpos for
        # host-side eval tools). Independent of scene metadata since the
        # deploy's x2_debug topic only carries IMU orientation.
        self._robot_pose_pub_sock = None
        self._robot_pose_pack = None
        if self.args.robot_pose_pub_port and self.args.robot_pose_pub_port > 0:
            if self._ensure_pyzmq_in_container(purpose="robot_pose PUB"):
                try:
                    import zmq
                    # The sim-thread may run with a stripped sys.path (the
                    # ros2 launcher rewrites it on exec). Make sure REPO_ROOT
                    # is present so ``gear_sonic`` resolves there too.
                    if str(REPO_ROOT) not in sys.path:
                        sys.path.insert(0, str(REPO_ROOT))
                    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import pack_robot_pose
                    self._robot_pose_pack = pack_robot_pose

                    ctx = zmq.Context.instance()
                    sock = ctx.socket(zmq.PUB)
                    sock.setsockopt(zmq.LINGER, 0)
                    endpoint = (
                        f"tcp://{self.args.robot_pose_pub_host}:{self.args.robot_pose_pub_port}"
                    )
                    sock.bind(endpoint)
                    self._robot_pose_pub_sock = sock
                    print(
                        f"[bridge] robot_pose PUB bound at {endpoint} "
                        f"(ground-truth pelvis qpos for eval tools)."
                    )
                except Exception as exc:
                    print(f"[bridge] robot_pose PUB setup failed: {exc}",
                          file=sys.stderr)
                    self._robot_pose_pub_sock = None
                    self._robot_pose_pack = None
            else:
                print(
                    "[bridge] robot_pose PUB disabled: pyzmq install failed.",
                    file=sys.stderr,
                )

        # Set up the scene_state PUB and scene_reset SUB if we have
        # scene metadata loaded. Both default to off when --no-scene-zmq
        # is set or when no scene metadata was found.
        self._scene_state_pub_sock = None
        scene_reset_thread = None
        if (
            self._scene_metadata is not None
            and not self.args.no_scene_zmq
            and (self._scene_freejoint_qadr or self._scene_welded_bid)
        ):
            # Self-heal pyzmq missing-from-container (same lazy-install
            # path the OmniHand SUB uses), otherwise the scene_state PUB
            # silently no-ops and the recorder's RobocasaTaskMirror sees
            # frozen object poses for the entire session.
            if self._ensure_pyzmq_in_container(purpose="scene_state PUB"):
                try:
                    import zmq
                    ctx = zmq.Context.instance()
                    pub_sock = ctx.socket(zmq.PUB)
                    # Drop pending messages on close for snappy shutdown.
                    pub_sock.setsockopt(zmq.LINGER, 0)
                    # CONFLATE goes on the SUB side (scene_state recorder
                    # SUB sets it), not on the PUB side here.
                    pub_endpoint = (
                        f"tcp://{self.args.scene_state_pub_host}:{self.args.scene_state_pub_port}"
                    )
                    pub_sock.bind(pub_endpoint)
                    self._scene_state_pub_sock = pub_sock
                    print(
                        f"[bridge] scene_state PUB bound at {pub_endpoint} "
                        f"({len(self._scene_freejoint_qadr)} freejoints, "
                        f"{len(self._scene_welded_bid)} bodies)."
                    )
                except Exception as exc:
                    print(f"[bridge] scene_state PUB setup failed: {exc}",
                          file=sys.stderr)
                    self._scene_state_pub_sock = None
            else:
                print(
                    "[bridge] scene_state PUB disabled: pyzmq install "
                    "failed (recorder will see frozen objects).",
                    file=sys.stderr,
                )

        # The scene_reset SUB runs whenever ZMQ is allowed, scene objects or
        # not: its payload also carries the sim-only elastic-band hooks used
        # to rehearse the deploy's SAFE_HOLD -> RECOVER assisted stand-up.
        if not self.args.no_scene_zmq:
            scene_reset_thread = threading.Thread(
                target=self._scene_reset_zmq_thread,
                name="x2-scene-reset-zmq",
                daemon=True,
            )
            scene_reset_thread.start()

        if not self.args.viewer:
            sim_thread = threading.Thread(
                target=self._sim_loop_headless,
                name="x2-mujoco-sim", daemon=True,
            )
            sim_thread.start()
            try:
                while not self._stop.is_set():
                    rclpy.spin_once(self.node, timeout_sec=0.05)
            except KeyboardInterrupt:
                pass
            self._stop.set()
            sim_thread.join(timeout=2.0)
            self.node.destroy_node()
            return

        # ---- viewer on: sim+render on main thread, ROS spin in bg ----
        spin_thread = threading.Thread(
            target=self._spin_until_stopped, name="x2-ros-spin", daemon=True,
        )
        spin_thread.start()

        sim_dt = float(self.args.sim_dt)
        state_period_steps = max(1, int(round(1.0 / float(self.args.state_rate_hz) / sim_dt)))
        imu_period_steps   = max(1, int(round(1.0 / float(self.args.imu_rate_hz)   / sim_dt)))

        # Forward keypresses to the band so 7/8/9 work just like in the
        # G1 sim (key 9 toggles the suspension on/off).
        kw = {}
        if self.elastic_band is not None:
            kw["key_callback"] = self.elastic_band.MujuocoKeyCallback
        try:
            with mujoco_viewer.launch_passive(
                self.mj_model, self.mj_data, **kw
            ) as viewer:
                self.viewer = viewer
                # Optional: lock the camera to a body (--cam-track-body) so
                # the robot stays in view as it walks. Useful for long
                # stitched motions where the robot drifts off the default
                # free-camera framing within a few seconds. Setting the
                # cam fields here (after launch_passive) is safe -- the
                # passive viewer reads cam.* on every sync().
                track_body = getattr(self.args, "cam_track_body", None)
                if track_body:
                    bid = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_BODY, track_body
                    )
                    if bid < 0:
                        print(
                            f"[bridge] WARN --cam-track-body {track_body!r} "
                            f"not found in MJCF; falling back to free cam."
                        )
                    else:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = int(bid)
                        viewer.cam.distance = float(self.args.cam_distance)
                        viewer.cam.elevation = float(self.args.cam_elevation)
                        viewer.cam.azimuth = float(self.args.cam_azimuth)
                        print(
                            f"[bridge] viewer camera tracking body "
                            f"{track_body!r} (bid={bid}) "
                            f"d={self.args.cam_distance:.2f} "
                            f"el={self.args.cam_elevation:+.1f} "
                            f"az={self.args.cam_azimuth:+.1f}"
                        )
                next_t = time.monotonic()
                while viewer.is_running() and not self._stop.is_set():
                    self._sim_step_once(state_period_steps, imu_period_steps,
                                        viewer=viewer)
                    # Render only at ~60 Hz to keep sim CPU-bound: sync()
                    # is cheap but does an X11 round-trip every call, so
                    # sub-sampling here helps headroom on the 1 kHz tick.
                    if self._sim_step_count % 16 == 0:
                        viewer.sync()
                    self._report_rtf(sim_dt)
                    next_t += sim_dt
                    now = time.monotonic()
                    sleep_for = next_t - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = now
        except KeyboardInterrupt:
            pass

        self._stop.set()
        spin_thread.join(timeout=2.0)
        self.node.destroy_node()

    def _spin_until_stopped(self):
        try:
            while not self._stop.is_set():
                rclpy.spin_once(self.node, timeout_sec=0.05)
        except Exception:
            # Don't drop a traceback on Ctrl-C / shutdown -- main thread
            # owns the lifecycle log.
            pass

    def _print_banner(self):
        print("=" * 72)
        print("  X2 Ultra MuJoCo <-> ROS 2 bridge")
        print("=" * 72)
        print(f"  MJCF                {self.mjcf_path}")
        print(f"  Floating base       {self.has_floating_base}")
        print(f"  IMU body            {self.imu_body}")
        print(f"  ROS_LOCALHOST_ONLY  {self._ros_localhost_only}")
        print(f"  ROS_DOMAIN_ID       {self._ros_domain_id}")
        print(f"  sim_dt              {self.args.sim_dt:.4f}s ({1.0/self.args.sim_dt:.0f} Hz)")
        print(f"  state pub rate      {self.args.state_rate_hz:.1f} Hz")
        print(f"  imu pub rate        {self.args.imu_rate_hz:.1f} Hz")
        if self.args.motion:
            print(f"  RSI motion          {self.args.motion} (frame {self.args.init_frame})")
        else:
            pose_label = getattr(self, "_init_pose_name", None)
            if pose_label is None:
                print(f"  Initial pose        DEFAULT_DOF (standing, pelvis_z=0.85m)")
            else:
                pelvis = float(self.mj_data.qpos[2]) if self.has_floating_base else float("nan")
                print(f"  Initial pose        {pose_label} (from "
                      f"{_INIT_POSES_YAML_PATH.name}, pelvis_z={pelvis:.3f}m)")
        print(f"  hold-stiffness-mult {self.args.hold_stiffness_mult}")
        if self.elastic_band is not None:
            band_extra = (
                f"length={self.elastic_band.length:.3f}m, "
                f"kp={self.elastic_band.kp_pos:.0f}/kd={self.elastic_band.kd_pos:.0f}"
            )
            if self.args.viewer:
                print(f"  ElasticBand         ENABLED  ({band_extra}; "
                      f"viewer keys: 9 toggle, 7/8 lower/raise)")
            else:
                if self.args.band_release_after_s >= 0.0:
                    print(f"  ElasticBand         ENABLED  ({band_extra}; "
                          f"auto-release {self.args.band_release_after_s:.2f}s "
                          f"after first deploy cmd)")
                else:
                    print(f"  ElasticBand         ENABLED  ({band_extra}; "
                          f"headless, NEVER auto-release)")
        else:
            print(f"  ElasticBand         disabled")
        print(f"  viewer              {self.args.viewer}")
        print("=" * 72)


def main():
    args = _build_arg_parser().parse_args()

    # DDS isolation -- set BEFORE rclpy imports happen inside the bridge ctor.
    if not args.no_localhost_only and "ROS_LOCALHOST_ONLY" not in os.environ:
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
    if "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(int(args.ros_domain_id))

    import rclpy
    rclpy.init()
    try:
        bridge = X2MujocoRosBridge(args)
        bridge.run()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
