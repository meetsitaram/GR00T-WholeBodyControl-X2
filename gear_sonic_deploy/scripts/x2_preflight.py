#!/usr/bin/env python3
"""
Gantry-aware safety preflight for the X2 Ultra whole-body deploy.

Run this *before* `deploy_x2.sh local` to confirm that:

  1. All four joint-group state topics + the IMU are publishing.
  2. MC is running (publisher count proxy on /aima/hal/joint/arm/command,
     since the GetMcAction service has a type-hash mismatch against our
     aimdk_msgs build per the record-and-replay docs).
  3. No third party is publishing on the arm command topic.
  4. The robot is mechanically *ready* for the deploy hand-off:
       a. all joints are within `--default-pose-tol` of the standing IC
          (default_angles[] from policy_parameters.hpp);
       b. all joint velocities are below `--max-joint-vel` (no one is
          shoving the robot);
       c. all joint efforts are below `--max-effort` Nm (gantry is
          actually carrying the weight, no joint is jammed/fighting);
       d. the IMU agrees: gravity vector points roughly +z body, tilt
          below `--imu-tilt-deg`, and the rate gyro is quiet
          (`--imu-stillness-threshold`).

Adapted from AgiBot's record-and-replay reference (mc_control.py):

  * Their version assumes the robot is **seated**: legs are checked to be
    unloaded (hip/knee torques < 2 Nm), arms-only work. That is the wrong
    contract for our whole-body deploy where the policy itself drives the
    legs and the robot is hoisted on a gantry.
  * It does NOT stop MC. deploy_x2.sh handles MC stop via the EM HTTP API
    (PC1 <PC1_IP>:50080); keeping the responsibilities separate means a
    failed preflight does not require an MC restart.

Designed to be run inside the docker_x2/ container (`get_x2_sonic_ready.sh`),
where rclpy + aimdk_msgs are already available.

Exit code 0 on PASS (or PASS-with-warnings), 1 on FAIL.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# Match aimdk_io.cpp: SensorDataQoS (BEST_EFFORT, KEEP_LAST 10). Subscribing
# with the wrong reliability silently drops every message.

try:
    from aimdk_msgs.msg import JointStateArray
except ImportError as e:
    print(
        "ERROR: aimdk_msgs not found on the Python path. Run this inside the\n"
        "       docker_x2/ container (./get_x2_sonic_ready.sh) or source the colcon\n"
        "       overlay that built aimdk_msgs.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from sensor_msgs.msg import Imu


# ─────────────────────────────────────────────────────────────────────────
# Joint-group → MuJoCo-index mapping (mirrors aimdk_io.hpp; if you change
# one, change both):
#   leg   [0,  12)   12 joints
#   waist [12, 15)    3 joints
#   arm   [15, 29)   14 joints
#   head  [29, 31)    2 joints
#
# Stored as (start, length) so destructuring `(start, length)` matches the
# meaning at every call site. The end index is `start + length`.
# ─────────────────────────────────────────────────────────────────────────
GROUP_RANGES = {
    "leg":   (0,  12),
    "waist": (12, 3),
    "arm":   (15, 14),
    "head":  (29, 2),
}
NUM_DOFS = 31
ARM_CMD_TOPIC = "/aima/hal/joint/arm/command"

# ─────────────────────────────────────────────────────────────────────────
# Joint names (MJCF order; canonical). Mirrors mujoco_joint_names[] in
# policy_parameters.hpp. Used only for human-readable diagnostics; the
# C++ deploy already validates names slot-by-slot at first state callback.
#
# TODO(rename): "MUJOCO_JOINT_NAMES" / "mujoco_joint_names" is a misnomer
# -- this is the canonical 31-joint name+order from the MJCF, used by
# IsaacLab training, MuJoCo sim2sim eval, AND the real-robot deploy. A
# better name is `POLICY_JOINT_NAMES` / `policy_joint_names`. Multi-file
# rename tracked in codegen_x2_policy_parameters.py near the
# _emit_string_array call site; do not rename one location in isolation.
# ─────────────────────────────────────────────────────────────────────────
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_pitch_joint",     "waist_roll_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_yaw_joint",       "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_yaw_joint",      "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",             "head_pitch_joint",
)
assert len(MUJOCO_JOINT_NAMES) == NUM_DOFS

# ─────────────────────────────────────────────────────────────────────────
# Default standing pose (MuJoCo order, radians). MIRROR of default_angles[]
# in gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp.
# Both ultimately come from gear_sonic/envs/.../x2_ultra.py InitialStateCfg
# via codegen_x2_policy_parameters.py. If the C++ table changes, regenerate
# this one by hand or re-run the codegen and copy.
# ─────────────────────────────────────────────────────────────────────────
DEFAULT_ANGLES = (
    -0.312, 0.0,   0.0,    0.669, -0.363, 0.0,
    -0.312, 0.0,   0.0,    0.669, -0.363, 0.0,
     0.0,   0.0,   0.0,
     0.2,   0.2,   0.0,   -0.6,   0.0,    0.0,   0.0,
     0.2,  -0.2,   0.0,   -0.6,   0.0,    0.0,   0.0,
     0.0,   0.0,
)
assert len(DEFAULT_ANGLES) == NUM_DOFS


# ─────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""


# ─────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────
class PreflightNode(Node):
    """Aggregates state from every topic the deploy uses."""

    def __init__(self, imu_topic: str):
        super().__init__("x2_preflight")
        self._imu_topic = imu_topic
        self._joint_msgs: dict[str, Optional[JointStateArray]] = {
            g: None for g in GROUP_RANGES
        }
        self._imu_msg: Optional[Imu] = None
        # Rolling window (timestamp-less) of recent IMU samples for
        # stillness / upright median checks.
        self._imu_angvel_window: list[tuple[float, float, float]] = []
        self._imu_linacc_window: list[tuple[float, float, float]] = []

        for group in GROUP_RANGES:
            topic = f"/aima/hal/joint/{group}/state"
            self.create_subscription(
                JointStateArray,
                topic,
                lambda msg, g=group: self._on_joint(g, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            Imu, self._imu_topic, self._on_imu, qos_profile_sensor_data,
        )

    # ── Subscriptions ────────────────────────────────────────────────────

    def _on_joint(self, group: str, msg: JointStateArray) -> None:
        self._joint_msgs[group] = msg

    def _on_imu(self, msg: Imu) -> None:
        self._imu_msg = msg
        a = msg.angular_velocity
        l = msg.linear_acceleration
        self._imu_angvel_window.append((a.x, a.y, a.z))
        self._imu_linacc_window.append((l.x, l.y, l.z))
        if len(self._imu_angvel_window) > 1024:
            self._imu_angvel_window = self._imu_angvel_window[-512:]
            self._imu_linacc_window = self._imu_linacc_window[-512:]

    # ── Accessors ────────────────────────────────────────────────────────

    def joint_seen(self, group: str) -> bool:
        return self._joint_msgs[group] is not None

    def all_joint_groups_seen(self) -> bool:
        return all(self.joint_seen(g) for g in GROUP_RANGES)

    def imu_seen(self) -> bool:
        return self._imu_msg is not None

    def reset_imu_window(self) -> None:
        self._imu_angvel_window = []
        self._imu_linacc_window = []

    def imu_max_abs_angvel(self) -> float:
        if not self._imu_angvel_window:
            return 0.0
        return max(max(abs(c) for c in v) for v in self._imu_angvel_window)

    def imu_median_linacc(self) -> tuple[float, float, float]:
        """Per-axis median of the linear_acceleration samples."""
        if not self._imu_linacc_window:
            return (0.0, 0.0, 0.0)
        xs = sorted(s[0] for s in self._imu_linacc_window)
        ys = sorted(s[1] for s in self._imu_linacc_window)
        zs = sorted(s[2] for s in self._imu_linacc_window)
        m = len(xs) // 2
        return (xs[m], ys[m], zs[m])

    def joint_arrays(self) -> tuple[list[float], list[float], list[float]]:
        """Return (pos, vel, effort) in MuJoCo order, packed across groups.

        Slots from groups that have not yet published are filled with NaN
        so callers can detect them.
        """
        nan = float("nan")
        pos = [nan] * NUM_DOFS
        vel = [nan] * NUM_DOFS
        eff = [nan] * NUM_DOFS
        for group, (start, length) in GROUP_RANGES.items():
            msg = self._joint_msgs[group]
            if msg is None:
                continue
            joints = msg.joints  # JointState[]
            if len(joints) != length:
                # Surface the mismatch but don't crash; the deploy itself
                # will refuse to ingest at first callback (see aimdk_io.cpp).
                continue
            for i, j in enumerate(joints):
                pos[start + i] = j.position
                vel[start + i] = j.velocity
                eff[start + i] = j.effort
        return pos, vel, eff

    def arm_cmd_publisher_count(self) -> int:
        own = self.get_name()
        infos = self.get_publishers_info_by_topic(ARM_CMD_TOPIC)
        return sum(1 for i in infos if i.node_name != own)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def spin_for(node: Node, seconds: float, dt: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=dt)


def status_icon(status: str) -> str:
    return {"PASS": "+", "WARN": "!", "FAIL": "x"}.get(status, "?")


def fmt_topn_offenders(values: list[float], n: int = 3) -> str:
    """Format the n joints with the largest |value|."""
    pairs = [
        (i, v) for i, v in enumerate(values) if not math.isnan(v)
    ]
    pairs.sort(key=lambda iv: abs(iv[1]), reverse=True)
    head = pairs[:n]
    return ", ".join(f"{MUJOCO_JOINT_NAMES[i]}={v:+.3f}" for i, v in head)


# ─────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────
def check_joint_groups(node: PreflightNode, timeout: float) -> list[CheckResult]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.all_joint_groups_seen():
            break

    out: list[CheckResult] = []
    for g, (_, length) in GROUP_RANGES.items():
        msg = node._joint_msgs[g]
        if msg is None:
            out.append(CheckResult(
                f"joint_state[{g}]", "FAIL",
                f"no message on /aima/hal/joint/{g}/state within {timeout:.1f}s",
            ))
            continue
        n_actual = len(msg.joints)
        if n_actual != length:
            out.append(CheckResult(
                f"joint_state[{g}]", "FAIL",
                f"/aima/hal/joint/{g}/state delivered {n_actual} joints, "
                f"expected {length}",
            ))
            continue
        out.append(CheckResult(
            f"joint_state[{g}]", "PASS",
            f"/aima/hal/joint/{g}/state alive ({n_actual} joints)",
        ))
    return out


def check_imu(node: PreflightNode, timeout: float) -> CheckResult:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.imu_seen():
            break
    if not node.imu_seen():
        return CheckResult(
            "imu_alive", "FAIL",
            f"no message on {node._imu_topic} within {timeout:.1f}s "
            f"(try --imu-topic /aima/hal/imu/torse/state on firmware that "
            f"ships with the SDK-example typo)",
        )
    return CheckResult("imu_alive", "PASS", f"{node._imu_topic} alive")


def check_mc_running(node: PreflightNode) -> CheckResult:
    """MC running ⇒ ≥1 publisher on /aima/hal/joint/arm/command.

    DDS discovery from a freshly-started node can take several seconds to
    converge (record-and-replay had the same race). Poll for up to 5 s.
    """
    deadline = time.monotonic() + 5.0
    pub_count = 0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        pub_count = node.arm_cmd_publisher_count()
        if pub_count > 0:
            break
        time.sleep(0.25)

    if pub_count > 0:
        return CheckResult(
            "mc_status", "PASS",
            f"MC is running ({pub_count} publisher(s) on {ARM_CMD_TOPIC})",
        )
    return CheckResult(
        "mc_status", "WARN",
        f"MC appears already stopped (0 publishers on {ARM_CMD_TOPIC}). "
        f"Continuing -- pass --no-stop-mc to deploy_x2.sh to suppress its "
        f"stop_app POST and the cleanup-trap restart.",
    )


def check_no_arm_cmd_conflict(
    node: PreflightNode, mc_running: bool,
) -> CheckResult:
    expected = 1 if mc_running else 0
    actual = node.arm_cmd_publisher_count()
    if actual > expected:
        return CheckResult(
            "arm_cmd_conflict", "FAIL",
            f"{actual - expected} unexpected publisher(s) on {ARM_CMD_TOPIC} "
            f"(total {actual}, expected {expected} from MC)",
        )
    return CheckResult(
        "arm_cmd_conflict", "PASS",
        f"no conflicting arm command publishers (total {actual}, expected {expected})",
    )


def check_imu_stillness(
    node: PreflightNode, sample_secs: float, threshold: float,
) -> CheckResult:
    """Sample IMU angular velocity for `sample_secs`; max |w| must be small.

    Threshold relaxed vs record-and-replay (0.1 rad/s) because a gantry-held
    robot can have a couple of degrees of low-frequency sway.
    """
    if not node.imu_seen():
        return CheckResult(
            "imu_stillness", "WARN", "IMU has no data; skipping stillness check",
        )
    node.reset_imu_window()
    spin_for(node, sample_secs)
    max_w = node.imu_max_abs_angvel()
    if max_w > threshold:
        return CheckResult(
            "imu_stillness", "FAIL",
            f"max |angular_velocity|={max_w:.3f} rad/s > {threshold:.3f} "
            f"(robot is moving — gantry settling or being handled?)",
        )
    return CheckResult(
        "imu_stillness", "PASS",
        f"max |angular_velocity|={max_w:.4f} rad/s < {threshold:.3f}",
    )


def check_imu_upright(
    node: PreflightNode, max_tilt_deg: float, gravity_tol: float,
) -> CheckResult:
    """Gravity vector from linear_acceleration must point ~+z body."""
    if not node._imu_linacc_window:  # noqa: SLF001
        return CheckResult(
            "imu_upright", "WARN",
            "no linear_acceleration samples in window; "
            "check_imu_stillness must run first",
        )

    gx, gy, gz = node.imu_median_linacc()
    g_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    horiz = math.sqrt(gx * gx + gy * gy)
    # |az| in denominator so a flipped robot (-z) still gets a finite tilt.
    tilt_rad = math.atan2(horiz, abs(gz))
    tilt_deg = math.degrees(tilt_rad)
    if abs(g_mag - 9.81) > gravity_tol:
        return CheckResult(
            "imu_upright", "FAIL",
            f"|gravity| = {g_mag:.2f} m/s^2 (expected ~9.81 +/- {gravity_tol}). "
            f"Median linear_acceleration = ({gx:+.2f}, {gy:+.2f}, {gz:+.2f}). "
            f"IMU may be saturated, miswired, or accelerating.",
        )
    if tilt_deg > max_tilt_deg:
        return CheckResult(
            "imu_upright", "FAIL",
            f"body tilt = {tilt_deg:.1f} deg > {max_tilt_deg:.1f} "
            f"(median linear_acceleration = ({gx:+.2f}, {gy:+.2f}, {gz:+.2f})). "
            f"Robot is not upright — gantry / harness alignment?",
        )
    return CheckResult(
        "imu_upright", "PASS",
        f"upright: |g|={g_mag:.2f} m/s^2, tilt={tilt_deg:.2f} deg "
        f"(g_body=({gx:+.2f},{gy:+.2f},{gz:+.2f}))",
    )


def check_joints_near_default(
    node: PreflightNode,
    tols_per_group: dict[str, float],
    strict: bool,
) -> list[CheckResult]:
    """Per-family pose check.

    Why per-family AND why WARN-by-default: a gantry-held robot is
    practically never at the SONIC standing IC. Operators report the
    robot leaning forward/back/sideways with knees partially bent and
    hands hanging at the sides -- all of which are mechanically fine
    but produce large deltas vs default_angles[]. Refusing to deploy in
    those conditions is operator-hostile.

    The deploy's SoftStartRamp (safety.cpp) ramps *policy output toward
    default standing pose*, NOT current pose toward default. So at t=0
    the commanded position is default_angles[i] regardless of where the
    joint actually is -- a big delta means a kp*delta step torque at
    deploy start (~99 Nm/rad on hips/knees, ~14 Nm/rad on arms/waist/
    head). For a 1 rad arm offset that is ~14 Nm of step torque, which
    is mechanically fine but will be visible.

    Default: WARN with the magnitude clearly reported so the operator
    knows what step to expect. Use --strict-pose to gate (e.g. when
    you want CI to enforce a known starting pose). The hard cap
    threshold (--pose-tol-*) is set generously by default so even
    --strict-pose only trips on truly weird poses (e.g. T-pose arms).
    """
    if not node.all_joint_groups_seen():
        return [CheckResult(
            "joints_near_default", "WARN",
            "skipped: not all joint state groups have published yet",
        )]
    pos, _, _ = node.joint_arrays()
    deltas = [
        (p - DEFAULT_ANGLES[i]) if not math.isnan(p) else float("nan")
        for i, p in enumerate(pos)
    ]

    out: list[CheckResult] = []
    for group, (start, length) in GROUP_RANGES.items():
        tol = tols_per_group[group]
        group_deltas = deltas[start:start + length]
        finite = [d for d in group_deltas if not math.isnan(d)]
        if not finite:
            out.append(CheckResult(
                f"joints_near_default[{group}]", "WARN",
                f"no position samples for group '{group}'",
            ))
            continue
        max_abs = max(abs(d) for d in finite)
        padded = [float("nan")] * NUM_DOFS
        for k, d in enumerate(group_deltas):
            padded[start + k] = d
        worst = fmt_topn_offenders(padded, n=3)
        if max_abs > tol:
            status = "FAIL" if strict else "WARN"
            note = (
                "Will SNAP at deploy t=0 (PD step). "
                if not strict else
                "Threshold exceeded (strict mode). "
            )
            out.append(CheckResult(
                f"joints_near_default[{group}]", status,
                f"max |q - q_default| = {max_abs:.3f} rad "
                f"({math.degrees(max_abs):.1f} deg) > {tol:.3f} "
                f"({math.degrees(tol):.1f} deg). "
                f"{note}Worst (delta rad): {worst}.",
            ))
        else:
            out.append(CheckResult(
                f"joints_near_default[{group}]", "PASS",
                f"max |q - q_default| = {max_abs:.3f} rad "
                f"({math.degrees(max_abs):.1f} deg) <= {tol:.3f}. "
                f"Top deltas: {worst}.",
            ))
    return out


def check_joints_low_velocity(
    node: PreflightNode, max_vel: float,
) -> CheckResult:
    if not node.all_joint_groups_seen():
        return CheckResult(
            "joints_low_velocity", "WARN",
            "skipped: not all joint state groups have published yet",
        )
    _, vel, _ = node.joint_arrays()
    finite = [v for v in vel if not math.isnan(v)]
    if not finite:
        return CheckResult(
            "joints_low_velocity", "FAIL",
            "no joint velocity samples",
        )
    max_abs = max(abs(v) for v in finite)
    worst = fmt_topn_offenders(vel, n=3)
    if max_abs > max_vel:
        return CheckResult(
            "joints_low_velocity", "FAIL",
            f"max |qdot| = {max_abs:.3f} rad/s > {max_vel:.3f}. "
            f"Top movers: {worst}. "
            f"Robot is being moved — let it settle before deploy.",
        )
    return CheckResult(
        "joints_low_velocity", "PASS",
        f"max |qdot| = {max_abs:.4f} rad/s <= {max_vel:.3f}",
    )


def check_joints_low_effort(
    node: PreflightNode, max_effort: float, strict: bool,
) -> CheckResult:
    """Max |effort| across all joints must be small.

    With MC running and the robot taking weight on a gantry, the motors
    should be carrying very little torque (the harness has the load, the
    PD loop is just holding pose). A high reading means either:
      * the gantry is not actually carrying weight — robot is hanging on
        its own joints (RISKY for the deploy hand-off);
      * a single joint is mechanically jammed or fighting an obstruction.

    Default is WARN (informational); pass --strict-effort to gate.
    """
    if not node.all_joint_groups_seen():
        return CheckResult(
            "joints_low_effort", "WARN",
            "skipped: not all joint state groups have published yet",
        )
    _, _, eff = node.joint_arrays()
    finite = [e for e in eff if not math.isnan(e)]
    if not finite:
        return CheckResult(
            "joints_low_effort", "WARN",
            "no joint effort samples (firmware may not populate JointState.effort)",
        )
    max_abs = max(abs(e) for e in finite)
    worst = fmt_topn_offenders(eff, n=3)
    if max_abs > max_effort:
        status = "FAIL" if strict else "WARN"
        return CheckResult(
            "joints_low_effort", status,
            f"max |effort| = {max_abs:.2f} Nm > {max_effort:.2f}. "
            f"Top loaded joints (Nm): {worst}. "
            f"Verify the gantry is carrying weight and no joint is jammed.",
        )
    return CheckResult(
        "joints_low_effort", "PASS",
        f"max |effort| = {max_abs:.2f} Nm <= {max_effort:.2f}. "
        f"Top loaded (Nm): {worst}.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="Safety preflight for the X2 Ultra whole-body deploy.",
    )
    p.add_argument(
        "--imu-topic", default="/aima/hal/imu/torso/state",
        help="IMU topic. Override to /aima/hal/imu/torse/state on firmware "
             "shipping with the SDK-example typo.",
    )
    p.add_argument(
        "--state-timeout", type=float, default=5.0,
        help="Seconds to wait for first message on each joint state / IMU "
             "topic before declaring it dead.",
    )
    p.add_argument(
        "--imu-stillness-window", type=float, default=1.0,
        help="Seconds of IMU samples to evaluate for stillness + upright.",
    )
    p.add_argument(
        "--imu-stillness-threshold", type=float, default=0.2,
        help="Max |angular_velocity| (rad/s) over the stillness window. "
             "Default 0.2 is relaxed from record-and-replay's 0.1 to "
             "tolerate gantry sway.",
    )
    p.add_argument(
        "--imu-tilt-deg", type=float, default=30.0,
        help="Max body tilt (deg from gravity-aligned upright). Default "
             "30 deg accommodates a gantry-held robot that is leaning; "
             "tighten to 10-15 deg for a powered standing run on the "
             "floor.",
    )
    p.add_argument(
        "--imu-gravity-tol", type=float, default=2.0,
        help="Tolerance on |gravity| vs 9.81 m/s^2 in the body frame.",
    )
    # Pose-tolerance defaults are deliberately permissive: a gantry-held
    # robot is rarely at the SONIC standing IC. Pose violations are WARN
    # by default (operator-informational, "deploy will snap by N rad");
    # use --strict-pose to gate.
    p.add_argument(
        "--pose-tol-leg", type=float, default=1.0,
        help="Max |q_meas - q_default| for any leg joint, radians. "
             "Default 1.0 rad (~57 deg) covers knees-straight to "
             "fully-crouched on a gantry.",
    )
    p.add_argument(
        "--pose-tol-waist", type=float, default=0.6,
        help="Max |q_meas - q_default| for any waist joint, radians. "
             "Default 0.6 rad (~34 deg) covers gantry lean fwd/back/side.",
    )
    p.add_argument(
        "--pose-tol-arm", type=float, default=1.5,
        help="Max |q_meas - q_default| for any arm joint, radians. "
             "Default 1.5 rad (~86 deg) covers hands hanging straight "
             "down at sides.",
    )
    p.add_argument(
        "--pose-tol-head", type=float, default=0.5,
        help="Max |q_meas - q_default| for any head joint, radians. "
             "Default 0.5 rad (~29 deg).",
    )
    p.add_argument(
        "--strict-pose", action="store_true",
        help="Promote pose-tolerance violations from WARN to FAIL. Use "
             "for CI / known-IC powered runs; leave off for gantry "
             "bring-up where any natural rest pose should not gate.",
    )
    p.add_argument(
        "--max-joint-vel", type=float, default=0.5,
        help="Max |qdot| per joint, radians/sec.",
    )
    p.add_argument(
        "--max-effort", type=float, default=15.0,
        help="Max |effort| per joint, Newton-meters. Soft (WARN) by default; "
             "use --strict-effort to make a violation FAIL.",
    )
    p.add_argument(
        "--strict-effort", action="store_true",
        help="Promote --max-effort violations from WARN to FAIL.",
    )
    args = p.parse_args()

    rclpy.init()
    try:
        node = PreflightNode(imu_topic=args.imu_topic)
    except Exception as e:
        print(f"  [x] rclpy node init failed: {e}", file=sys.stderr)
        rclpy.shutdown()
        return 1

    try:
        print("  Running X2 preflight ...")
        results: list[CheckResult] = []

        # 1. State topics alive.
        results.extend(check_joint_groups(node, args.state_timeout))
        results.append(check_imu(node, timeout=2.0))

        # 2. MC + arm-cmd conflict.
        mc_result = check_mc_running(node)
        results.append(mc_result)
        mc_running = mc_result.status == "PASS"
        results.append(check_no_arm_cmd_conflict(node, mc_running))

        # 3. IMU dynamics — stillness fills the linear_acceleration window
        #    that imu_upright then medians.
        if node.imu_seen():
            results.append(
                check_imu_stillness(
                    node, args.imu_stillness_window,
                    args.imu_stillness_threshold,
                )
            )
            results.append(
                check_imu_upright(
                    node, args.imu_tilt_deg, args.imu_gravity_tol,
                )
            )

        # 4. Joint-space "ready for hand-off" checks.
        if node.all_joint_groups_seen():
            tols_per_group = {
                "leg":   args.pose_tol_leg,
                "waist": args.pose_tol_waist,
                "arm":   args.pose_tol_arm,
                "head":  args.pose_tol_head,
            }
            results.extend(
                check_joints_near_default(node, tols_per_group, args.strict_pose)
            )
            results.append(
                check_joints_low_velocity(node, args.max_joint_vel)
            )
            results.append(
                check_joints_low_effort(node, args.max_effort, args.strict_effort)
            )

        # ── Print summary ────────────────────────────────────────────────
        print()
        print("  Preflight results:")
        for r in results:
            line = f"  [{status_icon(r.status)}] {r.name}"
            if r.detail:
                line += f" -- {r.detail}"
            print(line)
        print()

        failed = [r for r in results if r.status == "FAIL"]
        if failed:
            print(f"  RESULT: FAIL ({len(failed)} check(s) failed)")
            return 1
        warned = [r for r in results if r.status == "WARN"]
        if warned:
            print(f"  RESULT: PASS with {len(warned)} warning(s)")
        else:
            print("  RESULT: PASS")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
