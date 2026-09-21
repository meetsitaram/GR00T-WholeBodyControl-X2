"""Reward functions for the manager-based RL environment MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)
import torch

from gear_sonic.envs.manager_env.mdp.commands import (
    ForceTrackingCommand,
    TrackingCommand,
    _get_body_indexes,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    tracking_anchor_pos = None
    tracking_anchor_ori = None
    tracking_relative_body_pos = None
    tracking_relative_body_ori = None
    tracking_relative_body_ori_weighted = None
    tracking_body_linvel = None
    tracking_body_angvel = None
    # Arm-scoped velocity tracking (reuse the body_linvel/angvel funcs with
    # body_names=arm links). For fast-arm motions (shadow boxing / dance) where the
    # generic all-body velocity terms dilute the arms; scope + a sharp kernel make
    # slow/damped arms actually cost reward. Off (None) unless set in a config.
    tracking_arm_linvel = None
    tracking_arm_angvel = None
    action_rate_l2 = None
    joint_limit = None
    undesired_contacts = None
    undesired_contacts_no_hands = None
    undesired_contacts_no_ankle_hand = None
    tracking_body_pos = None
    tracking_body_ori = None
    tracking_vr_3point_global = None
    tracking_vr_3point_local = None
    tracking_vr_3point_force = None
    tracking_vr_2wrists_ori_tight = None
    tracking_vr_2wrists_local_ori = None
    # SONIC v1.1 reward stack (local_feet_acc_energy_5pt)
    energy_consumption = None
    tracking_head_local_ori = None
    anti_shake_ang_vel = None
    tracking_vr_5point_local = None
    motion_5point_local_pos = None
    feet_acc = None
    is_terminated = None
    upright_penalty = None
    # Soft-landing terms (2026-07-16): touchdown impact velocity + GRF
    # overshoot. Off (None) unless composed in a config -- see
    # config/manager_env/rewards/terms/feet_{impact_vel,grf_overshoot}.yaml.
    feet_impact_vel = None
    feet_grf_overshoot = None
    # Dof-subset tracking (2026-08-15, phase-2b wrist pin): makes
    # reward-invisible joint angles observable. Off unless composed.
    tracking_wrist_dof = None
    # Torque-demand-near-limit penalty (2026-09-02, stage 1b of the incumbent
    # vendor-plant adaptation). Off unless composed / overridden. NOTE: a term
    # missing here fails at env construction with "RewardsCfg.__init__() got
    # an unexpected keyword argument" -- hydra `--cfg job` does NOT catch it.
    torque_saturation = None


def tracking_anchor_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor position tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) position to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel. Smaller values produce
            sharper falloff and stricter tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    sq_dist = (diff * diff).sum(dim=-1)
    return torch.exp(-sq_dist / (std * std))


def tracking_anchor_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute anchor orientation tracking reward using a Gaussian kernel.

    Encourages the robot's anchor (root) orientation to match the reference motion anchor.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    angular_err = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    return torch.exp(-angular_err.square() / (std * std))


def upright_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_name: str | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize tilt of bodies away from upright.

    Compute the squared magnitude of the x/y components of the gravity vector
    in each body's local frame, summed across all specified bodies. When a body
    is perfectly upright the local gravity is [0, 0, -1] and the penalty is 0.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        body_name: Single body name (for backwards compatibility).
        body_names: List of body names. If both are None, defaults to ["pelvis"].

    Returns:
        Penalty tensor of shape (num_envs,). Zero when upright, positive when tilted.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    if body_names is None:
        body_names = [body_name] if body_name else ["pelvis"]

    total_penalty = torch.zeros(env.num_envs, device=env.device)
    for name in body_names:
        body_idx = robot.body_names.index(name)
        body_quat = robot.data.body_quat_w[:, body_idx]
        g_local = quat_apply(quat_inv(body_quat), command.down_dir)
        total_penalty += g_local[:, 0] ** 2 + g_local[:, 1] ** 2

    return total_penalty


def tracking_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body positions to match the reference motion. The reward is
    the mean squared distance across all tracked bodies, passed through an exponential.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_pos_slow_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None,
    v_max: float = 0.35, v_width: float = 0.1, gate_body: str = "pelvis",
) -> torch.Tensor:
    """World-frame body position tracking (see ``tracking_body_pos_error``) GATED to slow
    reference motion. gate = sigmoid((v_max - |v_ref|) / v_width) on the reference linear
    speed of ``gate_body`` (default pelvis): ~1 below v_max, ~0 above v_max + 3*v_width.

    Added 2026-09-05 for the micro-step fine-tune (pico_v12_chores): the feet get a tight
    world-position kernel (a skipped 5 cm step must cost reward) ONLY where the reference
    moves slowly, so fast walks / dances / boxing keep exactly their existing reward.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    base = torch.exp(-per_body_err.mean(dim=-1) / (std * std))
    g_idx = _get_body_indexes(command, [gate_body])
    v_ref = command.body_lin_vel_w[:, g_idx].norm(dim=-1).mean(dim=-1)
    gate = torch.sigmoid((v_max - v_ref) / max(v_width, 1e-6))
    return base * gate


def tracking_vr_3point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 3-point tracking reward in world frame using a Gaussian kernel.

    Encourages the robot's 3 VR tracking points (typically left wrist, right wrist,
    head) to match their reference positions in world frame.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_diff = command.robot_vr_3point_pos_w - command.vr_3point_body_pos_w
    per_point_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_point_err.mean(dim=-1) / (std * std))


def tracking_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in world frame.

    Measure the orientation error of 2 wrist bodies against the reference motion,
    similar to tracking_relative_body_ori_error but restricted to wrist links.

    NOTE: The rigid extension defined in vr_3point_body_offset can be skipped for
    orientation error since it does not affect rotations.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_local_vr_2wrists_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute wrist orientation tracking reward in the anchor's local frame.

    Transform both reference and robot wrist orientations into the anchor (root)
    frame before computing the angular error. This makes the reward invariant to
    global root orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: List of wrist body names (must be provided).

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    assert body_names is not None, "body_names must be provided"
    body_indexes = _get_body_indexes(command, body_names)
    num_bodies = len(body_indexes)

    # reference motion
    ref_wrist_quat_w = command.body_quat_w[:, body_indexes]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, num_bodies, 1)
    ref_wrist_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_wrist_quat_w)

    # robot
    robot_wrist_quat_w = command.robot_body_quat_w[:, body_indexes]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, num_bodies, 1
    )
    robot_wrist_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_wrist_quat_w)

    error = quat_error_magnitude(ref_wrist_quat_local, robot_wrist_quat_local) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def energy_consumption(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize instantaneous mechanical power across the robot joints."""
    if isinstance(asset_cfg, dict):
        robot_name = asset_cfg.get("name", "robot")
    else:
        robot_name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[robot_name]
    return torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=-1)


_TORQUE_SAT_IDS: dict = {}


def torque_saturation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 1.0,
    joint_names: list | None = None,
) -> torch.Tensor:
    """Penalize torque DEMAND above the joint effort limit (stage 1b, 2026-09-02).

    Sum over joints of ``max(0, |computed_torque| / effort_limit - margin)``.
    ``joint_names`` restricts the sum to those joints. List them EXPLICITLY
    (operator ruling 2026-09-02, no regex): stage 1b uses
    ``["waist_pitch_joint", "waist_roll_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint"]`` -- "limit this to just waist and ankle pitch
    for now to verify the theory works". ``None`` = all joints. Names are
    resolved once via ``Articulation.find_joints`` and cached per
    (robot, names); an empty match raises.
    ``computed_torque`` is the PD output BEFORE the effort clip, so this is the
    same ratio the ``act/demand_p99_*`` diagnostics log; ``energy_consumption``
    and every other term use ``applied_torque`` (post-clip) and are therefore
    blind to over-demand. With the X2 waist at its hardware 24 N.m the policy
    kept demanding ~1.4x the limit for 6k iterations of stage 1 -- the
    diagnostics showed it, nothing in the reward could act on it, and on the
    robot it is the waist snap-back. ``margin=1.0`` penalizes only actual
    over-demand; ``margin<1`` starts the gradient before the clip binds.
    Linear (not squared) so a saturating tick is felt at any excess.
    """
    if isinstance(asset_cfg, dict):
        robot_name = asset_cfg.get("name", "robot")
    else:
        robot_name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[robot_name]
    tau_c = robot.data.computed_torque
    lim = getattr(robot.data, "joint_effort_limits", None)
    if lim is None:
        lim = getattr(robot.data, "joint_effort_limit", None)
    if lim is None:
        raise RuntimeError(
            "torque_saturation: articulation data has no joint effort limits; "
            "refusing to silently return zero penalty"
        )
    ratio = tau_c.abs() / lim.abs().clamp(min=1e-6)
    if joint_names:
        key = (id(robot), tuple(joint_names))
        ids = _TORQUE_SAT_IDS.get(key)
        if ids is None:
            ids, names = robot.find_joints(list(joint_names))
            if len(ids) == 0:
                raise RuntimeError(
                    f"torque_saturation: joint_names {list(joint_names)} matched no joint "
                    f"of {robot.joint_names}"
                )
            print(f"[torque_saturation] scoped to {len(ids)} joints: {names}", flush=True)
            ids = torch.as_tensor(ids, device=ratio.device, dtype=torch.long)
            _TORQUE_SAT_IDS[key] = ids
        ratio = ratio.index_select(-1, ids)
    return torch.clamp(ratio - margin, min=0.0).sum(dim=-1)


def tracking_local_head_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Compute head orientation tracking reward in the anchor's local frame.

    Transform the head (torso_link) orientation into the anchor's local frame for
    both the reference motion and the robot, then compute the angular error. This
    encourages the robot to match the head-to-root relative orientation.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get the head body index (torso_link)
    head_body_names = ["torso_link"]
    body_indexes = _get_body_indexes(command, head_body_names)

    # reference motion: head orientation in world frame, transformed to anchor's local frame
    ref_head_quat_w = command.body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    ref_anchor_quat_w = command.anchor_quat_w.view(env.num_envs, 1, 4)
    ref_head_quat_local = quat_mul(quat_inv(ref_anchor_quat_w), ref_head_quat_w)

    # robot: head orientation in world frame, transformed to anchor's local frame
    robot_head_quat_w = command.robot_body_quat_w[:, body_indexes]  # [num_envs, 1, 4]
    robot_anchor_quat_w = command.robot_anchor_quat_w.view(env.num_envs, 1, 4)
    robot_head_quat_local = quat_mul(quat_inv(robot_anchor_quat_w), robot_head_quat_w)

    error = quat_error_magnitude(ref_head_quat_local, robot_head_quat_local) ** 2
    return torch.exp(-error.squeeze(-1) / std**2)


def tracking_local_vr_3point_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    point_weights: list[float] | None = None,
):
    """Compute VR 3-point tracking reward in the anchor's local frame.

    Transform tracking points into the anchor (root) local frame before computing
    position error, making the reward invariant to global root position/orientation.
    Supports optional per-point weighting.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        point_weights: Optional weights for each tracking point. Order matches
            vr_3point_body config (typically [left_wrist, right_wrist, head]).
            If None, all points are weighted equally.
            Example: [2, 2, 1] gives wrists 2x importance vs head.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    ref_3point_pos = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.vr_3point_body), 1
    )
    robot_3point_diff = command.robot_vr_3point_pos_w - command.robot_anchor_pos_w[:, None, :]
    robot_3point_pos = quat_apply(quat_inv(robot_root_quat), robot_3point_diff)
    diff = robot_3point_pos - ref_3point_pos
    error = torch.sum(torch.square(diff), dim=-1)  # [num_envs, num_points]

    if point_weights is not None:
        # Weighted mean: sum(w_i * e_i) / sum(w_i)
        weights = torch.tensor(point_weights, dtype=error.dtype, device=error.device)
        weighted_error = (error * weights).sum(dim=-1) / weights.sum()
    else:
        # Simple mean (equal weights)
        weighted_error = error.mean(dim=-1)

    return torch.exp(-weighted_error / std**2)


def tracking_local_vr_5point_error(env: ManagerBasedRLEnv, command_name: str, std: float):
    """Compute VR 5-point tracking reward in the anchor's local frame.

    Same approach as tracking_local_vr_3point_error but with 5 tracking points
    (e.g., 2 wrists + head + 2 feet).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ref_5point_diff = command.reward_point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    ref_5point_pos = quat_apply(quat_inv(ref_root_quat), ref_5point_diff)
    robot_root_quat = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(
        1, len(command.cfg.reward_point_body), 1
    )
    robot_5point_diff = (
        command.robot_reward_point_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    )
    robot_5point_pos = quat_apply(quat_inv(robot_root_quat), robot_5point_diff)
    diff = robot_5point_pos - ref_5point_pos
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_vr_3point_error_pos_force(
    env: ManagerBasedRLEnv, motion_command_name: str, force_command_name: str, std: float
):
    """Compute VR 3-point tracking reward with force-based compliance correction.

    Add a force-proportional offset to the wrist tracking error so that applied
    external forces shift the tracking target, enabling compliant behavior under
    force perturbations.

    Args:
        env: The environment.
        motion_command_name: Name of the motion tracking command term.
        force_command_name: Name of the force tracking command term.
        std: Standard deviation for the Gaussian kernel.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    motion_command: TrackingCommand = env.command_manager.get_term(motion_command_name)
    force_command: ForceTrackingCommand = env.command_manager.get_term(force_command_name)
    diff = motion_command.robot_vr_3point_pos_w - motion_command.vr_3point_body_pos_w
    force_error_wrists = (
        force_command.last_force_applied * force_command.eef_stiffness_buf[:, :, None]
    )
    diff[:, :2] += force_error_wrists
    error = torch.sum(torch.square(diff), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def tracking_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward in world frame using a Gaussian kernel.

    Encourages tracked body orientations to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_w[:, tracked], command.robot_body_quat_w[:, tracked]
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body position tracking reward using anchor-relative reference positions.

    Use reference body positions that have been shifted to share the robot's anchor
    (root) position, so only the relative pose matters rather than absolute position.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    pos_diff = command.body_pos_relative_w[:, tracked] - command.robot_body_pos_w[:, tracked]
    per_body_err = (pos_diff * pos_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body orientation tracking reward using anchor-relative reference orientations.

    Use reference body orientations that have been transformed to share the robot's
    anchor (root) orientation, making the reward invariant to global heading.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    angular_err = quat_error_magnitude(
        command.body_quat_relative_w[:, tracked],
        command.robot_body_quat_w[:, tracked],
    )
    return torch.exp(-angular_err.square().mean(dim=-1) / (std * std))


def tracking_relative_body_ori_weighted_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute anchor-relative body orientation tracking reward with per-body weights.

    Same as tracking_relative_body_ori_error but allows different bodies to contribute
    differently to the mean error. Useful for relaxing tracking on certain joints
    (e.g., wrists during manipulation).

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel on the angular error.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.
        body_weights: Dict mapping body name to weight multiplier. Bodies not listed
            default to 1.0. E.g. {"left_wrist_yaw_link": 0.1} to relax wrist tracking.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    if body_weights is not None:
        tracked_names = [command.cfg.body_names[i] for i in body_indexes]
        weights = torch.tensor(
            [body_weights.get(name, 1.0) for name in tracked_names],
            device=error.device,
            dtype=error.dtype,
        )
        weighted_error = (error * weights).sum(-1) / weights.sum()
    else:
        weighted_error = error.mean(-1)
    return torch.exp(-weighted_error / std**2)


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body linear velocity tracking reward using a Gaussian kernel.

    Encourages tracked body linear velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_lin_vel_w[:, tracked] - command.robot_body_lin_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Compute body angular velocity tracking reward using a Gaussian kernel.

    Encourages tracked body angular velocities to match the reference motion.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        std: Standard deviation for the Gaussian kernel.
        body_names: Subset of bodies to track. If None, uses all tracked bodies.

    Returns:
        Reward tensor of shape (num_envs,) in [0, 1].
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    tracked = _get_body_indexes(command, body_names)
    vel_diff = command.body_ang_vel_w[:, tracked] - command.robot_body_ang_vel_w[:, tracked]
    per_body_err = (vel_diff * vel_diff).sum(dim=-1)
    return torch.exp(-per_body_err.mean(dim=-1) / (std * std))


def feet_impact_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize downward foot speed at the instant of touchdown (foot slam).

    Fires ONLY on the tick a foot transitions air -> contact
    (``compute_first_contact``), penalizing its downward vertical velocity
    squared. Swing dynamics and steady stance are untouched, so this shapes
    the LANDING (decelerate before impact) without teaching hovering or
    hesitant steps -- deliberately NOT a dense "slow feet near ground"
    shaping, which would feed the known low-speed dead-band.

    Motivation (2026-07-16): recovery steps and dance moves land hard; on
    real hardware the impact spikes excite unmodeled dynamics.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor scoped to the feet, e.g.
            ``SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")``.
        asset_cfg: Robot asset scoped to the SAME bodies (for velocities).

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative
        weight; start small, ~2-5% of tracking-reward magnitude).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # [E, F] bool: feet that just made contact this step
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    asset = env.scene[asset_cfg.name]
    # downward speed at touchdown (positive = moving down): [E, F]
    down_speed = torch.relu(-asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2])
    impact = down_speed * first_contact.float()
    return (impact * impact).sum(dim=-1)


def feet_grf_overshoot_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 1.5,
) -> torch.Tensor:
    """Penalize vertical ground-reaction force above ``soft_ratio`` x weight.

    Complements ``feet_impact_vel_penalty``: that term shapes the approach
    (velocity at touchdown), this one shapes the LOADING right after contact
    -- a hard "catch" after a nominally slow touchdown still spikes force.
    Normalized by robot weight so the weight is robot-agnostic; forces below
    the soft threshold (single-stance + normal dynamics headroom) incur zero
    penalty.

    Args:
        env: The environment.
        sensor_cfg: Contact sensor scoped to the feet.
        asset_cfg: Robot asset (for total mass; default whole robot).
        soft_ratio: Per-foot force threshold as a multiple of total robot
            weight (1.5 = full bodyweight + 50% dynamic headroom).

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (negative weight).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # [E, F] vertical GRF on the feet
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    asset = env.scene[asset_cfg.name]
    # [E] robot weight (N)
    weight_n = asset.data.default_mass.sum(dim=-1).to(fz.device) * 9.81
    excess = torch.relu(fz - soft_ratio * weight_n.unsqueeze(-1)) / weight_n.unsqueeze(-1)
    return (excess * excess).sum(dim=-1)


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize excessive angular velocity on selected bodies with a deadzone.

    Discourage high-frequency jitter on small links (wrists, head) while allowing
    normal intentional motion within the threshold. Speeds below the threshold
    incur zero penalty.

    Args:
        env: The environment.
        command_name: Name of the tracking command term.
        threshold: Angular velocity deadzone (rad/s). No penalty below this.
        body_names: Bodies to penalize. If None, uses all tracked bodies.

    Returns:
        Penalty tensor of shape (num_envs,). Positive values (use negative weight).
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # [E, B, 3]
    ang_vel = command.robot_body_ang_vel_w[:, body_indexes]
    # magnitude per body: [E, B]
    speed = torch.linalg.norm(ang_vel, dim=-1)
    # deadzone then square: [E, B]
    excess = torch.relu(speed - threshold)
    penalty = (excess * excess).mean(dim=-1)
    return penalty


def tracking_dof_subset_error(
    env: ManagerBasedRLEnv, command_name: str, std: float,
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Joint-ANGLE tracking reward for a named subset of dofs (Gaussian kernel).

    Added 2026-08-15 (phase-2b wrist pin): axes whose angles move no tracked
    keypoint (wrist roll/pitch) are invisible to every Cartesian tracking term,
    so adapters drift them freely ("wandering wrist", see frozen-core ledger).
    This term makes such axes observable in dof space. Uses the command's own
    reference/actual joint buffers, honoring the dof-mismatch index map.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    cache_key = "_dof_subset_ids_" + ",".join(joint_names or [])
    ids = getattr(command, cache_key, None)
    if ids is None:
        asset_names = env.scene["robot"].data.joint_names
        robot_ids = [asset_names.index(n) for n in (joint_names or [])]
        if getattr(command, "has_dof_mismatch", False):
            body_idx = command.body_joint_indices.tolist()
            cmd_cols = [body_idx.index(i) for i in robot_ids]
        else:
            cmd_cols = robot_ids
        ids = (torch.as_tensor(robot_ids, device=env.device),
               torch.as_tensor(cmd_cols, device=env.device))
        setattr(command, cache_key, ids)
    robot_ids_t, cmd_cols_t = ids
    err = (command.joint_pos[:, cmd_cols_t]
           - command.robot_joint_pos[:, robot_ids_t]).abs().mean(dim=-1)
    return torch.exp(-err.square() / (std * std))
