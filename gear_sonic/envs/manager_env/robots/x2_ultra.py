from typing import Literal

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# Agibot X2 Ultra motor parameters.
# VENDOR DATASHEET 2026-08-23 (AgiBot hardware engineer, via the vendor liaison).
# Supersedes the H2-derived estimates that carried the "MUST be tuned once real
# motor datasheets are obtained" caveat.
#
# The sheet gives MOTOR-SIDE rotor inertia (confirmed by the vendor: "The inertia
# data is motor rotor only, not reflected inertia from load onto motor" -- the
# column header saying "Reflected" is wrong). Joint-side armature, which is what
# IsaacLab wants, is I_rotor * N^2:
#
#   PF52  I=8.3e-6   N=19.42856  -> 8.3e-6  * 377.47 = 3.132992e-3   (24 N-m)
#   PF70  I=2.21e-5  N=20.0      -> 2.21e-5 * 400.00 = 8.840000e-3   (36 nom / 45 actual)
#   PF90  I=6.06e-5  N=21.90582  -> 6.06e-5 * 479.86 = 2.907981e-2   (120 N-m)
#
# SINGLE SOURCE (2026-08-30): values are LOADED from the active plant yaml
# (config/robot_plant/, via plant_config — no isaaclab deps) instead of
# literal copies. The literals lived here for the whole legacy era while
# the datasheet sat unread; check_plant_consistency.py guards the still-
# duplicated MJCF. Era branches keep their frozen literals untouched.
try:
    from .plant_config import load_plant as _load_plant
except ImportError:
    # frozen_core_sonic_codec.harvest_x2_params() loads THIS file by path via
    # importlib.util.spec_from_file_location("_x2u", ...) -- no parent package,
    # so the relative import above raises. That path is the frozen-G1 (frozen-core)
    # lineage's ONLY way of reading the plant; the incumbent imports us as a
    # package module and never sees this. (2026-09-02: killed two frozen-G1
    # relaunches with "attempted relative import with no known parent package".)
    from gear_sonic.envs.manager_env.robots.plant_config import load_plant as _load_plant

_PLANT = _load_plant()   # default chain: $X2_PLANT > x2_ultra.yaml (the active plant)
ARMATURE_PF52 = _PLANT.armature["waist_pitch"]    # 24 N-m family
ARMATURE_PF70 = _PLANT.armature["ankle_pitch"]    # 36 N-m family
ARMATURE_PF90 = _PLANT.armature["hip"]            # 120 N-m family

# Vendor joint -> motor map (unit counts 10/6/9 reconcile exactly).
# PREVIOUS VALUES AND THEIR ERRORS, kept so the change is auditable:
#   hip/knee      was 0.025101925  -> PF90, 1.16x too low
#   waist_yaw     was 0.010177520  -> PF90, 2.86x too low  (had its own class;
#                                     it is the SAME motor as hip/knee)
#   waist_pitch/roll was 0.003609725 -> PF52, 1.15x too high
#   ankle_pitch   was 0.003609725  -> PF70, 2.45x too low  (was lumped with PF52)
#   ankle_roll    was 0.003609725  -> PF52, 1.15x too high
#   shoulder_p/r  was 0.003609725  -> PF70, 2.45x too low  (was lumped with PF52)
#   shoulder_yaw/elbow was 0.003609725 -> PF52, 1.15x too high
#   wrist_yaw     was 0.003609725  -> PF52, 1.15x too high
# ANKLE and SHOULDER each spanned TWO motor families; one constant could not
# serve both, which is where the 2.45x errors came from.
ARMATURE_HIP_KNEE = ARMATURE_PF90
ARMATURE_WAIST_YAW = ARMATURE_PF90
ARMATURE_WAIST_PR = ARMATURE_PF52
ARMATURE_ANKLE_PITCH = ARMATURE_PF70
ARMATURE_ANKLE_ROLL = ARMATURE_PF52
ARMATURE_SHOULDER_PR = ARMATURE_PF70       # shoulder pitch + roll
ARMATURE_SHOULDER_YAW_ELBOW = ARMATURE_PF52
ARMATURE_WRIST_YAW = ARMATURE_PF52
# NOT SUPPLIED by the vendor ("head and wrist joints have not been sorted out
# yet, will be supplemented later") -- still H2-derived estimates.
ARMATURE_WRIST = _PLANT.armature["wrist_pitch"]  # 4.8 N-m  [ESTIMATE, from yaml]
ARMATURE_HEAD = _PLANT.armature["head"]          # 2.6/0.6 N-m  [ESTIMATE, from yaml]

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * NATURAL_FREQ**2
STIFFNESS_WAIST_YAW = ARMATURE_WAIST_YAW * NATURAL_FREQ**2
STIFFNESS_WAIST_PR = ARMATURE_WAIST_PR * NATURAL_FREQ**2
STIFFNESS_ANKLE_PITCH = ARMATURE_ANKLE_PITCH * NATURAL_FREQ**2
STIFFNESS_ANKLE_ROLL = ARMATURE_ANKLE_ROLL * NATURAL_FREQ**2
STIFFNESS_SHOULDER_PR = ARMATURE_SHOULDER_PR * NATURAL_FREQ**2
STIFFNESS_SHOULDER_YAW_ELBOW = ARMATURE_SHOULDER_YAW_ELBOW * NATURAL_FREQ**2
STIFFNESS_WRIST = ARMATURE_WRIST * NATURAL_FREQ**2
STIFFNESS_WRIST_YAW = ARMATURE_WRIST_YAW * NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_KNEE * NATURAL_FREQ
DAMPING_WAIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_WAIST_YAW * NATURAL_FREQ
DAMPING_WAIST_PR = 2.0 * DAMPING_RATIO * ARMATURE_WAIST_PR * NATURAL_FREQ
DAMPING_ANKLE_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_PITCH * NATURAL_FREQ
DAMPING_ANKLE_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_ROLL * NATURAL_FREQ
DAMPING_SHOULDER_PR = 2.0 * DAMPING_RATIO * ARMATURE_SHOULDER_PR * NATURAL_FREQ
DAMPING_SHOULDER_YAW_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_SHOULDER_YAW_ELBOW * NATURAL_FREQ
DAMPING_WRIST = 2.0 * DAMPING_RATIO * ARMATURE_WRIST * NATURAL_FREQ
DAMPING_WRIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_WRIST_YAW * NATURAL_FREQ
DAMPING_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_HEAD * NATURAL_FREQ

# Body names in IsaacLab BFS traversal order (32 bodies including root pelvis).
# Verified against runtime robot.joint_names.  Isaac Lab sorts children
# alphabetically within each BFS level, so head_yaw_link ("h") precedes
# left/right_shoulder_pitch_link ("l"/"r") at the same depth.
X2_ULTRA_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "head_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]

# DOF index mappings between IsaacLab and MuJoCo orderings (31 DOF).
# isaaclab_to_mujoco[i] = MuJoCo index of the joint at IsaacLab index i.
X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10,
    30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12,
    17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16,
]

# Body index mappings between IsaacLab and MuJoCo orderings (32 bodies).
X2_ULTRA_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 30, 16, 23, 5,
    11, 31, 17, 24, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29,
]
X2_ULTRA_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 21, 3, 6, 9,
    13, 18, 22, 24, 26, 28, 30, 14, 19, 23, 25, 27, 29, 31, 12, 17,
]

X2_ULTRA_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": X2_ULTRA_ISAACLAB_JOINTS,
    # The motion_lib code uses these as numpy/torch gather indices:
    #   dof_il = dof_mj[mapping[i]]  →  mapping[i] must be the MJ source for IL position i
    # G1 follows this convention; X2 arrays are swapped to match.
    "isaaclab_to_mujoco_dof": X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF,
    "mujoco_to_isaaclab_dof": X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF,
    "isaaclab_to_mujoco_body": X2_ULTRA_MUJOCO_TO_ISAACLAB_BODY,
    "mujoco_to_isaaclab_body": X2_ULTRA_ISAACLAB_TO_MUJOCO_BODY,
}


def make_x2_ultra_cfg(
    actuator_regime: Literal["implicit", "explicit"] = "implicit",
    frictionloss: float = 0.0,
    foot: Literal["mesh", "sphere"] = "mesh",
    ankle_kp_scale: float = 1.0,
    waist_pr_effort: float | None = None,
) -> ArticulationCfg:
    """Build an X2 Ultra ``ArticulationCfg`` with optional MuJoCo-mirroring tweaks.

    All defaults reproduce the long-standing ``X2_ULTRA_CFG`` exactly so this
    factory is safe to use as a drop-in replacement. The non-default values are
    used by the ``isaaclab_mujoco_mirror`` diagnostic
    (``docs/source/user_guide/sim2sim_mujoco.md`` G18) to reproduce the
    MuJoCo deployment regime inside IsaacLab.

    Args:
        actuator_regime: ``"implicit"`` (default) keeps PD inside the PhysX
            implicit integrator (training-equivalent). ``"explicit"`` uses
            ``IdealPDActuatorCfg`` so PD runs as ``ctrl``-driven torque,
            mirroring MuJoCo's deploy regime (sim2sim_mujoco.md G5).
        frictionloss: Per-joint Coulomb friction in N.m. Default 0 (matches
            training). Set to 0.3 to mirror the MJCF ``frictionloss="0.3"``.
        foot: ``"mesh"`` (default) loads the standard URDF with mesh foot
            colliders. ``"sphere"`` loads ``x2_ultra_sphere_feet.urdf`` with
            12 sphere collisions per foot at the exact MJCF positions.
        ankle_kp_scale: Multiplier on ankle pitch/roll KP only. Default 1.0
            (training-equivalent). Set to 1.5 to mirror the deployed
            ``DEPLOYMENT_KP_SCALE["ankle"]`` baked into ``eval_x2_mujoco.py``
            (G16b).
        waist_pr_effort: Physics torque ceiling (N.m) for waist pitch/roll.
            **Default 24.0 since 2026-08-23** (vendor datasheet: waist
            pitch/roll are PF52, rated 24 N.m). Previously 36.0, itself down
            from 48.0 — see the history table in the actuator block below.

            **THIS DEFAULT ALSO SETS THE ACTION SCALE.**
            ``X2_ULTRA_ACTION_SCALE`` is derived from ``X2_ULTRA_CFG``, which
            is built from these defaults, so moving this number rescales the
            action space for every run AND every re-export (waist_pr, both
            ankle_pitch and both shoulder_pitch/roll -- 8 joints). Do not
            re-export or re-evaluate a checkpoint trained before 2026-08-23
            against this config: it would attach new scales to old weights
            with no error.

            UNRESOLVED: whether 24 is *rated* or *peak*. The vendor sheet
            distinguishes them for PF70 (nominal 36 / actual 45) but gives
            only a nominal for PF52, and the standing rule is to use SATURATION
            efforts in sim. If 24 turns out
            to be rated-only, this should rise.
            NOTE: intentionally does NOT re-derive ``X2_ULTRA_ACTION_SCALE``
            — action-space semantics stay source-compatible; only the
            physics ceiling changes.
    """

    if actuator_regime == "implicit":
        ActuatorCls = ImplicitActuatorCfg
    elif actuator_regime == "explicit":
        ActuatorCls = IdealPDActuatorCfg
    else:
        raise ValueError(f"Unknown actuator_regime={actuator_regime!r}")

    if foot == "mesh":
        urdf_name = "x2_ultra.urdf"
    elif foot == "sphere":
        urdf_name = "x2_ultra_sphere_feet.urdf"
    else:
        raise ValueError(f"Unknown foot={foot!r}")

    fric_kw = {"friction": frictionloss} if frictionloss > 0.0 else {}
    # (ankle stiffness is now applied per-joint in the 'feet' actuator,
    #  since ankle_pitch/roll are different motors -- see below.)

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=f"{ASSET_DIR}/robot_description/urdf/x2_ultra/{urdf_name}",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # X2 Ultra pelvis height is ~0.68m in MJCF default pose;
            # spawn slightly higher to avoid ground clipping with bent knees.
            pos=(0.0, 0.0, 0.78),
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_elbow_joint": -0.6,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_pitch_joint": 0.2,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ActuatorCls(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                # Effort ceilings come from the plant yaml (single source of
                # truth, config/robot_plant/x2_ultra.yaml) -- never literals.
                # Plant.effort_for raises KeyError if an entry is missing.
                effort_limit_sim={
                    ".*_hip_yaw_joint": _PLANT.effort_for("hip_yaw_joint"),
                    ".*_hip_roll_joint": _PLANT.effort_for("hip_roll_joint"),
                    ".*_hip_pitch_joint": _PLANT.effort_for("hip_pitch_joint"),
                    ".*_knee_joint": _PLANT.effort_for("knee_joint"),
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 11.936,
                    ".*_hip_roll_joint": 11.936,
                    ".*_hip_pitch_joint": 11.936,
                    ".*_knee_joint": 11.936,
                },
                stiffness={
                    ".*_hip_pitch_joint": STIFFNESS_HIP_KNEE,
                    ".*_hip_roll_joint": STIFFNESS_HIP_KNEE,
                    ".*_hip_yaw_joint": STIFFNESS_HIP_KNEE,
                    ".*_knee_joint": STIFFNESS_HIP_KNEE,
                },
                damping={
                    ".*_hip_pitch_joint": DAMPING_HIP_KNEE,
                    ".*_hip_roll_joint": DAMPING_HIP_KNEE,
                    ".*_hip_yaw_joint": DAMPING_HIP_KNEE,
                    ".*_knee_joint": DAMPING_HIP_KNEE,
                },
                armature={
                    ".*_hip_pitch_joint": ARMATURE_HIP_KNEE,
                    ".*_hip_roll_joint": ARMATURE_HIP_KNEE,
                    ".*_hip_yaw_joint": ARMATURE_HIP_KNEE,
                    ".*_knee_joint": ARMATURE_HIP_KNEE,
                },
                **fric_kw,
            ),
            "feet": ActuatorCls(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                # Ankle effort = AgiBot URDF ENFORCED control limits
                # (x2_31dof_hand.urdf on robot PC2: ankle_pitch 36, ankle_roll 24)
                # -- these match our sim URDF exactly. The MOTORS are physically
                # stronger (EtherCAT boot log: ankle pitch = motor_type 5 =
                # PFP-74-56 peak 60; ankle roll = motor_type 6 = PFP-59-60 peak 36)
                # but the robot's own model caps them at 36/24, so we match the
                # enforced limit for sim2real fidelity, NOT the raw motor peak.
                # Revisit ONLY if the deploy/MC torque-clamp is confirmed to pass
                # the full motor peak through to the joint.
                effort_limit_sim={
                    ".*_ankle_pitch_joint": _PLANT.effort_for("ankle_pitch_joint"),
                    ".*_ankle_roll_joint": _PLANT.effort_for("ankle_roll_joint"),
                },
                velocity_limit_sim={
                    ".*_ankle_pitch_joint": 13.088,
                    ".*_ankle_roll_joint": 15.077,
                },
                # SPLIT 2026-08-23: ankle_pitch is PF70, ankle_roll is PF52 --
                # two different motors that shared one constant, which made
                # ankle_pitch's armature 2.45x too low.
                stiffness={
                    ".*_ankle_pitch_joint": STIFFNESS_ANKLE_PITCH * float(ankle_kp_scale),
                    ".*_ankle_roll_joint": STIFFNESS_ANKLE_ROLL * float(ankle_kp_scale),
                },
                damping={
                    ".*_ankle_pitch_joint": DAMPING_ANKLE_PITCH,
                    ".*_ankle_roll_joint": DAMPING_ANKLE_ROLL,
                },
                armature={
                    ".*_ankle_pitch_joint": ARMATURE_ANKLE_PITCH,
                    ".*_ankle_roll_joint": ARMATURE_ANKLE_ROLL,
                },
                **fric_kw,
            ),
            "waist_yaw": ActuatorCls(
                joint_names_expr=["waist_yaw_joint"],
                effort_limit_sim=_PLANT.effort_for("waist_yaw_joint"),
                velocity_limit_sim=11.936,
                stiffness=STIFFNESS_WAIST_YAW,
                damping=DAMPING_WAIST_YAW,
                armature=ARMATURE_WAIST_YAW,
                **fric_kw,
            ),
            "waist": ActuatorCls(
                joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
                # Waist pitch/roll motor = motor_type 6 = PFP-59-60, physical peak
                # 36 N-m (EtherCAT boot log + datasheet). The AgiBot URDF caps
                # these at 48 -- ABOVE the motor's physical peak, i.e. phantom
                # torque the real motor cannot deliver. Sim inherited that 48 and
                # let the policy arrest forward lean with torque the hardware
                # can't produce -> "stable in sim, falls forward on real" (deploy
                # walking_recovery.yaml: waist_pitch resistance ran +45% vs MC).
                # Clamp to the true physical ceiling, 36.
                # SINGLE SOURCE (2026-09-02): None -> the plant yaml's ceiling
                # (24, vendor PF52). An explicit float is an intentional
                # override only -- there is NO literal default anywhere, so a
                # missing hydra key can never silently mean 36 again.
                effort_limit_sim=(
                    waist_pr_effort if waist_pr_effort is not None
                    else _PLANT.effort_for("waist_pitch_joint")
                ),
                velocity_limit_sim=13.088,
                stiffness=STIFFNESS_WAIST_PR,
                damping=DAMPING_WAIST_PR,
                armature=ARMATURE_WAIST_PR,
                **fric_kw,
            ),
            "head": ActuatorCls(
                joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
                effort_limit_sim={
                    "head_yaw_joint": 2.6,
                    "head_pitch_joint": 0.6,
                },
                velocity_limit_sim={
                    "head_yaw_joint": 6.019,
                    "head_pitch_joint": 6.28,
                },
                stiffness=STIFFNESS_HEAD,
                damping=DAMPING_HEAD,
                armature=ARMATURE_HEAD,
                **fric_kw,
            ),
            "arms": ActuatorCls(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_yaw_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_roll_joint",
                ],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 36.0,
                    ".*_shoulder_roll_joint": 36.0,
                    ".*_shoulder_yaw_joint": 24.0,
                    ".*_elbow_joint": 24.0,
                    ".*_wrist_yaw_joint": 24.0,
                    # Wrist pitch/roll = vendor joint PFP-41-50: peak torque 6 N-m,
                    # peak speed 200 rpm (=20.944 rad/s). Prior velocity_limit 4.188
                    # (~40 rpm) was a ~5x error and the binding constraint that made
                    # the wrist untrackable (see project_x2_wrist_investigation).
                    ".*_wrist_pitch_joint": 6.0,
                    ".*_wrist_roll_joint": 6.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 13.088,
                    ".*_shoulder_roll_joint": 13.088,
                    ".*_shoulder_yaw_joint": 15.077,
                    ".*_elbow_joint": 15.077,
                    ".*_wrist_yaw_joint": 15.077,
                    ".*_wrist_pitch_joint": 20.944,  # 200 rpm peak (PFP-41-50); was 4.188 (~40 rpm, wrong)
                    ".*_wrist_roll_joint": 20.944,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": STIFFNESS_SHOULDER_PR,
                    ".*_shoulder_roll_joint": STIFFNESS_SHOULDER_PR,
                    ".*_shoulder_yaw_joint": STIFFNESS_SHOULDER_YAW_ELBOW,
                    ".*_elbow_joint": STIFFNESS_SHOULDER_YAW_ELBOW,
                    ".*_wrist_yaw_joint": STIFFNESS_WRIST_YAW,
                    ".*_wrist_pitch_joint": STIFFNESS_WRIST,
                    ".*_wrist_roll_joint": STIFFNESS_WRIST,
                },
                damping={
                    ".*_shoulder_pitch_joint": DAMPING_SHOULDER_PR,
                    ".*_shoulder_roll_joint": DAMPING_SHOULDER_PR,
                    ".*_shoulder_yaw_joint": DAMPING_SHOULDER_YAW_ELBOW,
                    ".*_elbow_joint": DAMPING_SHOULDER_YAW_ELBOW,
                    ".*_wrist_yaw_joint": DAMPING_WRIST_YAW,
                    ".*_wrist_pitch_joint": DAMPING_WRIST,
                    ".*_wrist_roll_joint": DAMPING_WRIST,
                },
                armature={
                    ".*_shoulder_pitch_joint": ARMATURE_SHOULDER_PR,
                    ".*_shoulder_roll_joint": ARMATURE_SHOULDER_PR,
                    ".*_shoulder_yaw_joint": ARMATURE_SHOULDER_YAW_ELBOW,
                    ".*_elbow_joint": ARMATURE_SHOULDER_YAW_ELBOW,
                    ".*_wrist_yaw_joint": ARMATURE_WRIST_YAW,
                    ".*_wrist_pitch_joint": ARMATURE_WRIST,
                    ".*_wrist_roll_joint": ARMATURE_WRIST,
                },
                **fric_kw,
            ),
        },
    )


def plant_waist_pr_effort() -> float:
    """The waist pitch/roll ceiling the plant yaml declares (fails loudly).

    The one place the number lives. Callers that want to PRINT the effective
    ceiling (the env-cfg launch banner) read it from here, so what is logged
    is what is enforced.
    """
    return _PLANT.effort_for("waist_pitch_joint")


X2_ULTRA_CFG = make_x2_ultra_cfg()

# Action scale: effort_limit / stiffness * 0.25 (same formula as H2)
# Built from STIFFNESS_* constants directly — independent of ankle_kp_scale so
# the policy's [-1, 1] -> joint-target-offset mapping stays training-equivalent
# even when the deployed PD is bumped (mirrors eval_x2_mujoco.py G16b note).
X2_ULTRA_ACTION_SCALE = {}
for a in X2_ULTRA_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            X2_ULTRA_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
