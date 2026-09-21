"""Compose the X2 Ultra MJCF with two articulated OmniHand-2025 chains.

This module exists purely to support **rendering and inspection videos**. The
production training MJCF (``gear_sonic/data/assets/robot_description/mjcf/
x2_ultra.xml``) deliberately ends each arm at ``*_wrist_roll_link`` -- 31 DOFs
total -- to match the SONIC tracking decoder, the deploy ONNX, and the AimDK
ROS 2 HAL contract. Hand commands flow out-of-band via ``/aima/hal/joint/hand/
command`` on the real robot.

When we want to *visually* render what those hand commands would have produced
kinematically (for the M3 inspection video, for the M5 camera-plumbing
pipeline that bakes camera frames into the LeRobot dataset, etc.), we compose
the X2 spec with two ``omnihand_{left,right}.urdf`` chains and recreate the
six URDF ``<mimic>`` relationships per side as MJCF ``<equality joint>``
constraints (URDF mimic semantics are not preserved by MuJoCo's URDF parser).

The composition is performed entirely in memory through ``mujoco.MjSpec``;
no static composite XML file is written to disk. This keeps the renderer
asset path stable and avoids divergence between the body MJCF and the
augmented MJCF.

Public entry points
-------------------

``build_x2_with_omnihand_spec``
    Returns a compiled ``MjSpec`` with both hand chains attached.

``HAND_QPOS_LAYOUT``
    Static description of where each side's active / passive hand qpos slots
    end up in the augmented model, plus the prefix that ``MjSpec.attach``
    applies to joint names.

``apply_active_hand_qpos``
    Helper that writes a 10-DOF (per side) active-joint vector into
    ``data.qpos``. The MJCF equality constraints then snap the 6 passive
    DOFs into place on the next ``mj_forward`` call. Use this from the
    *kinematic* (``mj_forward``-only) renderer paths -- it bypasses
    actuator dynamics by stamping qpos directly.

``apply_active_hand_ctrl``
    Helper that writes a 10-DOF (per side) active-joint vector into
    ``data.ctrl``, targeting the position actuators added by
    :func:`_add_finger_actuators`. Use this from the *dynamic*
    (``mj_step``) bridge / sim paths -- the actuators produce smooth
    PD-tracking torques during integration, the equality constraints
    enforce passive = multiplier × active continuously, and there are
    no qpos discontinuities for MuJoCo's integrator to differentiate
    into spurious velocities.

The 10 active joints per side, in order, match exactly:

- ``OMNIHAND_FINGER_NAMES_PER_SIDE`` in
  ``gear_sonic/data/robot_model/supplemental_info/x2_ultra/x2_ultra_supplemental_info.py``
- The motor index order in the AgiBot OmniHand-2025 SDK Python API
- The 10-D layout of ``action.left_hand_joints`` / ``action.right_hand_joints``
  in our LeRobot v2.1 datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Static contract (must stay in sync with x2_ultra_supplemental_info.py)
# ────────────────────────────────────────────────────────────────────────────


# 10 active joints per side, in canonical OmniHand-2025 motor index order.
ACTIVE_FINGER_JOINTS: tuple[str, ...] = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)


# 6 passive (mimic) relationships per side encoded in the SDK URDF as
#   <mimic joint="<active>" multiplier="m" offset="0"/>.
# We re-create each as an MJCF equality constraint: passive = active * mult.
@dataclass(frozen=True)
class _MimicRule:
    passive: str
    active: str
    multiplier: float


PASSIVE_MIMIC_RULES: tuple[_MimicRule, ...] = (
    _MimicRule(passive="thumb_pip",  active="thumb_mcp", multiplier=1.33),
    _MimicRule(passive="thumb_dip",  active="thumb_mcp", multiplier=1.30),
    _MimicRule(passive="index_dip",  active="index_pip", multiplier=1.097),
    _MimicRule(passive="middle_dip", active="middle_pip", multiplier=1.097),
    _MimicRule(passive="ring_dip",   active="ring_pip",  multiplier=1.097),
    _MimicRule(passive="pinky_dip",  active="pinky_pip", multiplier=1.097),
)


# OmniHand joints that *exist* in the upstream URDF as free hinges
# but are not part of the actuated kinematics by hardware design.
# ``middle_abad`` is the canonical example: the middle finger has a
# single active DOF (``middle_pip``) -- unlike index/ring/pinky which
# also have ``*_abad`` for sideways spread -- but the URDF still ships
# a ``middle_abad`` hinge declaration (probably copy-pasted from the
# other fingers). Without intervention that joint:
#   1. has no actuator (not in ``ACTIVE_FINGER_JOINTS``)
#   2. has no equality coupling it to anything (not in ``PASSIVE_MIMIC_RULES``)
#   3. has no MJCF range -- ``model.jnt_range`` reports (0, 0) so
#      MuJoCo doesn't even clamp it
# so any tiny numerical perturbation from the equality solver
# accumulates as runaway angular velocity around its sideways axis
# and the visual mesh punches through the palm collision shell. We
# pin each entry below to qpos=0 with a one-coefficient equality
# (``passive - 0 = 0``), which is the MJCF analogue of welding the
# joint shut without changing the model topology that the SDK URDF
# expects.
LOCKED_PASSIVE_JOINTS: tuple[str, ...] = (
    "middle_abad",
)


# Default URDF root names (produced by ``MjSpec.from_file`` for the SDK URDFs).
# In the upstream right URDF active joint names look like ``R_thumb_roll_joint``
# (capitalised side prefix). After ``MjSpec.attach(prefix=...)`` they become
# ``<prefix>R_thumb_roll_joint``. We use ``left_/right_`` as our prefix so
# attached joint names land at ``left_R_thumb_roll_joint`` / ``right_L_*``,
# which is unambiguous when we later look them up.
@dataclass(frozen=True)
class _SideConfig:
    side: str  # "left" or "right"
    sdk_prefix: str  # "L_" or "R_" (upstream URDF naming)
    urdf_path: Path
    parent_body: str  # X2 body to attach the SDK base_link onto
    mount_pos: tuple[float, float, float]
    mount_quat_wxyz: tuple[float, float, float, float]


# ────────────────────────────────────────────────────────────────────────────
# Mount transforms
# ────────────────────────────────────────────────────────────────────────────


# Mount transforms tuned against the X2 wrist_roll mesh extents (see
# ``gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes/`` STL bbox
# probe -- the wrist_roll mesh extends from z=0 down to z≈-0.182m in the body
# local frame, i.e. the wrist tip points along -Z).  The OmniHand palm has its
# origin at the cuff-mounting cylinder with fingers extending along +Z. So the
# palm origin should land at z=-0.182m in the wrist_roll body frame, with the
# palm rotated 180° about the local Y axis so its +Z aligns with the wrist's
# -Z (and the SDK's "thumb-out" Y direction is preserved across hands).
#
# These defaults are visual-only; the renderer accepts overrides via
# ``mount_offset_z`` / extra rotation knobs. Tune by re-rendering the
# ego-view inspection video.

# Mount rotation: per-side composition of two rotations applied in body-
# frame order.
#
# Step 1 (BOTH sides): 180° about wrist_roll local Y. This flips the
# OmniHand palm's +Z (fingertip direction) from wrist +Z to wrist -Z
# (away from the elbow), so fingers extend along the forearm.
#
# Step 2: roll about wrist_roll local Z (the wrist long axis). The X2
# left and right wrist_roll bodies are mirrored about the body
# centerline -- so the SAME world-frame "palm down" orientation
# corresponds to OPPOSITE local-Z rolls on the two sides:
#
# * Right wrist (``_RIGHT_MOUNT_QUAT_WXYZ``): +90° about local Z.
#   With the SDK's right palm placing thumb at palm +Y, this swings
#   the thumb to wrist +X (the right wrist's medial side toward the
#   body centerline) and leaves the back of the hand facing up
#   (palm-down).
#
# * Left wrist (``_LEFT_MOUNT_QUAT_WXYZ``): -90° about local Z.
#   The SDK's left palm places thumb at palm -Y (mirrored from the
#   right). The -90° roll swings the thumb to the LEFT wrist's medial
#   side too, and -- crucially -- inverts the palm-up that the right-
#   wrist roll would otherwise produce on the mirrored body.
#
# Without the sign flip, both hands look anatomically correct
# *individually*, but the left palm faces up while the right palm
# faces down, which breaks the piano-down posture rubric.
#
# To re-derive::
#
#     from scipy.spatial.transform import Rotation as R
#     base = R.from_quat([0, 1, 0, 0])  # 180° about Y, xyzw
#     right = (R.from_euler("z", +90, degrees=True) * base).as_quat()
#     left  = (R.from_euler("z", -90, degrees=True) * base).as_quat()
#     # convert each xyzw -> wxyz
_RIGHT_MOUNT_QUAT_WXYZ: tuple[float, float, float, float] = (
    0.0, 0.7071067811865476, 0.7071067811865476, 0.0
)
_LEFT_MOUNT_QUAT_WXYZ: tuple[float, float, float, float] = (
    0.0, 0.7071067811865476, -0.7071067811865476, 0.0
)
# Backwards-compatibility alias retained for the M3.5 acceptance test
# that pinned a single "default" mount quaternion. Now bound to the
# right-side value because that was the original published convention.
_DEFAULT_MOUNT_QUAT_WXYZ: tuple[float, float, float, float] = _RIGHT_MOUNT_QUAT_WXYZ

# Mount Z for the OmniHand palm in the wrist_roll body local frame, in meters.
#
# This MUST match the cut_z used by ``clip_x2_wrist_for_omnihand.py`` -- the
# OmniHand palm cuff (radius 0.028 m) mates seamlessly with the clipped X2
# wrist neck (radius ~0.029 m at z ≈ -0.055 m). Mounting deeper than the cut
# leaves a gap where the palm cuff is suspended below the trimmed wrist
# tube; mounting shallower clips the palm cuff into the wrist roll motor
# casing.
_DEFAULT_MOUNT_Z = -0.055


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gear_sonic" / "data" / "assets" / "robot_description"


def _default_side_configs(asset_root: Path | None = None) -> tuple[_SideConfig, _SideConfig]:
    if asset_root is None:
        # gear_sonic/scripts/.. -> repo root
        asset_root = Path(__file__).resolve().parents[2] / "gear_sonic" / "data" / "assets" / "robot_description"
    return (
        _SideConfig(
            side="left",
            sdk_prefix="L_",
            urdf_path=asset_root / "omnihand" / "omnihand_left.urdf",
            parent_body="left_wrist_roll_link",
            mount_pos=(0.0, 0.0, _DEFAULT_MOUNT_Z),
            mount_quat_wxyz=_LEFT_MOUNT_QUAT_WXYZ,
        ),
        _SideConfig(
            side="right",
            sdk_prefix="R_",
            urdf_path=asset_root / "omnihand" / "omnihand_right.urdf",
            parent_body="right_wrist_roll_link",
            mount_pos=(0.0, 0.0, _DEFAULT_MOUNT_Z),
            mount_quat_wxyz=_RIGHT_MOUNT_QUAT_WXYZ,
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
# Public layout description
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class HandQposLayout:
    """Maps the 10 active OmniHand DOFs (per side) into augmented ``qpos``.

    ``active_qposadr[side][i]`` is the qpos index of the *i*-th active joint
    (where ``i`` indexes ``ACTIVE_FINGER_JOINTS``). ``passive_qposadr[side]``
    is keyed by passive joint short name (e.g. ``"thumb_pip"``).

    The layout is a function of the augmented model only -- it is computed
    once at compose time and surfaced so callers can write hand commands
    without re-resolving joint names every frame.
    """

    body_qposadr_start: int
    """First qpos index occupied by the X2 body floating base (always 0)."""

    body_njnt: int
    """Number of body joints (free joint contributes 7, hinges contribute 1)."""

    active_qposadr: Mapping[str, list[int]]
    """``side -> [qposadr_per_active_joint_in_canonical_order]``."""

    passive_qposadr: Mapping[str, Mapping[str, int]]
    """``side -> {passive_joint_short_name -> qposadr}``."""

    full_joint_names: Mapping[str, list[str]]
    """``side -> [<prefix>L/R_<short>_joint, ...]`` after ``attach``."""

    active_actadr: Mapping[str, list[int]] = field(default_factory=dict)
    """``side -> [actuator_id_per_active_joint_in_canonical_order]``.

    Populated by :func:`_add_finger_actuators` when the augmented spec
    is built. Empty for legacy callers that compose without finger
    actuators (kinematic-only renderer path); :func:`apply_active_hand_ctrl`
    will raise a clear error in that case.
    """


# ────────────────────────────────────────────────────────────────────────────
# Composition
# ────────────────────────────────────────────────────────────────────────────


def _attach_hand_spec(
    parent: mujoco.MjSpec,
    side: _SideConfig,
) -> None:
    """Attach a single ``omnihand_{side}`` URDF onto the given X2 wrist body."""
    child = mujoco.MjSpec.from_file(str(side.urdf_path))
    # The vendored omnihand URDF declares ``<compiler meshdir="meshes"/>`` AND
    # references each mesh as ``filename="meshes/foo.STL"``. Older mujoco
    # (<= 3.5) silently strips the redundant ``meshes/`` prefix from each
    # filename and resolves to ``<urdf_dir>/meshes/foo.STL``; newer mujoco
    # (>= 3.7) literally joins the two and tries to open
    # ``<urdf_dir>/meshes/meshes/foo.STL``, which doesn't exist.
    #
    # Try the original (3.5-friendly) behaviour first. If it fails with the
    # "meshes/meshes" path, force ``meshdir`` to the URDF dir and retry. This
    # keeps the M5 acceptance gate (which runs against mujoco 3.5 in .venv)
    # green while letting the live VLA bridge (env_isaaclab @ mujoco 3.7) work.
    try:
        child.compile()  # resolves mesh files relative to the URDF dir
    except ValueError as exc:
        if "meshes/meshes/" not in str(exc):
            raise
        child.meshdir = ""
        child.compile()

    # Find the X2 wrist body and add a frame at the desired mount pose.
    parent_body = parent.body(side.parent_body)
    frame = parent_body.add_frame(
        name=f"{side.side}_omnihand_mount",
        pos=list(side.mount_pos),
        quat=list(side.mount_quat_wxyz),
    )

    # ``MjSpec.attach`` mounts the child spec at the given frame and prepends
    # ``prefix`` to every named element. The SDK URDF uses 'L_' / 'R_' prefixes
    # already so we add a 'left_' / 'right_' outer prefix for unambiguous
    # disambiguation with X2's own ``left_*`` / ``right_*`` body names.
    parent.attach(child, prefix=f"{side.side}_", frame=frame)


def _add_mimic_equalities(spec: mujoco.MjSpec) -> None:
    """Recreate URDF ``<mimic>`` rules as MJCF ``<equality joint>`` constraints.

    MuJoCo's ``equality`` of type ``mjEQ_JOINT`` evaluates
    ``q1 - polycoef[0] - polycoef[1]*q2 - polycoef[2]*q2^2 - ...``
    where ``q1`` is ``name1`` and ``q2`` is ``name2``. Setting
    ``polycoef = [0, multiplier, 0, 0, 0]`` enforces ``q1 = multiplier * q2``.
    """
    for side_cfg in _default_side_configs():
        for rule in PASSIVE_MIMIC_RULES:
            j_passive = f"{side_cfg.side}_{side_cfg.sdk_prefix}{rule.passive}_joint"
            j_active = f"{side_cfg.side}_{side_cfg.sdk_prefix}{rule.active}_joint"
            # MjSpec.add_equality requires an 11-element data buffer:
            # for mjEQ_JOINT, the first 5 slots are polycoef[0..4], slots
            # 5-9 are unused, slot 10 is a default scaler (1.0).
            # polycoef = [0, multiplier, 0, 0, 0] enforces
            #     q_passive - 0 - multiplier * q_active = 0  ==>  passive = mult * active
            spec.add_equality(
                name=f"{side_cfg.side}_mimic_{rule.passive}",
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=j_passive,
                name2=j_active,
                data=[0.0, float(rule.multiplier), 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )
        # Pin every vestigial-by-design joint to qpos=0. mjEQ_JOINT
        # without ``name2`` evaluates ``q1 - polycoef[0] = 0`` (the
        # higher-order coefficients drop out because there's no q2
        # to multiply by). The constant scaler in slot 10 stays 1.0.
        # This is the equality-constraint analogue of welding the
        # joint shut, and is preferred over deleting the joint from
        # the spec because the upstream SDK URDF / kinematic chain
        # still references the body it lives on -- removing the
        # joint risks breaking attachment or downstream callers
        # that introspect the model by joint name.
        for short in LOCKED_PASSIVE_JOINTS:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{short}_joint"
            spec.add_equality(
                name=f"{side_cfg.side}_lock_{short}",
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=jname,
                data=[0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )


def _swap_wrist_roll_visual_to_clipped(
    spec: mujoco.MjSpec,
    asset_root: Path,
) -> None:
    """Swap each wrist_roll body's *visual* geom to the clipped (no-fist) mesh.

    The X2 ``*_wrist_roll_link.STL`` mesh bakes in a static dummy "fist" stub
    at its tip that overlaps the OmniHand palm. ``clip_x2_wrist_for_omnihand``
    produces a trimmed mesh ending at the wrist neck (z ≈ ``_DEFAULT_MOUNT_Z``);
    this function registers that trimmed mesh as a new asset and points the
    wrist_roll body's visual geom at it. The collision geom keeps the
    original (full) mesh so contact behaviour is unchanged.

    Visual vs collision are distinguished by MJCF group: ``group=1`` is the
    visual class, ``group=3`` is the collision class. The X2 MJCF authors
    each ``*_wrist_roll_link`` body with two geoms following that convention.
    """
    clip_dir = asset_root / "omnihand" / "meshes"
    for side in ("left", "right"):
        clipped_stl = clip_dir / f"{side}_wrist_roll_clipped_link.STL"
        if not clipped_stl.is_file():
            raise FileNotFoundError(
                f"Clipped wrist mesh not found: {clipped_stl}. "
                "Run gear_sonic/scripts/clip_x2_wrist_for_omnihand.py first."
            )

        # Register a new mesh asset with an absolute path so we don't have
        # to fight the X2 MJCF's compiler.meshdir relative-path resolution.
        new_mesh_name = f"{side}_wrist_roll_clipped_link"
        spec.add_mesh(
            name=new_mesh_name,
            file=str(clipped_stl.resolve()),
        )

        body = spec.body(f"{side}_wrist_roll_link")
        if body is None:
            raise RuntimeError(f"X2 body {side}_wrist_roll_link not found in spec")

        swapped = False
        for geom in body.geoms:
            if geom.group == 1 and geom.meshname == f"{side}_wrist_roll_link":
                geom.meshname = new_mesh_name
                swapped = True
        if not swapped:
            raise RuntimeError(
                f"Could not find wrist_roll visual geom on {side}_wrist_roll_link "
                "to swap; X2 MJCF layout may have changed."
            )


def _disable_hand_collisions(spec: mujoco.MjSpec, side_cfgs: tuple[_SideConfig, ...]) -> None:
    """Mark all attached hand geoms as visual-only (contype=0, conaffinity=0).

    The renderer is purely kinematic -- we never want hand fingers to collide
    with the X2 body, the floor, or each other. URDF→MJCF conversion encodes
    the collision boxes/cylinders from the SDK URDF as separate geoms with
    ``contype=1, conaffinity=1`` by default; zero them out so the dynamics
    engine ignores them.

    URDF-derived geoms have empty names -- name-based filtering does NOT
    work. We instead walk the spec's body tree starting at each side's
    attach point (``{side}_wrist_roll_link``) and recurse into every
    OmniHand descendant body, zeroing every geom we find.
    """
    for side_cfg in side_cfgs:
        root = spec.body(side_cfg.parent_body)
        if root is None:
            raise RuntimeError(
                f"_disable_hand_collisions: {side_cfg.parent_body} not found"
            )
        # Walk only descendants newly attached by the OmniHand chain. The
        # wrist_roll_link itself owns the X2 wrist visual + collision geoms;
        # those must keep their original contype/conaffinity so the X2
        # collision model is unchanged.  Traverse children only.
        stack: list = list(root.bodies)
        while stack:
            body = stack.pop()
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0
            stack.extend(body.bodies)


# ── Hand contact-filter classes ─────────────────────────────────────────
#
# When ``disable_hand_collisions=False`` (the robocasa scene path), the
# URDF-derived collision primitives on the OmniHand inherit MuJoCo's
# default ``contype=1, conaffinity=1`` and start to collide with each
# other (palm vs. fingertip when curled, neighboring fingertips, the
# thumb_mcp box vs. the thumb_pip box, etc.). The constraint forces
# generated by those overlapping primitives routinely exceed the
# position-actuator force limit (``_FINGER_FMAX = 3 Nm``) and pin the
# fingers half-open: a teleop "fully close" command never reaches the
# commanded curl because the actuator can't push past the self-contact.
#
# MuJoCo's contype/conaffinity bitmasks let us solve this without
# touching any scene geom: we put every hand collision geom on its own
# advertised channel (``contype = _HAND_COLLISION_CHANNEL``) but only
# listen to the default world channel (``conaffinity = _WORLD_CHANNEL``).
# The contact rule ``(c1 & a2) || (c2 & a1)`` then evaluates to:
#
#   hand vs. world (cube, bowl, table, floor, X2 body, …)
#                  : (2 & 1) | (1 & 1) = 0 | 1 = 1   -> contact ✓
#   hand vs. hand  : (2 & 1) | (2 & 1) = 0 | 0 = 0   -> filtered
#
# Trade-off: this filters BOTH palm-vs-finger AND finger-vs-finger
# uniformly. The palm-vs-finger filter is strictly required (that's what
# was pinning the fingers half-open). The finger-vs-finger filter is a
# side-effect; in practice when fingers curl, adjacent fingertips don't
# usually overlap, but the thumb crossing over the index finger does.
# If a future task needs finger-vs-finger contacts (e.g. realistic
# precision pinch), split palm and finger geoms onto different
# advertised channels and have fingers OR the finger channel into their
# own conaffinity. For grasp-and-place teleop the simpler uniform
# (2, 1) on every hand geom is sufficient and cheap.
#
# Critically this scheme leaves EVERY scene geom (cube, bowl, table,
# floor, X2 body) at its default (1, 1) -- we don't need to patch
# anything outside the OmniHand subtree, which keeps the scene-build
# step minimal and means new tasks "just work" without explicit opt-in.
#
# Visual-only geoms already carry ``contype=0, conaffinity=0`` from the
# URDF import (group=1) -- skip those so we don't accidentally promote
# a visual mesh into the active contact set.
_HAND_COLLISION_CHANNEL: int = 2
_WORLD_CHANNEL: int = 1

# Backwards-compat aliases: kept so callers (and old tests) that import
# the original symbols don't break, but new code should prefer the
# *_CHANNEL names which describe the bitmask role rather than a "class".
_HAND_COLLISION_CLASS: int = _HAND_COLLISION_CHANNEL
_WORLD_COLLISION_CLASS: int = _WORLD_CHANNEL


def _filter_hand_self_collisions(
    spec: mujoco.MjSpec, side_cfgs: tuple[_SideConfig, ...]
) -> None:
    """Re-class hand collision geoms so finger-vs-world fires but finger-vs-hand doesn't.

    See the module-level constants ``_HAND_COLLISION_CHANNEL`` and
    ``_WORLD_CHANNEL`` for the contact-filter math. Walks the body tree
    rooted at each ``{side}_wrist_roll_link`` and rewrites every
    OmniHand-derived collision geom to
    ``contype=_HAND_COLLISION_CHANNEL=2``,
    ``conaffinity=_WORLD_CHANNEL=1``; only visuals (already at
    ``contype=0, conaffinity=0`` from the URDF import) are skipped.

    Concretely after this filter:

    * Fingers / palm vs. cube, bowl, table, floor, X2 body: collide ✓
    * Fingers / palm vs. each other (within or across the two hands): no
      contact (uniform self-collision filter -- this is what stops the
      "fingers won't fully close" bug where palm-vs-fingertip overlaps
      pinned the curl half-open).

    Two-tier walk:

    1. The wrist body ITSELF: the OmniHand palm primitives (cylinder
       at z≈-0.062 and box at z≈-0.112) are attached via the
       ``{side}_omnihand_mount`` frame and end up directly under the
       wrist body, NOT under a child body. We must include them in
       the filter -- otherwise palm-vs-finger contacts pin the fingers
       open before they can curl in toward the palm.

       The wrist body also owns X2's own wrist_roll collision, which
       MUST keep its (contype=1, conaffinity=1) so the X2 collision
       model stays unchanged. We discriminate by primitive type: X2's
       wrist collision is a MESH (type=mjGEOM_MESH, ``meshname`` set
       to ``{side}_wrist_roll_link``), while OmniHand palm primitives
       are non-mesh shapes (cylinder / box) with empty ``meshname``.

    2. Every CHILD body recursively (the SDK URDF's thumb_*, *_pip,
       *_dip chains): every collision geom here belongs to OmniHand
       fingers and is uniformly re-classed.
    """
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    for side_cfg in side_cfgs:
        root = spec.body(side_cfg.parent_body)
        if root is None:
            raise RuntimeError(
                f"_filter_hand_self_collisions: {side_cfg.parent_body} not found"
            )

        # ── Tier 1: wrist body's own geoms (OmniHand palm primitives only). ──
        for geom in root.geoms:
            if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
                continue  # visual-only, ignore
            if int(geom.type) == mesh_type:
                # Mesh geoms in the wrist body belong to X2 (the wrist
                # collision mesh). Leave their contact behaviour
                # untouched so the X2 collision model is unchanged.
                continue
            geom.contype = _HAND_COLLISION_CHANNEL
            geom.conaffinity = _WORLD_CHANNEL

        # ── Tier 2: descendant bodies (every OmniHand finger chain). ──
        stack: list = list(root.bodies)
        while stack:
            body = stack.pop()
            for geom in body.geoms:
                # Skip visuals already disabled by the URDF import.
                # Promoting them into the collision set would be a
                # regression (extra contact pairs, slower broadphase,
                # and visually-only meshes participating in dynamics).
                if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
                    continue
                geom.contype = _HAND_COLLISION_CHANNEL
                geom.conaffinity = _WORLD_CHANNEL
            stack.extend(body.bodies)


# ────────────────────────────────────────────────────────────────────────────
# Finger position actuators
# ────────────────────────────────────────────────────────────────────────────


# Default PD gains for the 10 active finger position actuators.
#
# OmniHand-2025 finger links are small (~30-50 g per phalanx) and the MJCF
# equality constraints already do most of the kinematic work projecting
# passive joints onto active. We pick gains stiff enough to track
# operator-driven setpoints (≤ 5 rad/s slew) within ~10 ms but soft
# enough that brief zeros at boot (before pyzmq SUB connects) don't fire
# multi-Newton transients into the augmented MJCF.
#
# Empirically we tune by eye on a 1 kHz mj_step loop: gains lower than
# kp≈10 lag perceptibly during sustained finger curls; gains higher than
# kp≈40 cause visible vibration on rest-pose-to-zero corrections at
# t=0. The values here are the midpoint that smoke-tests cleanly.
_FINGER_KP = 20.0  # Nm/rad  (position gain)
_FINGER_KV = 1.0   # Nm/(rad/s) (damping)
_FINGER_FMAX = 3.0  # Nm (per-joint torque limit)


# ── Finger inertial stabilization (armature + passive damping) ─────────
#
# The OmniHand URDF link inertias are extremely small -- the PIP segments
# are ~1.2e-5 kg·m^2 and the DIP segments are ~1.5e-6 kg·m^2 (lighter
# than a paper clip). With the soft mimic equality solver
# (``solimp[0] = 0.9`` by default in MuJoCo) injecting tiny but non-zero
# constraint forces every step at 1 kHz, and no passive damping at the
# joint level to absorb that energy, the integrator turns those forces
# into >70° of finger-joint drift in 5 seconds even when the position
# actuators are commanded to *hold* the rest pose. The fingers visually
# wiggle nonstop in the SONIC bridge sim (``mj_step`` path), while the
# kinematic renderer (``mj_forward`` only, no integration) looks fine.
#
# The fix is purely numerical, not physical: add a small ``armature``
# (rotor inertia at the joint) and a small passive ``damping`` to every
# OmniHand finger joint. Armature inflates the diagonal of the mass
# matrix without changing the link inertia (so pose / FK / contacts are
# unchanged), making the joints insensitive to small constraint impulses.
# Passive damping bleeds off any residual oscillation between actuator
# updates.
#
# Sweep results (5 s of mj_step with ctrl frozen at rest pose, max over
# all 32 non-locked finger joints; see PR diagnosis for the script):
#
# +------------------------------+--------+----------+---------+
# | config                       | drift° | jitter   | max_dev |
# |                              |        | (deg/s)  |         |
# +==============================+========+==========+=========+
# | CURRENT (zero, zero)         |   75.5 |   3393   |  100.5  |
# | armature=1e-3 only           |    0.9 |     13.0 |   10.2  |
# | armature=1e-3 + damping=0.05 |    0.9 |      3.6 |    2.6  |
# +------------------------------+--------+----------+---------+
#
# armature alone gives a 90× drift / 260× jitter improvement; adding a
# small passive damping further smooths the residual oscillation to
# < 0.1° per second of drift. Higher armature (1e-2) trades 1° more
# steady-state drift for marginally less jitter -- not worth it.
#
# Tracking response with these values: at kp=20, the closed-loop time
# constant is ~50 ms (well under the bridge's 1 kHz integration step
# and indistinguishable at the 50 Hz teleop command rate). Human finger
# motion lives at < 5 Hz, so the operator will not perceive the added
# inertia; the SONIC body policy never sees these joints anyway.
_FINGER_ARMATURE = 1e-3   # kg·m^2 added to joint diagonal (rotor inertia)
_FINGER_DAMPING = 0.05    # Nm·s/rad passive joint damping


def _apply_finger_inertial_stabilization(
    model: mujoco.MjModel, side_cfgs: tuple[_SideConfig, ...]
) -> None:
    """Set ``armature`` + ``damping`` on every OmniHand finger DOF.

    Walks every joint declared by every side's hand chain (active +
    mimic-passive + locked-passive) and sets a small rotor inertia
    plus passive damping on its underlying DOF. See the docstring on
    ``_FINGER_ARMATURE`` for the empirical motivation.

    Implementation note: we patch ``model.dof_armature`` /
    ``model.dof_damping`` on the *compiled* ``MjModel`` rather than
    going through the ``MjSpec.joint(name).damping`` setter. The
    ``MjsJoint.damping`` Python attribute is scalar-typed in MuJoCo
    3.5.x but became a 3-element NDArray in 3.7.x (per-DOF for
    compound joints), and our deploy Docker image and host venv are
    pinned to different minor versions. The compiled-model arrays
    are scalar-per-DOF on every MuJoCo version, so this path is
    forward / backward compatible.

    Skips body / wrist joints -- those are SONIC-policy-controlled and
    must not be perturbed (the policy was trained on their original
    mass-matrix conditioning).
    """
    finger_short_names: set[str] = set()
    for short in ACTIVE_FINGER_JOINTS:
        finger_short_names.add(short)
    for rule in PASSIVE_MIMIC_RULES:
        finger_short_names.add(rule.passive)
    for short in LOCKED_PASSIVE_JOINTS:
        finger_short_names.add(short)

    for side_cfg in side_cfgs:
        for short in finger_short_names:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{short}_joint"
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, jname
            )
            if jid < 0:
                # Joint not present on this side (e.g. a future passive
                # rule we haven't built yet). Silently skip.
                continue
            dofadr = int(model.jnt_dofadr[jid])
            if dofadr < 0:
                continue
            model.dof_armature[dofadr] = _FINGER_ARMATURE
            model.dof_damping[dofadr] = _FINGER_DAMPING


def _add_finger_actuators(
    spec: mujoco.MjSpec, side_cfgs: tuple[_SideConfig, ...]
) -> None:
    """Add ``<position>``-class actuators on the 10 active joints per side.

    The augmented MJCF has equality constraints for the 6 mimic relationships
    per side but no actuators on the active fingers -- the offline renderer
    drove fingers by stamping ``mj_data.qpos`` directly and calling
    ``mj_forward``.

    That kinematic strategy explodes under ``mj_step`` (the SONIC C++
    deploy bridge's hot path): each step writes new qpos, MuJoCo's
    integrator differences it into a huge implicit qvel, the equality
    solver fights to enforce passive=mult×active against those
    infinities, and a finger DOF goes QACC NaN within 0.5s of sim time.

    Position actuators close that loop properly: the bridge writes
    setpoints to ``mj_data.ctrl[active_actadr]``, MuJoCo's integrator
    handles the dynamics smoothly via PD torque
    ``tau = kp*(ctrl - q) - kv*dq``, and the equality constraints
    operate on a continuous trajectory the way they were designed to.
    Hardware semantics (real OmniHand motors are torque-driven via PD)
    are also preserved -- no sim-to-train gap is opened.

    The MJCF schema for a position actuator is::

        <position joint="J" kp="20" kv="1" forcerange="-3 3" inheritrange="1"/>

    which lowers to ``MjsActuator`` with::

        gaintype = mjGAIN_FIXED   gainprm = [kp, ...]
        biastype = mjBIAS_AFFINE  biasprm = [0, -kp, -kv, ...]
        trntype  = mjTRN_JOINT    target  = "J"
        ctrllimited = 1; ctrlrange inherited from joint range
        forcelimited = 1; forcerange = [-fmax, fmax]
    """
    for side_cfg in side_cfgs:
        for short in ACTIVE_FINGER_JOINTS:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{short}_joint"
            spec.add_actuator(
                name=f"pos_{jname}",
                trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
                target=jname,
                gaintype=int(mujoco.mjtGain.mjGAIN_FIXED),
                gainprm=[_FINGER_KP, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                biastype=int(mujoco.mjtBias.mjBIAS_AFFINE),
                biasprm=[0, -_FINGER_KP, -_FINGER_KV, 0, 0, 0, 0, 0, 0, 0],
                # ``inheritrange=1`` tells MuJoCo to use the joint's own
                # range (set from the URDF) as the actuator's ctrlrange,
                # so operator-driven setpoints get hard-clamped to the
                # mechanically valid envelope before PD kicks in.
                ctrllimited=1,
                inheritrange=1.0,
                forcelimited=1,
                forcerange=[-_FINGER_FMAX, _FINGER_FMAX],
            )


def build_x2_with_omnihand_spec(
    *,
    asset_root: Path | None = None,
    mount_offset_z: float | None = None,
    disable_hand_collisions: bool = True,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, HandQposLayout]:
    """Compose the X2 + OmniHand augmented model.

    Parameters
    ----------
    asset_root:
        Override the ``robot_description`` directory (defaults to the
        repo-vendored path).
    mount_offset_z:
        Override the wrist→palm Z offset in the wrist_roll local frame
        (defaults to ``-0.182m`` -- the bottom of the wrist_roll mesh bbox).
    disable_hand_collisions:
        If True (default), every OmniHand finger geom is stripped of its
        ``contype`` / ``conaffinity`` after composition so the dynamics
        engine ignores fingertip-vs-anything contacts. This matches the
        SONIC pure-kinematic recording use case where finger-vs-body /
        finger-vs-floor / finger-vs-finger contacts would inject spurious
        forces into the proprio stream and slow the sim. Pass ``False``
        when you want the fingers to physically interact with scene
        objects (e.g. robocasa pick-and-place tasks). With ``False`` the
        URDF-derived collision primitives (boxes on the thumb MCP/PIP and
        cylinders on every fingertip) are re-classed via
        :func:`_filter_hand_self_collisions` onto the
        ``contype=_HAND_COLLISION_CHANNEL=2,
        conaffinity=_WORLD_CHANNEL=1`` bitmask. Net effect:

        * fingers / palm collide with cube, bowl, table, floor, and the
          X2 body just like any other (1, 1) world geom (no scene-side
          patching needed -- the touchable objects keep their defaults);
        * fingers / palm do NOT collide with each other (palm-vs-finger
          AND finger-vs-finger are both filtered).

        Without this filter the URDF primitives slightly overlap each
        other when the fingers curl in, and the resulting constraint
        forces exceed the position-actuator force limit (3 Nm) so the
        fingers physically pin half-open and the operator's "fully close"
        teleop command never reaches the commanded curl.

    Returns
    -------
    (spec, model, layout)
        ``spec`` is the live ``MjSpec`` -- handy if the caller wants to
        further mutate it before re-compiling.

        ``model`` is the compiled ``MjModel``, ready to allocate ``MjData``
        against. We return the model from this call rather than re-parsing
        ``spec.to_xml()`` because the latter forces relative mesh paths to be
        resolved against the working directory, which breaks our vendored
        layout.

        ``layout`` describes where every hand qpos lives, so callers can
        write 10-D active vectors without re-resolving joint names.
    """
    if asset_root is None:
        asset_root = _vendor_root()

    x2_mjcf = asset_root / "mjcf" / "x2_ultra.xml"
    if not x2_mjcf.is_file():
        raise FileNotFoundError(f"x2_ultra.xml not found at {x2_mjcf}")

    spec = mujoco.MjSpec.from_file(str(x2_mjcf))

    # Replace the wrist_roll visual mesh with the clipped (no-fist) variant
    # before attaching the OmniHand chains, so the seam between the X2 wrist
    # tube and the OmniHand palm cuff lines up at z = _DEFAULT_MOUNT_Z.
    _swap_wrist_roll_visual_to_clipped(spec, asset_root)

    side_cfgs = list(_default_side_configs(asset_root))
    if mount_offset_z is not None:
        side_cfgs = [
            _SideConfig(
                side=s.side,
                sdk_prefix=s.sdk_prefix,
                urdf_path=s.urdf_path,
                parent_body=s.parent_body,
                mount_pos=(s.mount_pos[0], s.mount_pos[1], float(mount_offset_z)),
                mount_quat_wxyz=s.mount_quat_wxyz,
            )
            for s in side_cfgs
        ]

    for side in side_cfgs:
        _attach_hand_spec(spec, side)

    _add_mimic_equalities(spec)
    _add_finger_actuators(spec, tuple(side_cfgs))
    # First compile sets up internal indices; second compile picks up the
    # contype/conaffinity tweaks we apply below.
    spec.compile()
    if disable_hand_collisions:
        _disable_hand_collisions(spec, tuple(side_cfgs))
    else:
        # Hand-vs-world contacts ENABLED (cube, bowl, table, floor, X2
        # body all collide with fingers / palm), hand-vs-hand contacts
        # FILTERED (palm-vs-finger AND finger-vs-finger) via the
        # contype/conaffinity bitmask trick. See
        # ``_filter_hand_self_collisions`` for the contact-rule math
        # and the X2_INTEGRATION_NOTES "Finger collisions on / off"
        # section for the operational rationale.
        _filter_hand_self_collisions(spec, tuple(side_cfgs))
    model = spec.compile()
    if model is None:
        raise RuntimeError("MjSpec.compile() returned None after composition")

    # Stabilise tiny-inertia OmniHand DOFs (post-compile so we hit the
    # final ``model.dof_armature`` / ``dof_damping`` arrays directly,
    # bypassing the version-incompatible ``MjsJoint.damping`` Python
    # setter -- see ``_apply_finger_inertial_stabilization`` docstring).
    _apply_finger_inertial_stabilization(model, tuple(side_cfgs))

    layout = _build_layout(model, tuple(side_cfgs))
    return spec, model, layout


def _build_layout(model: mujoco.MjModel, side_cfgs: tuple[_SideConfig, ...]) -> HandQposLayout:
    active_qposadr: dict[str, list[int]] = {}
    passive_qposadr: dict[str, dict[str, int]] = {}
    full_joint_names: dict[str, list[str]] = {}
    active_actadr: dict[str, list[int]] = {}

    for side_cfg in side_cfgs:
        active_idx: list[int] = []
        active_names: list[str] = []
        active_act_idx: list[int] = []
        for short in ACTIVE_FINGER_JOINTS:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{short}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(
                    f"Active OmniHand joint '{jname}' not found in augmented MJCF "
                    f"(model.njnt={model.njnt}). Composition likely failed."
                )
            active_idx.append(int(model.jnt_qposadr[jid]))
            active_names.append(jname)

            # Resolve the position actuator added by ``_add_finger_actuators``.
            # Missing is non-fatal: legacy renderer paths that compose without
            # finger actuators only use the qpos-write helper and don't call
            # ``apply_active_hand_ctrl``. ``apply_active_hand_ctrl`` raises a
            # clear error if ``active_actadr`` is empty.
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pos_{jname}")
            if aid >= 0:
                active_act_idx.append(int(aid))
        active_qposadr[side_cfg.side] = active_idx
        if len(active_act_idx) == len(ACTIVE_FINGER_JOINTS):
            active_actadr[side_cfg.side] = active_act_idx

        passive_map: dict[str, int] = {}
        for rule in PASSIVE_MIMIC_RULES:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{rule.passive}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Passive joint '{jname}' missing.")
            passive_map[rule.passive] = int(model.jnt_qposadr[jid])
        passive_qposadr[side_cfg.side] = passive_map
        full_joint_names[side_cfg.side] = active_names + [
            f"{side_cfg.side}_{side_cfg.sdk_prefix}{r.passive}_joint" for r in PASSIVE_MIMIC_RULES
        ]

    return HandQposLayout(
        body_qposadr_start=0,
        body_njnt=model.njnt,  # informational only
        active_qposadr=active_qposadr,
        passive_qposadr=passive_qposadr,
        full_joint_names=full_joint_names,
        active_actadr=active_actadr,
    )


# ────────────────────────────────────────────────────────────────────────────
# qpos write helper
# ────────────────────────────────────────────────────────────────────────────


def apply_active_hand_qpos(
    data: mujoco.MjData,
    layout: HandQposLayout,
    *,
    left_active: np.ndarray | None = None,
    right_active: np.ndarray | None = None,
) -> None:
    """Write 10-D active hand vectors into ``data.qpos`` and project mimic DOFs.

    Both the 10 active joints and the 6 passive (mimic) joints per side are
    written directly using the multipliers from ``PASSIVE_MIMIC_RULES``.  We
    project explicitly because ``mj_forward`` does not snap ``qpos`` onto
    equality constraints -- those drive constraint *forces* during dynamics
    integration, not kinematic projection.  The MJCF equality constraints
    remain in the model as a safety net for callers that step dynamics
    instead of taking the kinematic shortcut.

    Either side can be left ``None`` to leave that hand's qpos untouched.
    """
    if left_active is not None:
        _write_side(data, layout, "left", np.asarray(left_active, dtype=np.float64))
    if right_active is not None:
        _write_side(data, layout, "right", np.asarray(right_active, dtype=np.float64))


def _write_side(
    data: mujoco.MjData,
    layout: HandQposLayout,
    side: str,
    active: np.ndarray,
) -> None:
    if active.shape != (len(ACTIVE_FINGER_JOINTS),):
        raise ValueError(
            f"{side} hand active vector must have shape ({len(ACTIVE_FINGER_JOINTS)},), got {active.shape}"
        )

    # Write active joints in canonical order.
    active_qpos_idxs = layout.active_qposadr[side]
    active_lookup: dict[str, float] = {}
    for k, qadr in enumerate(active_qpos_idxs):
        v = float(active[k])
        data.qpos[qadr] = v
        active_lookup[ACTIVE_FINGER_JOINTS[k]] = v

    # Project mimic relationships: passive = multiplier * active.
    passive_map = layout.passive_qposadr[side]
    for rule in PASSIVE_MIMIC_RULES:
        qadr = passive_map[rule.passive]
        data.qpos[qadr] = float(rule.multiplier) * active_lookup[rule.active]


# ────────────────────────────────────────────────────────────────────────────
# ctrl write helper (dynamics path: bridge / sim)
# ────────────────────────────────────────────────────────────────────────────


def apply_active_hand_ctrl(
    data: mujoco.MjData,
    layout: HandQposLayout,
    *,
    left_active: np.ndarray | None = None,
    right_active: np.ndarray | None = None,
) -> None:
    """Write 10-D active hand vectors into ``data.ctrl`` for dynamics stepping.

    Use this from any caller that integrates dynamics with ``mj_step``:
    the live VLA bridge, the SONIC closed-loop sim, etc. The 10 active
    joints per side are driven by position actuators added in
    :func:`_add_finger_actuators`; writing setpoints to ``mj_data.ctrl``
    lets MuJoCo's integrator close the PD loop and the equality
    constraints enforce mimic relationships continuously.

    Either side can be left ``None`` to leave that hand's ctrl untouched.

    Raises
    ------
    RuntimeError
        If ``layout.active_actadr`` is empty -- meaning the augmented
        spec was built without :func:`_add_finger_actuators` (legacy
        renderer-only path). Switch to :func:`apply_active_hand_qpos`
        for kinematic ``mj_forward`` use, or rebuild the spec via
        :func:`build_x2_with_omnihand_spec` (which now installs the
        actuators by default).
    """
    if not layout.active_actadr:
        raise RuntimeError(
            "apply_active_hand_ctrl: layout has no active_actadr "
            "(spec was composed without finger actuators). Either rebuild "
            "via build_x2_with_omnihand_spec(), or use apply_active_hand_qpos "
            "for kinematic mj_forward callers."
        )
    if left_active is not None:
        _write_side_ctrl(data, layout, "left", np.asarray(left_active, dtype=np.float64))
    if right_active is not None:
        _write_side_ctrl(data, layout, "right", np.asarray(right_active, dtype=np.float64))


def _write_side_ctrl(
    data: mujoco.MjData,
    layout: HandQposLayout,
    side: str,
    active: np.ndarray,
) -> None:
    if active.shape != (len(ACTIVE_FINGER_JOINTS),):
        raise ValueError(
            f"{side} hand active vector must have shape ({len(ACTIVE_FINGER_JOINTS)},), got {active.shape}"
        )
    if side not in layout.active_actadr:
        raise RuntimeError(
            f"apply_active_hand_ctrl: side {side!r} has no actuator indices in layout."
        )
    actadr = layout.active_actadr[side]
    for k, aid in enumerate(actadr):
        data.ctrl[aid] = float(active[k])


# ────────────────────────────────────────────────────────────────────────────
# Rest-pose initialisation
# ────────────────────────────────────────────────────────────────────────────


def get_active_hand_rest_pose(side: str) -> np.ndarray:
    """Canonical OmniHand "open hand" rest pose for one side, shape (10,).

    Wraps :data:`HAND_GRASP_OPEN_RAD_LEFT` / :data:`HAND_GRASP_OPEN_RAD_RIGHT`
    from :mod:`gear_sonic.utils.teleop.x2_hand_retarget` (the same
    constants the retargeter's ``ratio=0`` endpoint resolves to). Joint
    order matches :data:`ACTIVE_FINGER_JOINTS` exactly, so this slot
    can be passed straight to :func:`apply_active_hand_qpos` /
    :func:`apply_active_hand_ctrl` without permutation.
    """
    # Lazy import to keep this module importable from contexts that
    # don't depend on the retargeter (e.g. URDF unit tests).
    from gear_sonic.utils.teleop.x2_hand_retarget import (
        HAND_GRASP_OPEN_RAD_LEFT,
        HAND_GRASP_OPEN_RAD_RIGHT,
    )
    if side == "left":
        return np.asarray(HAND_GRASP_OPEN_RAD_LEFT, dtype=np.float64)
    if side == "right":
        return np.asarray(HAND_GRASP_OPEN_RAD_RIGHT, dtype=np.float64)
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def apply_active_hand_rest_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: HandQposLayout,
    *,
    sides: tuple[str, ...] = ("left", "right"),
) -> None:
    """Initialise active OmniHand qpos AND ctrl to the canonical rest pose.

    Without this, ``mj_data.qpos[active_joints]`` defaults to 0 (which
    lands on a joint-limit edge for several OmniHand joints, e.g.
    ``thumb_abad``'s [0..+100]° envelope), and
    ``mj_data.ctrl[finger_actuators]`` also defaults to 0 (so the
    position actuators with ``inheritrange=1`` perpetually drive the
    joint *toward* qpos=0). The combined effect is a slamming motion
    against the joint stops as soon as ``mj_step`` engages -- visible
    as random finger jitter even when the operator hasn't engaged hand
    teleop, and a tail of QACC-NaN warnings on the lower-stiffness
    fingers.

    Calling this once at bridge startup (after ``mj_data`` is allocated
    but before the first ``mj_step``) parks the active joints AND the
    actuator setpoints at the open-hand rest pose returned by
    :func:`get_active_hand_rest_pose`. ``mj_forward`` is run once to
    propagate the mimic equality constraints (passive joints settle to
    ``multiplier × active``) so the model is in a self-consistent state
    before stepping.
    """
    for side in sides:
        rest = get_active_hand_rest_pose(side)
        if side in layout.active_qposadr:
            apply_active_hand_qpos(
                data,
                layout,
                **{f"{side}_active": rest},
            )
        if side in layout.active_actadr:
            _write_side_ctrl(data, layout, side, rest)
    # Settle mimic equalities (passive = multiplier * active) and
    # contacts so the very first mj_step doesn't have to reconcile
    # equality residuals on the same step it integrates.
    mujoco.mj_forward(model, data)


__all__ = [
    "ACTIVE_FINGER_JOINTS",
    "PASSIVE_MIMIC_RULES",
    "HandQposLayout",
    "build_x2_with_omnihand_spec",
    "apply_active_hand_qpos",
    "apply_active_hand_ctrl",
    "apply_active_hand_rest_pose",
    "get_active_hand_rest_pose",
]
