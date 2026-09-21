"""Frozen-core-for-SONIC T1: X2 <-> G1 tracker codec + drop-in wrapper actor.

Wraps the FROZEN GEAR-SONIC G1 policy (T0-verified torch rebuild) in the
planner arc's kinematic alignment Phi so it can drive X2:

    X2 obs (990 proprio + 680 tokenizer)
        --unpack--> per-frame X2 quantities (IL order)
        --Phi encode--> G1 quantities (IL order)
        --pack--> G1 obs (930 proprio + 640 tokenizer)
        --frozen G1 encoder/FSQ/g1_dyn decoder--> 29 G1 actions
        --Phi decode (absolute-target space)--> 31 X2 actions (head = 0)

Single-source-of-truth rules (no transcribed tables):
  * Phi affine rows: imported from
    ``motionbricks/scripts/export_g1core_x2_planner_onnx._PHI_TABLE``
    (robot-verified by the shipped planner).
  * X2 params: imported from ``eval_x2_mujoco`` (the MuJoCo harness whose
    obs pipeline is verified to ~3.6e-7 vs the live training module).
  * G1 params: harvested by stub-importing
    ``gear_sonic/envs/manager_env/robots/g1.py`` (isaaclab mocked, values
    evaluated from the cfg literals) — the same cfg the release policy
    trained with (release config.yaml: robot.type == g1_model_12_dex).
  * G1 obs layouts: T0-probed ([cmd58|ori6] x 10; see
    the T0 parity probe (not shipped)) + the deploy shim's segment maps.

Conventions handled (each one is a documented trap):
  * tokenizer ref joint positions are ABSOLUTE -> plain affine.
  * proprio joint positions are DEFAULT-RELATIVE -> defaults algebra.
  * joint velocities map with slope only (no offset).
  * actions are scaled offsets from default -> map in absolute-target
    space using each side's (default, action_scale).
  * decoder last-action history must be G1-NATIVE (what the frozen policy
    itself output), never mapped X2 actions.
  * G1 raw actions clipped at +/-20 (release config action_clip_value).
  * base ang-vel / gravity-dir / anchor-ori-6D are body-frame,
    embodiment-neutral -> pass through.

Run self-tests (IsaacLab conda env):
    $ISAACLAB_PYTHON gear_sonic/scripts/frozen_core_sonic_codec.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "gear_sonic" / "scripts"
for p in (str(SCRIPTS_DIR), str(REPO_ROOT / "motionbricks" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from export_g1core_x2_planner_onnx import _PHI_TABLE  # noqa: E402

NUM_G1, NUM_X2, NFRAMES = 29, 31, 10
G1_TOK_DIM, X2_TOK_DIM = 640, 680          # 10*(2*29+6) / 10*(2*31+6)
G1_PROP_DIM, X2_PROP_DIM = 930, 990        # 10*(3+3*29+3) / 10*(3+3*31+3)
G1_ACTION_CLIP = 20.0                      # release config action_clip_value


# ---------------------------------------------------------------------------
# Parameter harvesters
# ---------------------------------------------------------------------------

def harvest_x2_params() -> dict:
    """X2 names/orders/defaults/scales from the ROBOT DEFINITION.

    Sourced from config/robot_plant/*.yaml (armature/effort/defaults), the MJCF
    (MuJoCo joint order) and robots/x2_ultra.py (IL<->MJ map) -- mirroring what
    harvest_g1_params does for G1.

    IT USED TO IMPORT eval_x2_mujoco AND CALL ITS PRIVATE
    ``_compute_gains_and_scales()``. That inverted the dependency: the action
    scales baked into the deploy ONNX -- a production artifact that runs on the
    robot -- were defined by an EVAL SCRIPT, which had quietly become the
    de-facto robot definition (it also kept its own DEFAULT_JOINT_POS, a fourth
    copy). When training moved to the vendor datasheet on 2026-08-23 the eval
    script did not, so every export would have carried stale scales.

    Plant selection honours $X2_PLANT, so a pre-vendor checkpoint can be
    exported with the scales it was trained with.
    """
    import pathlib
    import re as _re
    import sys as _sys

    _robots = (pathlib.Path(__file__).resolve().parents[1]
               / "envs" / "manager_env" / "robots")
    if str(_robots) not in _sys.path:
        _sys.path.insert(0, str(_robots))
    from plant_config import load_plant

    plant = load_plant()

    # MuJoCo joint order comes from the MJCF itself, not from a python list.
    mjcf = (pathlib.Path(__file__).resolve().parents[1] / "data" / "assets"
            / "robot_description" / "mjcf" / "x2_ultra.xml").read_text()
    # NOTE: attribute order varies in this MJCF -- many joints carry
    # class="..." BEFORE name=, so a <joint\s+name= regex silently finds
    # only 16 of 31. Match each <joint ...> tag, then pull name from it.
    mj_full = []
    for _tag in _re.findall(r"<joint\b[^>]*>", mjcf):
        _m = _re.search(r'name="([^"]+)"', _tag)
        if _m and _m.group(1) != "floating_base_joint":
            mj_full.append(_m.group(1))

    # IL <-> MJ map from the robot cfg (stub import; isaaclab mocked).
    import types as _types
    import unittest.mock as _mock
    for _m in ("isaaclab", "isaaclab.actuators", "isaaclab.assets",
               "isaaclab.assets.articulation", "isaaclab.sim", "isaaclab.utils"):
        _sys.modules.setdefault(_m, _mock.MagicMock())
    # x2_ultra.py now does `from .plant_config import ...` (vendor-plant
    # refactor) -- a bare file-style load has no package context and dies
    # with "attempted relative import". Synthesize a parent package whose
    # __path__ is the robots dir so the relative import resolves, while
    # still avoiding a real gear_sonic package import (isaaclab stays
    # mocked above).
    import types as _t2
    _pkg = _sys.modules.get("_x2u_pkg")
    if _pkg is None:
        _pkg = _t2.ModuleType("_x2u_pkg")
        _pkg.__path__ = [str(_robots)]
        _sys.modules["_x2u_pkg"] = _pkg
    _spec = importlib.util.spec_from_file_location(
        "_x2u_pkg.x2_ultra", str(_robots / "x2_ultra.py"))
    _x2u = importlib.util.module_from_spec(_spec)
    _sys.modules["_x2u_pkg.x2_ultra"] = _x2u
    _spec.loader.exec_module(_x2u)
    il_to_mj = list(_x2u.X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF)

    if len(mj_full) != len(il_to_mj):
        raise ValueError(
            f"MJCF gave {len(mj_full)} joints, IL map has {len(il_to_mj)}")

    def _first_match(table, joint, what):
        for k, v in table.items():
            if k in joint:
                return float(v)
        raise KeyError(f"no {what} entry matches {joint!r}")

    scale_mj = [plant.action_scale_for(n) for n in mj_full]
    default_mj = [_first_match(plant.default_joint_pos, n, "default_joint_pos")
                  if any(k in n for k in plant.default_joint_pos) else 0.0
                  for n in mj_full]
    mj_names = [n.removesuffix("_joint") for n in mj_full]
    return {
        "mj_names": mj_names,
        "il_names": [mj_names[i] for i in il_to_mj],
        "il_to_mj": il_to_mj,
        "default_il": np.asarray(default_mj, np.float64)[il_to_mj],
        "scale_il": np.asarray(scale_mj, np.float64)[il_to_mj],
    }


def harvest_g1_params() -> dict:
    """G1 names/orders/defaults/scales from the framework robot cfg.

    Stub-imports ``robots/g1.py`` with isaaclab mocked so the cfg literals
    evaluate without an Isaac Sim runtime. Mirrors the framework's own
    scale rule: ``modular_tracking_env_cfg`` sets
    ``actions.joint_pos.scale = G1_MODEL_12_ACTION_SCALE`` (dict of joint
    regex -> value) with ``use_default_offset=True``.
    """
    import importlib.util
    import re
    import types
    import unittest.mock as mock

    class _Kw:
        def __init__(self, *a, **k):
            self.__dict__.update(k)

        def replace(self, **k):
            self.__dict__.update(k)
            return self

    stubs = {}
    for name in (
        "isaaclab", "isaaclab.actuators", "isaaclab.assets",
        "isaaclab.assets.articulation", "isaaclab.sim", "isaaclab.utils",
        "isaaclab.utils.assets",
    ):
        stubs[name] = types.ModuleType(name)
    stubs["isaaclab.actuators"].ImplicitActuatorCfg = _Kw
    stubs["isaaclab.assets.articulation"].ArticulationCfg = _Kw
    _Kw.InitialStateCfg = _Kw
    sim = stubs["isaaclab.sim"]
    sim.UsdFileCfg = _Kw
    sim.UrdfFileCfg = _Kw
    sim.RigidBodyPropertiesCfg = _Kw
    sim.ArticulationRootPropertiesCfg = _Kw
    sim.CollisionPropertiesCfg = _Kw
    sim.MassPropertiesCfg = _Kw
    UrdfConverterCfg = _Kw
    UrdfConverterCfg.JointDriveCfg = _Kw
    UrdfConverterCfg.JointDriveCfg.PDGainsCfg = _Kw
    sim.UrdfConverterCfg = UrdfConverterCfg
    stubs["isaaclab.utils.assets"].ISAACLAB_NUCLEUS_DIR = ""

    g1_path = REPO_ROOT / "gear_sonic" / "envs" / "manager_env" / "robots" / "g1.py"
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("_g1_cfg_harvest", g1_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    # MuJoCo joint order ground truth: the G1 MJCF itself.
    mjcf = (REPO_ROOT / "gear_sonic" / "data" / "assets" / "robot_description"
            / "mjcf" / "g1_29dof_rev_1_0.xml").read_text()
    mj_joint_names = re.findall(r'<joint\s+name="([^"]+)"', mjcf)
    mj_joint_names = [n for n in mj_joint_names if n != "floating_base_joint"]
    if len(mj_joint_names) != NUM_G1:
        raise ValueError(f"MJCF gave {len(mj_joint_names)} joints, want {NUM_G1}")

    # NAMING TRAP: g1.py's G1_ISAACLAB_TO_MUJOCO_DOF is indexed by MJ
    # position; the true [il] -> mj map is G1_MUJOCO_TO_ISAACLAB_DOF
    # (verified: it reproduces IsaacLab's BFS-interleaved joint order and
    # matches eval_x2_mujoco.IL_TO_MJ_DOF's pattern for X2).
    il_to_mj = list(mod.G1_MUJOCO_TO_ISAACLAB_DOF)         # [il] -> mj, 29
    il_joint_names = [mj_joint_names[i] for i in il_to_mj]
    if (il_joint_names[1] != "right_hip_pitch_joint"
            or il_joint_names[2] != "waist_yaw_joint"):
        raise ValueError(f"G1 IL order not BFS-interleaved: {il_joint_names[:4]}")
    cfg = mod.G1_CYLINDER_MODEL_12_DEX_CFG
    joint_pos_patterns = cfg.init_state.joint_pos          # regex -> rad
    scale_patterns = mod.G1_MODEL_12_ACTION_SCALE          # regex -> scale

    def resolve(patterns: dict, joint: str, fallback: float | None) -> float:
        hits = [v for p, v in patterns.items() if re.fullmatch(p, joint)]
        if len(hits) > 1:
            raise ValueError(f"{joint}: multiple pattern matches")
        if hits:
            return float(hits[0])
        if fallback is None:
            raise KeyError(f"{joint}: no pattern match and no fallback")
        return fallback

    default_il = np.array(
        [resolve(joint_pos_patterns, n, 0.0) for n in il_joint_names], np.float64)
    scale_il = np.array(
        [resolve(scale_patterns, n, None) for n in il_joint_names], np.float64)

    # Full 12_dex actuator plant (kp/kd/armature/effort per IL joint) — the
    # v1.1 training plant. NOT the v1-release constants hardcoded in
    # eval_g1_mujoco (those pair with the v1 ONNX + action scale 1.0).
    kp_il = np.zeros(NUM_G1); kd_il = np.zeros(NUM_G1)
    armature_il = np.zeros(NUM_G1); effort_il = np.zeros(NUM_G1)
    for i, n in enumerate(il_joint_names):
        for act in cfg.actuators.values():
            if not any(re.fullmatch(p, n) for p in act.joint_names_expr):
                continue

            def _resolve_attr(attr, default=None):
                v = getattr(act, attr, default)
                if isinstance(v, dict):
                    hits = [val for p, val in v.items() if re.fullmatch(p, n)]
                    v = hits[0] if hits else default
                return v

            kp_il[i] = float(_resolve_attr("stiffness"))
            kd_il[i] = float(_resolve_attr("damping"))
            armature_il[i] = float(_resolve_attr("armature", 0.0) or 0.0)
            effort_il[i] = float(_resolve_attr("effort_limit_sim"))
            break
        else:
            raise KeyError(f"{n}: no actuator group matches")

    return {
        "mj_names": [n.removesuffix("_joint") for n in mj_joint_names],
        "il_names": [n.removesuffix("_joint") for n in il_joint_names],
        "il_to_mj": il_to_mj,
        "default_il": default_il,
        "scale_il": scale_il,
        "kp_il": kp_il,
        "kd_il": kd_il,
        "armature_il": armature_il,
        "effort_il": effort_il,
    }


# ---------------------------------------------------------------------------
# Phi in IsaacLab order, both directions
# ---------------------------------------------------------------------------

def build_phi_il(g1p: dict, x2p: dict) -> dict:
    """Resolve _PHI_TABLE (CSV/MJ order, x2 = a*g1 + b) into IL-order maps.

    Returns
    -------
    dict with numpy arrays:
      enc_src[29]  X2 IL index feeding each G1 IL joint
      enc_a/enc_b[29]  affine of that X2 source (x2 = a*g1 + b)
      dec_src[31]  G1 IL index feeding each X2 IL joint (-1 = head)
      dec_a/dec_b[31]
    """
    # _PHI_TABLE rows are in X2 CSV order == eval_x2_mujoco.MUJOCO_JOINT_NAMES
    tbl_names = [r[0] for r in _PHI_TABLE]
    if tbl_names != x2p["mj_names"]:
        raise ValueError(
            "Phi table order != X2 MuJoCo names:\n"
            f"  table: {tbl_names}\n  mj:    {x2p['mj_names']}")

    g1_mj_names = g1p["mj_names"]
    dec_src = np.full(NUM_X2, -1, np.int64)
    dec_a = np.zeros(NUM_X2)
    dec_b = np.zeros(NUM_X2)
    enc_src = np.full(NUM_G1, -1, np.int64)
    enc_a = np.zeros(NUM_G1)
    enc_b = np.zeros(NUM_G1)

    for x2_mj, (x2_name, g1_mj, a, b_deg) in enumerate(_PHI_TABLE):
        x2_il = x2p["il_to_mj"].index(x2_mj)
        if g1_mj < 0:
            continue  # head pair: decoder emits 0 (default), encoder drops
        g1_il = g1p["il_to_mj"].index(g1_mj)
        b = np.deg2rad(b_deg)
        dec_src[x2_il], dec_a[x2_il], dec_b[x2_il] = g1_il, a, b
        if enc_src[g1_il] >= 0:
            raise ValueError(f"G1 joint {g1_mj_names[g1_mj]} fed twice")
        enc_src[g1_il], enc_a[g1_il], enc_b[g1_il] = x2_il, a, b

    if (enc_src < 0).any():
        missing = [g1p["il_names"][i] for i in np.where(enc_src < 0)[0]]
        raise ValueError(f"G1 joints with no X2 source: {missing}")

    # Wrist b-term re-fit (2026-08-15, FROZEN_CORE_WRIST_B_REFIT=0 to disable).
    # The G1 backbone holds a characteristic wrist posture on the micromotor
    # axes; with b=0 the commanded X2 target sits a CONSTANT offset from the
    # reference (measured closed-loop over a 10-clip battery, spread <=0.6:
    # left_wrist_roll pinned at its +0.72 stop, target median +1.13 past ref).
    # delta = median(commanded_target - ref) in X2 radians; new b = b - delta,
    # applied to BOTH enc and dec so the affine stays a bijection: the policy
    # perceives its preferred G1 posture exactly when X2 sits at the reference.
    # left_wrist_pitch skipped (spread 0.76 - posture-dependent), right_wrist_yaw
    # negligible (-0.03).
    if os.environ.get("FROZEN_CORE_WRIST_B_REFIT", "1") != "0":
        _WRIST_B_DELTA = {"left_wrist_roll_joint": 1.13,
                          "right_wrist_pitch_joint": -0.70,
                          "left_wrist_yaw_joint": 0.17,
                          "right_wrist_roll_joint": -0.18}
        # Export-time per-checkpoint re-fit (wandering-wrist remedy, ledger
        # 2026-08-15): reward-invisible wrist axes drift under continued
        # training, so each SHIPPED artifact re-measures median(target-ref)
        # on the fit battery and passes the residuals here. JSON dict of
        # {x2_joint_name: delta_rad}, ADDED to the base deltas above.
        _extra = os.environ.get("FROZEN_CORE_WRIST_B_EXTRA")
        if _extra:
            import json as _json
            for _k, _v in _json.loads(_extra).items():
                _k = _k if _k.endswith("_joint") else _k + "_joint"
                _WRIST_B_DELTA[_k] = _WRIST_B_DELTA.get(_k, 0.0) + float(_v)
        for x2_mj, (x2_name, g1_mj, _a, _b) in enumerate(_PHI_TABLE):
            d = _WRIST_B_DELTA.get(x2_name if x2_name.endswith("_joint")
                                   else x2_name + "_joint")
            if d is None or g1_mj < 0:
                continue
            x2_il = x2p["il_to_mj"].index(x2_mj)
            g1_il = g1p["il_to_mj"].index(g1_mj)
            dec_b[x2_il] -= d
            enc_b[g1_il] -= d

    return {"enc_src": enc_src, "enc_a": enc_a, "enc_b": enc_b,
            "dec_src": dec_src, "dec_a": dec_a, "dec_b": dec_b}


class SonicPhiCodec:
    """Pure-numpy obs/action codec. All vectors IsaacLab order."""

    def __init__(self, g1p: dict, x2p: dict):
        self.g1p, self.x2p = g1p, x2p
        self.phi = build_phi_il(g1p, x2p)
        p = self.phi
        self.d_g1 = g1p["default_il"]
        self.s_g1 = g1p["scale_il"]
        self.d_x2 = x2p["default_il"]
        self.s_x2 = x2p["scale_il"]
        # decoder rows with a G1 source
        self.dec_rows = np.where(p["dec_src"] >= 0)[0]
        self.dec_gsrc = p["dec_src"][self.dec_rows]

    # -- joint-space maps (vectorised over leading dims) ----------------
    def q_x2_to_g1(self, q_x2: np.ndarray) -> np.ndarray:
        """Absolute joint positions X2 -> G1: g1 = (x2 - b) / a."""
        p = self.phi
        return (q_x2[..., p["enc_src"]] - p["enc_b"]) / p["enc_a"]

    def v_x2_to_g1(self, v_x2: np.ndarray) -> np.ndarray:
        p = self.phi
        return v_x2[..., p["enc_src"]] / p["enc_a"]

    def q_g1_to_x2(self, q_g1: np.ndarray) -> np.ndarray:
        """Absolute joint positions G1 -> X2; head joints get X2 default."""
        out = np.broadcast_to(self.d_x2, q_g1.shape[:-1] + (NUM_X2,)).copy()
        p = self.phi
        out[..., self.dec_rows] = (
            p["dec_a"][self.dec_rows] * q_g1[..., self.dec_gsrc]
            + p["dec_b"][self.dec_rows])
        return out

    def action_g1_to_x2(self, act_g1: np.ndarray) -> np.ndarray:
        """G1 raw action -> X2 raw action via absolute-target space."""
        q_tgt_g1 = self.d_g1 + act_g1 * self.s_g1
        q_tgt_x2 = self.q_g1_to_x2(q_tgt_g1)
        return (q_tgt_x2 - self.d_x2) / self.s_x2

    # -- obs pack/unpack -------------------------------------------------
    def unpack_x2_tokenizer(self, tok: np.ndarray):
        """(680,) -> jp(10,31) abs, jv(10,31), ori(10,6)."""
        blk = tok.reshape(NFRAMES, 2 * NUM_X2 + 6)
        cmd_flat = blk[:, :2 * NUM_X2].reshape(-1)          # 620
        jp = cmd_flat[:NFRAMES * NUM_X2].reshape(NFRAMES, NUM_X2)
        jv = cmd_flat[NFRAMES * NUM_X2:].reshape(NFRAMES, NUM_X2)
        return jp, jv, blk[:, 2 * NUM_X2:]

    def pack_g1_tokenizer(self, jp: np.ndarray, jv: np.ndarray,
                          ori: np.ndarray) -> np.ndarray:
        """jp/jv (10,29) + ori (10,6) -> (640,) in the T0-probed layout."""
        cmd = np.concatenate([jp.reshape(-1), jv.reshape(-1)])   # 580
        blk = np.concatenate([cmd.reshape(NFRAMES, 2 * NUM_G1),
                              ori], axis=-1)                     # (10, 64)
        return blk.reshape(-1).astype(np.float32)

    def tokenizer_x2_to_g1(self, tok_x2: np.ndarray) -> np.ndarray:
        jp, jv, ori = self.unpack_x2_tokenizer(tok_x2)
        return self.pack_g1_tokenizer(
            self.q_x2_to_g1(jp), self.v_x2_to_g1(jv), ori)

    def proprio_x2_to_g1(self, prop_x2: np.ndarray,
                         g1_action_hist: np.ndarray) -> np.ndarray:
        """(990,) + G1-native action history (10,29) -> (930,).

        Term layout (both sides, frame-major within term, oldest first):
        [ang_vel 10x3 | jp_rel 10xN | jv 10xN | last_act 10xN | grav 10x3]
        """
        o = 0
        angvel = prop_x2[o:o + 30]; o += 30
        jp_rel = prop_x2[o:o + NFRAMES * NUM_X2].reshape(NFRAMES, NUM_X2)
        o += NFRAMES * NUM_X2
        jv = prop_x2[o:o + NFRAMES * NUM_X2].reshape(NFRAMES, NUM_X2)
        o += NFRAMES * NUM_X2
        o += NFRAMES * NUM_X2  # skip X2 action history (replaced by G1-native)
        grav = prop_x2[o:o + 30]
        jp_g1_rel = self.q_x2_to_g1(jp_rel + self.d_x2) - self.d_g1
        return np.concatenate([
            angvel,
            jp_g1_rel.reshape(-1),
            self.v_x2_to_g1(jv).reshape(-1),
            g1_action_hist.reshape(-1),
            grav,
        ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Drop-in actor (UniversalTokenActor-compatible signature)
# ---------------------------------------------------------------------------

class FrozenCoreG1SonicActor(nn.Module):
    """Frozen GEAR-SONIC G1 policy wrapped in Phi — drives X2.

    Same call contract as ``eval_x2_mujoco.UniversalTokenActor``:
    ``forward(proprioception(990), tokenizer_obs(680)) -> action(31)``.
    Batch dim 1 only (keeps its own G1-native action history).
    """

    # Frozen G1 release checkpoint (nvidia/GEAR-SONIC sonic_release/last.pt);
    # SONIC_RELEASE_CKPT overrides the default HF-cache location (MODELS.md).
    HF_CKPT = Path(os.environ.get(
        "SONIC_RELEASE_CKPT",
        "~/.cache/huggingface/hub/models--nvidia--GEAR-SONIC/snapshots/"
        "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/sonic_release/last.pt",
    )).expanduser()

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu",
                 waist_dev_clamp: float | None = None,
                 remap_wrist: bool = False,
                 remap_waist: bool = False):
        """``waist_dev_clamp``: clamp X2 waist TARGET deviation (rad).

        OFF (None) for sim eval — clamping masks tracking ability. For any
        ROBOT path it is MANDATORY at 0.45 (waist-saturation incident:
        hardware waist_pr is 24 Nm vs the 32 Nm the incumbent trained at;
        the Kimodo G1->X2 converter applies the same 0.45 clamp).

        ``remap_wrist``: MUST match how the checkpoint trained. Models
        trained with ``backbone.remap_wrist_ranges: true`` (S0 fresh-vendor
        lineage, 2026-08-30 onward) need True; every earlier checkpoint
        trained on the unmapped affine and needs False. A mismatch is
        silent — same shapes, wrong wrist coefficients (3.07x on pitch).
        """
        super().__init__()
        self.waist_dev_clamp = waist_dev_clamp
        self.remap_wrist = bool(remap_wrist)
        from gear_sonic.trl.modules.onnx_helpers import (  # noqa: E402
            FsqQuantizer, _mlp_from_state_dict, tolerant_torch_load)

        sd = tolerant_torch_load(str(ckpt_path or self.HF_CKPT))["policy_state_dict"]
        self.encoder, enc_dims = _mlp_from_state_dict(
            sd, "actor_module.encoders.g1.module.")
        self.decoder, dec_dims = _mlp_from_state_dict(
            sd, "actor_module.decoders.g1_dyn.module.")
        assert enc_dims[0] == G1_TOK_DIM and dec_dims[-1] == NUM_G1
        self.fsq = FsqQuantizer(levels=32)
        self.codec = SonicPhiCodec(harvest_g1_params(), harvest_x2_params())
        if remap_wrist:
            remap_wrist_ranges(self.codec)
            print("[frozen-core-g1-actor] wrist range-map APPLIED", flush=True)
        if remap_waist:
            remap_waist_ranges(self.codec)
            print("[frozen-core-g1-actor] waist range-map APPLIED", flush=True)
        # dec-based history inversion needed whenever ANY remap made
        # enc/dec asymmetric (wrist) — harmless when they are true
        # inverses (waist rows), so key it on either flag.
        self.remap_wrist = bool(remap_wrist or remap_waist) or self.remap_wrist
        self.device_ = device
        self.to(device).eval()
        self.reset()

    def reset(self):
        self._g1_action_hist = np.zeros((NFRAMES, NUM_G1), np.float32)

    @torch.no_grad()
    def forward(self, proprioception, tokenizer_obs):
        prop_x2 = np.asarray(proprioception, np.float32).reshape(-1)
        tok_x2 = np.asarray(tokenizer_obs, np.float32).reshape(-1)
        assert prop_x2.shape[0] == X2_PROP_DIM and tok_x2.shape[0] == X2_TOK_DIM

        tok_g1 = self.codec.tokenizer_x2_to_g1(tok_x2)
        if self.remap_wrist:
            # Remap-trained models: the decode map is NOT a bijection
            # (wrist_roll and wrist_yaw both source G1 yaw), so a recurrent
            # G1 buffer diverges from what training saw. Training (and the
            # fused deploy graph) reconstruct the G1 history by inverting
            # the X2 action history in proprio — do the identical inversion
            # here (numpy twin of FrozenCoreG1ActorModule._act_x2_to_g1;
            # duplicate-index scatter = last write wins, same both sides).
            c = self.codec
            p = c.phi
            act_x2_hist = prop_x2[650:960].reshape(NFRAMES, NUM_X2).astype(np.float64)
            q_x2 = c.d_x2 + act_x2_hist * c.s_x2
            q_g1_from = (q_x2[:, c.dec_rows] - p["dec_b"][c.dec_rows]) \
                / p["dec_a"][c.dec_rows]
            hist = np.zeros((NFRAMES, NUM_G1), np.float64)
            hist[:, c.dec_gsrc] = (q_g1_from - c.d_g1[c.dec_gsrc]) / c.s_g1[c.dec_gsrc]
            self._g1_action_hist = hist.astype(np.float32)
        prop_g1 = self.codec.proprio_x2_to_g1(prop_x2, self._g1_action_hist)

        t = lambda x: torch.from_numpy(x).to(self.device_)[None]  # noqa: E731
        latent = self.encoder(t(tok_g1))
        token = self.fsq(latent)
        act_g1 = self.decoder(torch.cat([token, t(prop_g1)], dim=-1))[0]
        act_g1 = act_g1.clamp(-G1_ACTION_CLIP, G1_ACTION_CLIP).cpu().numpy()

        # roll G1-native history (oldest first)
        self._g1_action_hist = np.roll(self._g1_action_hist, -1, axis=0)
        self._g1_action_hist[-1] = act_g1

        act_x2 = self.codec.action_g1_to_x2(act_g1.astype(np.float64))
        if self.waist_dev_clamp is not None:
            c = self.codec
            for x2n in ("waist_yaw", "waist_pitch", "waist_roll"):
                i = self.codec.x2p["il_names"].index(x2n)
                dev = act_x2[i] * c.s_x2[i]
                act_x2[i] = np.clip(dev, -self.waist_dev_clamp,
                                    self.waist_dev_clamp) / c.s_x2[i]
        return torch.from_numpy(act_x2.astype(np.float32))[None]


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _self_test() -> int:
    np.set_printoptions(precision=4, suppress=True)
    g1p, x2p = harvest_g1_params(), harvest_x2_params()
    print(f"[codec] G1 IL joints ({len(g1p['il_names'])}): {g1p['il_names'][:6]} ...")
    print(f"[codec] X2 IL joints ({len(x2p['il_names'])}): {x2p['il_names'][:6]} ...")
    codec = SonicPhiCodec(g1p, x2p)

    # 1. default-pose consistency: Phi(d_g1) should be near d_x2
    q_x2_pred = codec.q_g1_to_x2(codec.d_g1)
    resid = q_x2_pred - codec.d_x2
    worst = np.argsort(-np.abs(resid))[:5]
    print("[codec] default-pose residual |Phi(d_g1) - d_x2| "
          f"max {np.abs(resid).max():.4f} rad; worst joints:")
    for i in worst:
        print(f"    {x2p['il_names'][i]:24s} {resid[i]:+.4f}")

    # 1b. wrist + waist mapping audit (operator-flagged joint differences).
    # Expected from _PHI_TABLE: waist joints pair by NAME (their MJ storage
    # slots differ: G1 yaw/roll/pitch vs X2 yaw/pitch/roll); wrists pair
    # CROSS-name: x2 wrist_yaw <- g1 wrist_roll with sign flip (a=-1),
    # x2 wrist_roll <- g1 wrist_yaw (no flip), pitch <-> pitch.
    print("[codec] wrist/waist decode rows (x2 <- g1, a, b):")
    expected_pairs = {
        "waist_yaw": "waist_yaw", "waist_pitch": "waist_pitch",
        "waist_roll": "waist_roll",
        "left_wrist_yaw": "left_wrist_roll",
        "left_wrist_pitch": "left_wrist_pitch",
        "left_wrist_roll": "left_wrist_yaw",
        "right_wrist_yaw": "right_wrist_roll",
        "right_wrist_pitch": "right_wrist_pitch",
        "right_wrist_roll": "right_wrist_yaw",
    }
    wm_ok = True
    for x2_il, x2n in enumerate(x2p["il_names"]):
        if "wrist" not in x2n and "waist" not in x2n:
            continue
        gsrc = codec.phi["dec_src"][x2_il]
        g1n = g1p["il_names"][gsrc]
        a_, b_ = codec.phi["dec_a"][x2_il], codec.phi["dec_b"][x2_il]
        flag = ""
        if expected_pairs[x2n] != g1n:
            flag, wm_ok = "  <-- UNEXPECTED PAIR", False
        if x2n.endswith("wrist_yaw") and a_ != -1.0:
            flag, wm_ok = "  <-- EXPECTED SIGN FLIP", False
        print(f"    {x2n:22s} <- {g1n:22s} a={a_:+.3f} b={np.rad2deg(b_):+6.2f}deg{flag}")

    # 2. affine roundtrip on matched joints
    rng = np.random.default_rng(0)
    q_g1 = rng.normal(0, 0.5, (256, NUM_G1))
    q_rt = codec.q_x2_to_g1(codec.q_g1_to_x2(q_g1))
    rt = np.abs(q_rt - q_g1).max()
    print(f"[codec] roundtrip g1->x2->g1 max err {rt:.2e}")

    # 3. G1 pack cross-check vs the T0-probed deploy-graph layout (needs the
    #    T0 parity probe module, which is not shipped; skipped without it)
    try:
        from frozen_core_t0_parity import build_layouts  # not shipped
        jp = rng.normal(0, 0.5, (4, NFRAMES, NUM_G1)).astype(np.float32)
        jv = rng.normal(0, 1.0, (4, NFRAMES, NUM_G1)).astype(np.float32)
        ori = rng.normal(0, 0.6, (4, NFRAMES, 6)).astype(np.float32)
        ref = build_layouts(jp.reshape(4, -1), jv.reshape(4, -1),
                            ori.reshape(4, -1))["probed[cmd58|ori6]x10"]
        packed = np.stack([codec.pack_g1_tokenizer(jp[i], jv[i], ori[i])
                           for i in range(4)])
        pk = np.abs(packed - ref).max()
    except ImportError:
        print("[codec] pack cross-check SKIPPED (T0 parity probe not shipped)")
        pk = 0.0
    print(f"[codec] pack_g1_tokenizer vs T0 probed layout max diff {pk:.2e}")

    # 4. X2 unpack/pack roundtrip
    tok = rng.normal(0, 0.5, X2_TOK_DIM).astype(np.float32)
    jp2, jv2, ori2 = codec.unpack_x2_tokenizer(tok)
    blk = np.concatenate([np.concatenate([jp2.reshape(-1), jv2.reshape(-1)])
                          .reshape(NFRAMES, 62), ori2], axis=-1).reshape(-1)
    ux = np.abs(blk - tok).max()
    print(f"[codec] X2 tokenizer unpack/repack roundtrip {ux:.2e}")

    # 5. full actor smoke
    actor = FrozenCoreG1SonicActor()
    import time
    prop = rng.normal(0, 0.3, X2_PROP_DIM).astype(np.float32)
    tokx = rng.normal(0, 0.3, X2_TOK_DIM).astype(np.float32)
    t0 = time.time()
    for _ in range(50):
        act = actor(prop, tokx)
    dt = (time.time() - t0) / 50
    a = act.numpy().ravel()
    head = [x2p["il_names"].index(n) for n in ("head_yaw", "head_pitch")
            if n in x2p["il_names"]]
    print(f"[codec] actor smoke: action shape {a.shape}, finite {np.isfinite(a).all()},"
          f" head rows {a[head] if head else 'n/a'}, {dt*1000:.1f} ms/step")

    ok = (rt < 1e-9 and pk == 0.0 and ux == 0.0 and np.isfinite(a).all()
          and (not head or np.abs(a[head]).max() < 1e-12))
    print(f"[codec] SELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())


class FrozenCoreSmplActor(nn.Module):
    """SMPL-conditioned whole-body actor on the FROZEN-G1 lineage.

    Assembly (2026-08-21, to answer: "does the frozen-G1 X2 model do
    whole-body with Pico?"):
      release smpl encoder (frozen G1 core last.pt — trained, 840-D)
        -> FSQ -> g1_dyn decoder from the frozen-core MERGED ckpt (X2-adapted)
        -> phi codec action map -> X2 (31)
    Same call contract as UniversalTokenActor / FrozenCoreG1SonicActor:
    forward(proprioception(990 X2), smpl_obs(840)) -> (1, 31).

    The latent-space bet being tested: the release trained all encoders to
    a SHARED aligned latent, and the T2 LoRA fine-tuned the decoder on
    g1-encoder tokens — smpl-encoder tokens should still land in-distribution.

    smpl_obs wrist term: the release smpl encoder learned G1 wrist dofs
    (IL [23..28]); harnesses built for X2 supply X2 wrists — so forward()
    REWRITES the 6 wrist slots per frame from the codec-mapped CURRENT G1
    wrists derived from the X2 proprio (self-consistent, embodiment-correct).
    """

    def __init__(self, merged_ckpt: str | None = None,
                 release_ckpt: str | None = None, device: str = "cpu",
                 remap_waist: bool = False):
        super().__init__()
        self.remap_waist = bool(remap_waist)
        from gear_sonic.trl.modules.onnx_helpers import (  # noqa: E402
            FsqQuantizer, _mlp_from_state_dict, tolerant_torch_load)
        release_path = str(release_ckpt or FrozenCoreG1SonicActor.HF_CKPT)
        rel = tolerant_torch_load(release_path)["policy_state_dict"]
        self.smpl_encoder, enc_dims = _mlp_from_state_dict(
            rel, "actor_module.encoders.smpl.module.")
        assert enc_dims[0] == 840, enc_dims
        # v1.1 cores are HEADING-normalized (robot yaw only, config trap T5):
        # same 6D shape as the full-orientation variant, so it MUST be
        # detected here and honored by the obs builder. v1.1's g1_dyn is 9
        # linears vs the original release's 7.
        n_lin = len([k for k in rel
                     if k.startswith("actor_module.decoders.g1_dyn.module.")
                     and k.endswith(".weight")])
        self.ori_mode = ("heading" if n_lin >= 9 or "v1_1" in release_path
                         else "full")
        mrg = tolerant_torch_load(str(merged_ckpt))["policy_state_dict"]
        for pfx in ("actor_module.decoders.g1_dyn.module.",
                    "actor_module.decoder."):  # v11 fine-tunes use flat naming
            if any(k.startswith(pfx) for k in mrg):
                # in-training ckpts carry UNMERGED LoRA factors; fold them
                # (W' = W + (alpha/rank) * B @ A; frozen_core_t2_export.py:50)
                LORA_SCALING = 1.0  # alpha/rank = 16/16
                sub, n_folded = {}, 0
                for k, v in mrg.items():
                    if not k.startswith(pfx) or k.endswith((".lora_A",
                                                            ".lora_B")):
                        continue
                    if k.endswith(".weight") and k[:-7] + ".lora_A" in mrg:
                        v = (v.double() + LORA_SCALING *
                             (mrg[k[:-7] + ".lora_B"].double()
                              @ mrg[k[:-7] + ".lora_A"].double())).float()
                        n_folded += 1
                    sub[k] = v
                if n_folded:
                    print(f"[frozen-core-smpl] folded LoRA into {n_folded} "
                          "decoder linears", flush=True)
                self.decoder, dec_dims = _mlp_from_state_dict(sub, pfx)
                break
        else:
            raise KeyError("no g1_dyn/flat decoder in merged ckpt")
        assert dec_dims[-1] == NUM_G1, dec_dims
        self.fsq = FsqQuantizer(levels=32)
        self.codec = SonicPhiCodec(harvest_g1_params(), harvest_x2_params())
        remap_wrist_ranges(self.codec)
        if self.remap_waist:
            remap_waist_ranges(self.codec)
            print("[frozen-core-smpl] waist range-map APPLIED", flush=True)
        self.device_ = device
        self.to(device).eval()
        self.reset()

    def reset(self):
        self._g1_action_hist = np.zeros((NFRAMES, NUM_G1), np.float32)

    @torch.no_grad()
    def forward(self, proprioception, smpl_obs):
        prop_x2 = np.asarray(proprioception, np.float32).reshape(-1)
        obs = np.asarray(smpl_obs, np.float32).reshape(-1).copy()
        assert prop_x2.shape[0] == X2_PROP_DIM and obs.shape[0] == 840

        prop_g1 = self.codec.proprio_x2_to_g1(prop_x2, self._g1_action_hist)
        # rewrite wrist slots (per-frame layout [72 joints | 6 ori | 6 wrist])
        # with codec-mapped current G1 wrists from the newest proprio frame
        if os.environ.get("WRIST_OBS_ZERO"):
            # diagnostic: break the measured-wrist feedback echo (suspected
            # lock-in fixed point: wrist at limit -> obs reports extreme ->
            # policy commands extreme)
            wrist_g1 = np.zeros(6, np.float32)
        else:
            jp_rel_new = prop_x2[30 + 9 * NUM_X2: 30 + 10 * NUM_X2]
            q_g1 = self.codec.q_x2_to_g1(jp_rel_new + self.codec.d_x2)
            wrist_g1 = (q_g1[23:29]).astype(np.float32)  # G1 IL wrist dofs
        for f in range(10):
            obs[f * 84 + 78: f * 84 + 84] = wrist_g1

        t = lambda x: torch.from_numpy(x).to(self.device_)[None]  # noqa: E731
        token = self.fsq(self.smpl_encoder(t(obs)))
        act_g1 = self.decoder(torch.cat([token, t(prop_g1)], dim=-1))[0]
        act_g1 = act_g1.clamp(-G1_ACTION_CLIP, G1_ACTION_CLIP).cpu().numpy()
        self._g1_action_hist = np.roll(self._g1_action_hist, -1, axis=0)
        self._g1_action_hist[-1] = act_g1
        act_x2 = self.codec.action_g1_to_x2(act_g1.astype(np.float64))
        return torch.from_numpy(act_x2.astype(np.float32))[None]


def remap_wrist_ranges(codec) -> None:
    """Range-ratio wrist mapping (smpl composite path ONLY).

    The phi table's wrist rows are ~identity affines from the paired-clip
    calibration — but those clips barely moved the wrists, so the fit never
    saw that G1 wrist ranges (+/-93 deg pitch/yaw) dwarf X2's (pitch +/-32,
    roll -90..+41). A frozen-G1 core emitting G1-magnitude wrist motion
    through a ~1:1 map pins X2's wrists at their limits 80-100% of the time
    (operator: "left wrist is twisted", 2026-08-22).

    Fix: map source range onto target range (slope = span ratio, offset =
    center shift), applied consistently to decode (G1->X2, actions) and
    encode (X2->G1, wrist obs slots). Ranges are read from the same MJCFs
    the codec already treats as ground truth. Wrist YAW is left alone (X2's
    yaw range is LARGER than its G1 source — no saturation there).
    The shipped planner's _PHI_TABLE is untouched.
    """
    import mujoco
    from eval_x2_mujoco import MJCF_PATH as X2_MJCF
    g1_mjcf = str(REPO_ROOT / "gear_sonic" / "data" / "assets"
                  / "robot_description" / "mjcf" / "g1_29dof_rev_1_0.xml")
    models = {"x2": mujoco.MjModel.from_xml_path(X2_MJCF),
              "g1": mujoco.MjModel.from_xml_path(g1_mjcf)}

    def jrange(which, short_name):
        m = models[which]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                              short_name + "_joint")
        assert j >= 0, f"{short_name}_joint not in {which} MJCF"
        return m.jnt_range[j]

    p = codec.phi
    changed = []
    for xi, xn in enumerate(codec.x2p["il_names"]):
        if "wrist" not in xn or "yaw" in xn:
            continue
        gi = int(p["dec_src"][xi])
        gn = codec.g1p["il_names"][gi]
        lo_x, hi_x = jrange("x2", xn)
        lo_g, hi_g = jrange("g1", gn)
        a = (hi_x - lo_x) / (hi_g - lo_g) * np.sign(p["dec_a"][xi])
        b = (hi_x + lo_x) / 2.0 - a * (hi_g + lo_g) / 2.0
        p["dec_a"][xi], p["dec_b"][xi] = a, b
        assert int(p["enc_src"][gi]) == xi, (xn, gn)
        p["enc_a"][gi], p["enc_b"][gi] = 1.0 / a, -b / a
        changed.append(f"{xn}<-{gn} a={a:+.3f} b={np.degrees(b):+.1f}d")
    print("[wrist-range-map] " + "; ".join(changed), flush=True)


def remap_waist_ranges(codec) -> None:
    """Range-ratio WAIST pitch/roll mapping (S1 lineage, 2026-08-31 onward).

    The paired-clip fit gave waist_pitch dec_a = +1.287 — AMPLIFYING G1
    commands into a joint with 67% of G1's travel (measured on hardware in
    damp mode: stops at +/-0.349 = vendor +/-20deg; G1 +/-0.520). Net ~1.9x
    overdrive: every ordinary G1 forward lean maps past X2's stop. On the
    robot this produced the S0 waist snap-back (66 events / 9 min session
    2026-08-31: target railed backward while position pinned forward,
    ~0.75 rad stored error released at 3-5 rad/s). The fit was range-blind:
    the calibration clips never approached either robot's stops — the same
    disease remap_wrist_ranges() fixed for the wrists on 2026-08-22.

    Same math as the wrist remap: slope = span ratio, offset = center
    shift, applied to decode and encode consistently. Ranges are read from
    the MJCFs, whose waist rows now carry the HAND-MEASURED hardware stops.
    Waist yaw untouched (X2's yaw range envelops G1 motion adequately and
    its fitted slope is ~1.0).

    ERA GUARD: models trained WITHOUT this remap (armA/S0 lineages) must
    never be exported or wrapped with it — same shapes, silently different
    waist coefficients. It is opt-in everywhere (backbone.remap_waist_ranges
    in training; --remap-waist at export; remap_waist= in the actors).
    """
    import mujoco
    from eval_x2_mujoco import MJCF_PATH as X2_MJCF
    g1_mjcf = str(REPO_ROOT / "gear_sonic" / "data" / "assets"
                  / "robot_description" / "mjcf" / "g1_29dof_rev_1_0.xml")
    models = {"x2": mujoco.MjModel.from_xml_path(X2_MJCF),
              "g1": mujoco.MjModel.from_xml_path(g1_mjcf)}

    def jrange(which, short_name):
        m = models[which]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                              short_name + "_joint")
        assert j >= 0, f"{short_name}_joint not in {which} MJCF"
        return m.jnt_range[j]

    # TARGET = the VENDOR-RATED operational range (x2_joint_details sheet:
    # pitch -18~+18deg, roll +/-28deg), NOT the mechanical stops. The stops
    # sit ~2deg beyond rated (hand-measured 2026-08-31: +/-20deg, +/-30deg;
    # the MJCF jnt_range now carries the stops for honest referee physics).
    # Mapping G1's envelope onto rated keeps routine commands in-warranty
    # and leaves the stop margin free to absorb overshoot — which also
    # matches training, whose URDF limits are rated (x0.9 soft factor).
    RATED_X2 = {"waist_pitch": (-0.314159, 0.314159),
                "waist_roll": (-0.488692, 0.488692)}
    _rx = jrange("x2", "waist_pitch")
    assert _rx[1] >= RATED_X2["waist_pitch"][1] - 1e-6, (
        f"x2 MJCF waist_pitch physics range {_rx} is NARROWER than the "
        "rated command range — physics must envelop commands")

    p = codec.phi
    changed = []
    for xn in ("waist_pitch", "waist_roll"):
        xi = codec.x2p["il_names"].index(xn)
        gi = int(p["dec_src"][xi])
        gn = codec.g1p["il_names"][gi]
        lo_x, hi_x = RATED_X2[xn]
        lo_g, hi_g = jrange("g1", gn)
        a = (hi_x - lo_x) / (hi_g - lo_g) * np.sign(p["dec_a"][xi])
        b = (hi_x + lo_x) / 2.0 - a * (hi_g + lo_g) / 2.0
        p["dec_a"][xi], p["dec_b"][xi] = a, b
        assert int(p["enc_src"][gi]) == xi, (xn, gn)
        # TRUE inverse — deliberately NOT the wrist remap's 1/a,-b/a rows
        # (documented convention bug the older lineages absorbed; see the
        # 2dd05b1 commit message). Codec convention is enc==dec, and the
        # encode formula g1=(x2-b)/a does the inverting — so enc==dec here
        # makes the G1 core PERCEIVE X2's stop as its own full-lean value
        # instead of under-reporting it (the under-report fed the S0
        # push-into-the-stop loop). Roundtrip on these rows is exact.
        p["enc_a"][gi], p["enc_b"][gi] = a, b
        changed.append(f"{xn}<-{gn} a={a:+.3f} b={np.degrees(b):+.1f}d")
    print("[waist-range-map] " + "; ".join(changed), flush=True)


class SmplG1Actor(nn.Module):
    """SMPL-conditioned NATIVE G1 actor — no codec, no embodiment transfer.

    smpl encoder + g1_dyn decoder straight from a G1-core checkpoint:
      forward(proprioception(930 G1 IL), smpl_obs(840)) -> (1, 29).

    decoder_ckpt lets a fine-tune's decoder (e.g. v11, flat naming +
    unmerged LoRA — folded here) replace the release decoder while the
    release supplies the smpl encoder. With decoder_ckpt=None both come
    from release_ckpt (pure v1.1 or original release).

    Wrist slots: the encoder expects the robot's own wrist dofs — on a G1
    that is just measured IL [23..29), taken from the newest proprio frame
    (G1 wrist defaults are 0, so jpos_rel == absolute there).
    """

    def __init__(self, decoder_ckpt: str | None = None,
                 release_ckpt: str | None = None, device: str = "cpu"):
        super().__init__()
        from gear_sonic.trl.modules.onnx_helpers import (  # noqa: E402
            FsqQuantizer, _mlp_from_state_dict, tolerant_torch_load)
        src = (tolerant_torch_load(str(decoder_ckpt))["policy_state_dict"]
               if decoder_ckpt else None)
        has_enc = src is not None and any(
            k.startswith("actor_module.encoders.smpl.") for k in src)
        if release_ckpt is None and has_enc:
            # single-path form: the ckpt IS a full G1 core (v1.1 / release)
            release_path, rel = str(decoder_ckpt), src
        else:
            release_path = str(release_ckpt or FrozenCoreG1SonicActor.HF_CKPT)
            rel = tolerant_torch_load(release_path)["policy_state_dict"]
        self.smpl_encoder, enc_dims = _mlp_from_state_dict(
            rel, "actor_module.encoders.smpl.module.")
        assert enc_dims[0] == 840, enc_dims
        n_lin = len([k for k in rel
                     if k.startswith("actor_module.decoders.g1_dyn.module.")
                     and k.endswith(".weight")])
        self.ori_mode = ("heading" if n_lin >= 9 or "v1_1" in release_path
                         else "full")
        if src is None:
            src = rel
        for pfx in ("actor_module.decoders.g1_dyn.module.",
                    "actor_module.decoder."):
            if any(k.startswith(pfx) for k in src):
                LORA_SCALING = 1.0  # alpha/rank = 16/16
                sub, n_folded = {}, 0
                for k, v in src.items():
                    if not k.startswith(pfx) or k.endswith((".lora_A",
                                                            ".lora_B")):
                        continue
                    if k.endswith(".weight") and k[:-7] + ".lora_A" in src:
                        v = (v.double() + LORA_SCALING *
                             (src[k[:-7] + ".lora_B"].double()
                              @ src[k[:-7] + ".lora_A"].double())).float()
                        n_folded += 1
                    sub[k] = v
                if n_folded:
                    print(f"[smpl-g1] folded LoRA into {n_folded} "
                          "decoder linears", flush=True)
                self.decoder, dec_dims = _mlp_from_state_dict(sub, pfx)
                break
        else:
            raise KeyError("no g1_dyn/flat decoder in checkpoint")
        assert dec_dims[-1] == NUM_G1, dec_dims
        self.fsq = FsqQuantizer(levels=32)
        self.device_ = device
        self.to(device).eval()

    def reset(self):
        pass  # stateless — G1 proprio comes straight from the driver

    @torch.no_grad()
    def forward(self, proprioception, smpl_obs):
        prop = np.asarray(proprioception, np.float32).reshape(-1)
        obs = np.asarray(smpl_obs, np.float32).reshape(-1).copy()
        assert prop.shape[0] == G1_PROP_DIM and obs.shape[0] == 840
        jp_rel_new = prop[30 + 9 * NUM_G1: 30 + 10 * NUM_G1]
        wrist = jp_rel_new[23:29].astype(np.float32)
        for f in range(10):
            obs[f * 84 + 78: f * 84 + 84] = wrist
        t = lambda x: torch.from_numpy(x).to(self.device_)[None]  # noqa: E731
        token = self.fsq(self.smpl_encoder(t(obs)))
        act = self.decoder(torch.cat([token, t(prop)], dim=-1))[0]
        act = act.clamp(-G1_ACTION_CLIP, G1_ACTION_CLIP)
        return act.cpu()[None]
