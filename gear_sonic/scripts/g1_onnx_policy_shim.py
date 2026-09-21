#!/usr/bin/env python3
"""Drop-in ``model.policy`` shim that runs the STOCK G1 SONIC ONNX inside the
IsaacLab ``im_eval`` vectorized sweep.

Why this exists
---------------
The released G1 SONIC policy ships only as an **encoder + decoder ONNX pair**
(``gear_sonic_deploy/policy/release/model_{encoder,decoder}.onnx``); there is no
gear_sonic-native ``.pt`` checkpoint. ``eval_agent_trl`` / ``im_eval`` normally
``torch.load`` a ``UniversalTokenActor`` ``.pt`` and drive ``model.policy``. This
class presents the *same* interface but forwards to onnxruntime, so the stock
ONNX can be swept over the whole G1 reference corpus for the feasibility
experiment.

Obs layout (verified against the deploy C++ obs registry)
---------------------------------------------------------
Encoder ONNX:  obs_dict[1, 1762] -> encoded_tokens[1, 64]
Decoder ONNX:  obs_dict[1, 994]  -> action[1, 29]

The **1762** encoder vector is the concatenation, in this order, of the deploy
``encoder_observations`` (see observation_config.yaml + g1_deploy_onnx_ref.cpp
GetObservationRegistry). For the **g1 encoder mode (mode_id 0)** the deploy
zero-fills every term except its ``required_observations``
(``encoder_mode_4``, ``motion_joint_positions_10frame_step5``,
``motion_joint_velocities_10frame_step5``,
``motion_anchor_orientation_10frame_step5``) -- so only those 4 segments are
populated here and the rest stay zero. ``encoder_mode_4`` for g1 = [0,0,0,0]
(it is ``[GetEncodeMode(), 0, 0, 0]`` and g1's mode id is 0).

The **994** decoder vector is ``[token_state(64) | proprioception(930)]`` where
proprioception is the deploy-ordered 10-frame history stack
``[base_angular_velocity(30) | body_joint_positions(290) |
body_joint_velocities(290) | last_actions(290) | gravity_dir(30)]``.

Everything is in **IsaacLab / URDF joint order** (the ONNX training convention);
the deploy only remaps to hardware/MuJoCo order at its motor I/O boundary, so
inside IsaacLab no joint permutation is applied here.

Design note: this shim is deliberately *stateless per step* -- the released
actor uses ``max_rollout_history=1`` so ``init_rollout``/``clear_rollout`` are
no-ops. All temporal context lives inside the observation (10-frame history +
future reference frames), not in a recurrent policy buffer.
"""

from __future__ import annotations

import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper


def _rewrite_dynamic_batch(onnx_path: str) -> bytes:
    """Return serialized ONNX bytes with a dynamic (symbolic) batch axis.

    The stock G1 ONNX was exported with a hard batch size of 1 (the deploy runs
    one robot); the encoder additionally bakes batch=1 into internal ``Reshape``
    target shapes fed by ``Constant`` nodes. To sweep many envs at once we make
    the leading axis symbolic and rewrite every Reshape whose leading target dim
    is a literal 1 to ``-1`` (infer). Verified numerically: at batch=1 this is
    bit-identical to the original, and on realistic g1-mode observations batched
    output equals per-row output exactly (the only divergence is worst-case FSQ
    boundary flips under unit-gaussian garbage in the zero-filled segments, which
    never occurs for real obs).
    """
    m = onnx.load(onnx_path)
    for t in list(m.graph.input) + list(m.graph.output):
        d0 = t.type.tensor_type.shape.dim[0]
        d0.dim_param = "N"
        d0.ClearField("dim_value")
    reshape_shape_inputs = {
        n.input[1] for n in m.graph.node if n.op_type == "Reshape" and len(n.input) >= 2
    }
    for init in m.graph.initializer:
        if init.name in reshape_shape_inputs:
            arr = numpy_helper.to_array(init).copy()
            if arr.ndim == 1 and arr.size >= 1 and arr[0] == 1:
                arr[0] = -1
                init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in m.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] in reshape_shape_inputs:
            for a in node.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t).copy()
                    if arr.ndim == 1 and arr.size >= 1 and arr[0] == 1:
                        arr[0] = -1
                        a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
    return m.SerializeToString()


# --- Encoder input (1762) segment offset map -------------------------------
# name -> (offset, dim). Order == observation_config.yaml encoder_observations.
ENCODER_SEGMENTS = {
    "encoder_mode_4": (0, 4),
    "motion_joint_positions_10frame_step5": (4, 290),
    "motion_joint_velocities_10frame_step5": (294, 290),
    "motion_root_z_position_10frame_step5": (584, 10),
    "motion_root_z_position": (594, 1),
    "motion_anchor_orientation": (595, 6),
    "motion_anchor_orientation_10frame_step5": (601, 60),
    "motion_joint_positions_lowerbody_10frame_step5": (661, 120),
    "motion_joint_velocities_lowerbody_10frame_step5": (781, 120),
    "vr_3point_local_target": (901, 9),
    "vr_3point_local_orn_target": (910, 12),
    "smpl_joints_10frame_step1": (922, 720),
    "smpl_anchor_orientation_10frame_step1": (1642, 60),
    "motion_joint_positions_wrists_10frame_step1": (1702, 60),
}
ENCODER_INPUT_DIM = 1762

# Terms the g1 encoder mode (mode_id 0) actually populates; the rest are zero.
G1_MODE_REQUIRED = (
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    "motion_anchor_orientation_10frame_step5",
    # encoder_mode_4 stays [0,0,0,0] for g1 -> nothing to write.
)

# --- Decoder input (994) segment offset map --------------------------------
TOKEN_DIM = 64
DECODER_PROP_SEGMENTS = {
    "his_base_angular_velocity_10frame_step1": (0, 30),
    "his_body_joint_positions_10frame_step1": (30, 290),
    "his_body_joint_velocities_10frame_step1": (320, 290),
    "his_last_actions_10frame_step1": (610, 290),
    "his_gravity_dir_10frame_step1": (900, 30),
}
DECODER_PROP_DIM = 930
DECODER_INPUT_DIM = TOKEN_DIM + DECODER_PROP_DIM  # 994
ACTION_DIM = 29


class G1OnnxPolicyShim(torch.nn.Module):
    """onnxruntime-backed stand-in for ``model.policy`` (a gear_sonic ``Actor``).

    Subclasses ``nn.Module`` so it can be assigned as ``PolicyAndValueWrapper.policy``
    (which registers its policy as a child module). It owns no parameters, so
    ``state_dict()`` is empty and ``eval()``/``to()`` are inherited no-ops.

    Parameters
    ----------
    encoder_onnx_path, decoder_onnx_path : str
        Paths to the stock G1 SONIC encoder/decoder ONNX files.
    device : torch.device
        Device the returned action tensors live on (env device).
    encoder_obs_keys : dict[str, str]
        Maps the 3 populated g1-mode encoder segment names to the obs_dict keys
        that carry them, e.g.
        ``{"motion_joint_positions_10frame_step5": "g1enc_joint_pos", ...}``.
    proprio_obs_key : str
        obs_dict key carrying the 930-d deploy-ordered proprioception stack.
        (If proprioception is assembled from separate history terms, pass a
        dict via ``proprio_obs_keys`` instead.)
    proprio_obs_keys : dict[str, str] | None
        Optional per-segment mapping for proprioception (deploy order). If given,
        overrides ``proprio_obs_key``.
    providers : list[str] | None
        onnxruntime execution providers. Defaults to CUDA then CPU.
    """

    def __init__(
        self,
        encoder_onnx_path: str,
        decoder_onnx_path: str,
        device,
        encoder_obs_keys: dict | None = None,
        proprio_obs_key: str | None = None,
        proprio_obs_keys: dict | None = None,
        providers: list | None = None,
        env=None,
        command_name: str = "motion",
        actor_obs_key: str = "actor_obs",
    ):
        super().__init__()
        if providers is None:
            avail = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in avail
                else ["CPUExecutionProvider"]
            )
        self.device = device
        self.encoder = ort.InferenceSession(
            _rewrite_dynamic_batch(encoder_onnx_path), providers=providers
        )
        self.decoder = ort.InferenceSession(
            _rewrite_dynamic_batch(decoder_onnx_path), providers=providers
        )
        self._enc_in = self.encoder.get_inputs()[0].name
        self._enc_out = self.encoder.get_outputs()[0].name
        self._dec_in = self.decoder.get_inputs()[0].name
        self._dec_out = self.decoder.get_outputs()[0].name

        self.encoder_obs_keys = dict(encoder_obs_keys) if encoder_obs_keys else None
        self.proprio_obs_key = proprio_obs_key
        self.proprio_obs_keys = dict(proprio_obs_keys) if proprio_obs_keys else None

        # env-mode: pull encoder terms from the command manager and reorder the
        # framework's actor_obs into deploy proprioception order.
        self.env = env
        self.command_name = command_name
        self.actor_obs_key = actor_obs_key
        self._proprio_reorder = None  # lazily-built slice plan from actor_obs

        # optional executed-trajectory recording (env var G1_SHIM_RECORD_DIR)
        self._rec_dir = os.environ.get("G1_SHIM_RECORD_DIR")
        self._rec_state = None  # lazily-built per-env recording state (see _record_step)
        if self._rec_dir:
            # Terminated (fallen) envs never reach their clip length, so their
            # buffers would be dropped on exit — flush partials so falls are
            # captured too (the post-fall frozen tail is part of the record).
            import atexit
            atexit.register(self._flush_unfinished)

        # Sanity: the ONNX I/O dims must match our offset maps.
        enc_shape = self.encoder.get_inputs()[0].shape
        dec_shape = self.decoder.get_inputs()[0].shape
        assert enc_shape[-1] == ENCODER_INPUT_DIM, (
            f"encoder ONNX expects {enc_shape[-1]}, map is {ENCODER_INPUT_DIM}"
        )
        assert dec_shape[-1] == DECODER_INPUT_DIM, (
            f"decoder ONNX expects {dec_shape[-1]}, map is {DECODER_INPUT_DIM}"
        )

    # --- interface expected by im_eval / eval_agent_trl --------------------
    # eval()/state_dict() are inherited from nn.Module (no params -> empty state).
    def to(self, *args, **kwargs):  # noqa: D102 - shim owns no params; keep env device
        return self

    def load_state_dict(self, *a, **k):  # no weights to load (ONNX carries them)
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def init_rollout(self):  # single-frame policy: nothing to reset
        pass

    def clear_rollout(self):
        pass

    def eval_mode(self):
        pass

    def train_mode(self):
        pass

    @property
    def act_inference(self):
        # im_eval does ``policy.act_inference`` then calls it -- expose the bound
        # method so both ``policy.act_inference(...)`` and the attribute work.
        return self._act_inference

    def rollout(self, obs_dict=None, **kwargs):
        # The run_once / render playback path (eval_agent_trl) calls
        # ``policy.rollout(obs_dict=...)``, reads ``policy.action_mean`` for the
        # action, and indexes the return dict's ``"obs_dict"`` for env.step.
        # This single-frame ONNX policy is deterministic, so action == action_mean.
        actions = self._act_inference(obs_dict, **kwargs)
        self.action_mean = actions
        return {"actions": actions, "obs_dict": obs_dict}

    # --- core forward ------------------------------------------------------
    def _to_np(self, t):
        return t.detach().to("cpu", torch.float32).numpy()

    def _encoder_segments_from_env(self):
        """Pull the 3 populated g1-mode encoder terms from the command manager.

        All three are reference-motion future frames in IsaacLab joint order
        (matching the deploy's motion joint order after its hardware remap):
          motion_joint_positions_10frame_step5  <- command_multi_future_joint_pos
          motion_joint_velocities_10frame_step5 <- joint_vel_multi_future
          motion_anchor_orientation_10frame_step5 <- motion_anchor_ori_b_mf
        """
        from gear_sonic.envs.manager_env.mdp import observations as _obs

        cmd = self.env.command_manager.get_term(self.command_name)
        return {
            "motion_joint_positions_10frame_step5": cmd.command_multi_future_joint_pos,
            "motion_joint_velocities_10frame_step5": cmd.joint_vel_multi_future,
            "motion_anchor_orientation_10frame_step5": _obs.motion_anchor_ori_b_mf(
                self.env, self.command_name, non_flatten=False
            ),
        }

    def _build_encoder_input(self, obs_dict, num_envs):
        enc = np.zeros((num_envs, ENCODER_INPUT_DIM), dtype=np.float32)
        source = (
            self._encoder_segments_from_env()
            if self.env is not None
            else {k: obs_dict[v] for k, v in self.encoder_obs_keys.items()}
        )
        for seg_name in G1_MODE_REQUIRED:
            off, dim = ENCODER_SEGMENTS[seg_name]
            vals = self._to_np(source[seg_name])
            assert vals.shape[-1] == dim, (
                f"encoder seg {seg_name}: dim {vals.shape[-1]}, expected {dim}"
            )
            enc[:, off : off + dim] = vals
        return enc

    def _build_proprio(self, obs_dict, num_envs):
        # The framework `policy` obs group is emitted in term order
        #   [base_ang_vel(30) | joint_pos(290) | joint_vel(290) | actions(290) | gravity_dir(30)]
        # which is EXACTLY the deploy decoder proprioception order
        #   [base_angular_velocity | body_joint_positions | body_joint_velocities
        #    | last_actions | gravity_dir]  (verified via observation_manager dump).
        # So actor_obs is fed verbatim -- no reorder.
        if self.env is not None or (
            self.proprio_obs_keys is None and self.actor_obs_key in obs_dict
        ):
            actor_obs = self._to_np(obs_dict[self.actor_obs_key])
            assert actor_obs.shape[-1] == DECODER_PROP_DIM, (
                f"actor_obs dim {actor_obs.shape[-1]} != {DECODER_PROP_DIM}"
            )
            return actor_obs.astype(np.float32, copy=False)
        if self.proprio_obs_keys is None:
            prop = self._to_np(obs_dict[self.proprio_obs_key])
            assert prop.shape[-1] == DECODER_PROP_DIM
            return prop.astype(np.float32, copy=False)
        prop = np.zeros((num_envs, DECODER_PROP_DIM), dtype=np.float32)
        for seg_name, key in self.proprio_obs_keys.items():
            off, dim = DECODER_PROP_SEGMENTS[seg_name]
            vals = self._to_np(obs_dict[key])
            assert vals.shape[-1] == dim
            prop[:, off : off + dim] = vals
        return prop

    @torch.no_grad()
    def _act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        if self.env is not None:
            num_envs = self.env.num_envs
        else:
            any_key = self.encoder_obs_keys[G1_MODE_REQUIRED[0]]
            num_envs = obs_dict[any_key].shape[0]

        enc_in = self._build_encoder_input(obs_dict, num_envs)
        tokens = self.encoder.run([self._enc_out], {self._enc_in: enc_in})[0]  # (N,64)

        prop = self._build_proprio(obs_dict, num_envs)
        dec_in = np.concatenate([tokens, prop], axis=1).astype(np.float32)  # (N,994)
        actions = self.decoder.run([self._dec_out], {self._dec_in: dec_in})[0]  # (N,29)

        dbg = os.environ.get("G1_SHIM_DEBUG")
        if dbg and not getattr(self, "_dbg_done", False):
            self._dbg_done = True
            self._dump_debug(dbg, obs_dict, enc_in, prop, tokens, actions)

        if self._rec_dir is not None:
            self._record_step()

        return torch.from_numpy(np.ascontiguousarray(actions)).to(self.device)

    # soma CSV joint column order (== input g1_recorded CSVs; MuJoCo/URDF order)
    _SOMA_JOINTS = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
        "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]

    def _record_step(self):
        """Record each env's executed pose, ONE CSV per clip.

        Correct across multiple ``im_eval`` env-loops: between loops the envs are
        reassigned to new clips, so we key each env by its CURRENT motion id and
        start a fresh clip whenever it changes. Each clip's CSV is written exactly
        once, when the clip reaches its length -- O(frames) I/O, not the O(frames^2)
        of repeatedly rewriting a growing CSV. Buffers are freed on flush.
        """
        from scipy.spatial.transform import Rotation as R

        robot = self.env.scene["robot"]
        cmd = self.env.command_manager.get_term(self.command_name)

        # Use the motion-lib's GLOBAL per-env current motion ids (updated every
        # env-loop by forward_motion_samples). cmd.motion_ids is env-local
        # (0..num_envs-1) and constant across loops, so it can't key clips here.
        cur = cmd.motion_lib._curr_motion_ids
        ids = cur.detach().cpu().numpy()
        lens = cmd.motion_lib.get_motion_num_steps(cur).detach().cpu().numpy()

        # Size per-env state to the ACTUAL id array length. IsaacLab can round an
        # odd num_envs up to a grid, so self.env.num_envs may exceed len(ids);
        # keying off len(ids) (and the min-cap below) avoids an IndexError.
        n = len(ids)
        if self._rec_state is None:
            os.makedirs(self._rec_dir, exist_ok=True)
            names = list(robot.data.joint_names)  # IsaacLab order
            self._soma_reindex = [names.index(j) for j in self._SOMA_JOINTS]
            self._rec_state = {
                "mid": np.full(n, -1, dtype=np.int64),  # current motion id per env
                "key": [None] * n,
                "buf": [None] * n,  # {"root","eul","dof"} lists for the current clip
                "len": np.zeros(n, dtype=np.int64),  # clip length (eval steps)
                "step": np.zeros(n, dtype=np.int64),  # frames recorded for current clip
                "done": np.zeros(n, dtype=bool),  # clip already flushed
                "keys": cmd.motion_lib._motion_data_keys,
            }
        st = self._rec_state

        origins = self.env.scene.env_origins.detach().cpu().numpy()
        root = (robot.data.root_pos_w.detach().cpu().numpy() - origins) * 100.0  # cm
        quat_wxyz = robot.data.root_quat_w.detach().cpu().numpy()
        eul = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        jp = np.rad2deg(robot.data.joint_pos.detach().cpu().numpy())[:, self._soma_reindex]

        for i in range(min(n, len(st["mid"]), len(root))):
            if ids[i] != st["mid"][i]:  # a new clip was assigned to this env
                if st["buf"][i] is not None and not st["done"][i]:
                    self._flush_clip(i)  # safety: flush an unfinished prior clip
                st["mid"][i] = ids[i]
                st["key"][i] = str(st["keys"][ids[i]])
                st["len"][i] = int(lens[i])
                st["step"][i] = 0
                st["done"][i] = False
                st["buf"][i] = {"root": [], "eul": [], "dof": []}
            if st["done"][i]:
                continue
            b = st["buf"][i]
            b["root"].append(root[i]); b["eul"].append(eul[i]); b["dof"].append(jp[i])
            st["step"][i] += 1
            if st["step"][i] >= st["len"][i]:  # clip complete -> write once
                self._flush_clip(i)
                st["done"][i] = True
            elif st["step"][i] % 250 == 0:
                # periodic partial write: Isaac hard-exits without running
                # atexit, so clips cut short (falls, eval end) would otherwise
                # vanish. Keep the buffer; the final write overwrites.
                self._flush_clip(i, keep_buffer=True)

    def _flush_unfinished(self):
        """atexit: write partial CSVs for envs that never completed their clip
        (terminated/fallen). Without this, exactly the clips where the robot
        fell — the most interesting ones to review — are silently missing."""
        st = self._rec_state
        if st is None:
            return
        for i in range(len(st["mid"])):
            if st["buf"][i] is not None and not st["done"][i]:
                self._flush_clip(i)
                st["done"][i] = True

    def _flush_clip(self, i, keep_buffer=False):
        st = self._rec_state
        buf = st["buf"][i]
        if not buf or not buf["root"]:
            return
        header = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
                  "root_rotateX", "root_rotateY", "root_rotateZ"] + [j + "_dof" for j in self._SOMA_JOINTS]
        rt = np.asarray(buf["root"]); eu = np.asarray(buf["eul"]); df = np.asarray(buf["dof"])
        m = len(rt)
        rows = np.concatenate([np.arange(m)[:, None].astype(np.float64), rt, eu, df], axis=1)
        key = st["key"][i]
        tmp = os.path.join(self._rec_dir, f".{key}.csv.tmp")
        out = os.path.join(self._rec_dir, f"{key}.csv")
        np.savetxt(tmp, rows, delimiter=",", header=",".join(header), comments="", fmt="%.6f")
        os.replace(tmp, out)
        if not keep_buffer:
            st["buf"][i] = None  # free memory

    def _dump_debug(self, path, obs_dict, enc_in, prop, tokens, actions):
        info = {
            "enc_in": enc_in,
            "prop": prop,
            "tokens": tokens,
            "actions": actions,
            "obs_dict_keys": {k: tuple(v.shape) for k, v in obs_dict.items()},
        }
        # true actor-group term order + dims (to verify the block-reorder map)
        try:
            om = self.env.observation_manager
            for grp in om.active_terms:
                info[f"group::{grp}"] = list(zip(om.active_terms[grp], om.group_obs_term_dim[grp]))
        except Exception as e:  # noqa: BLE001
            info["obs_manager_err"] = repr(e)
        # raw robot + reference state at this instant
        try:
            robot = self.env.scene["robot"]
            cmd = self.env.command_manager.get_term(self.command_name)
            info["robot_joint_names"] = list(robot.data.joint_names)
            info["robot_joint_pos0"] = robot.data.joint_pos[0].detach().cpu().numpy()
            info["robot_default_joint_pos0"] = robot.data.default_joint_pos[0].detach().cpu().numpy()
            info["robot_root_quat_w0"] = robot.data.root_quat_w[0].detach().cpu().numpy()
            info["ref_joint_pos0"] = cmd.joint_pos[0].detach().cpu().numpy() if hasattr(cmd, "joint_pos") else None
        except Exception as e:  # noqa: BLE001
            info["robot_state_err"] = repr(e)
        import pickle
        with open(path, "wb") as f:
            pickle.dump({k: (v if not hasattr(v, "shape") else np.asarray(v)) for k, v in info.items()}, f)
        print(f"[G1OnnxPolicyShim] wrote step-0 debug dump to {path}", flush=True)
