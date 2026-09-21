"""Robot-agnostic streaming wrapper around motion_inference for velocity-intent control.

This module is the neural counterpart to the G1 demo's `full_navigation_agent`
([`motionbricks/motionbricks/motion_backbone/demo/full_agent.py`](../../demo/full_agent.py)).

Two prediction paths are supported:

  1. ``replan_with_velocity(intent)`` — velocity-only inference. Target body
     poses are zeroed and masked off; only the velocity intent (and optionally
     the implied target world-frame position) constrains the target window.
     This was the original X2 path; it's kept for ablation and as a fallback
     when no clip library is loaded.
  2. ``replan_with_pose_template(intent, mode_idx)`` — pose-template
     inference matching the SONIC / MotionBricks paper design and the G1
     demo. A representative 4-frame segment from the mode's reference clip
     is sampled, realigned to the implied target root position / heading,
     and fed as the target keyframe constraint to the in-betweener.
     Requires a baked clip library (``out/X2-clip.ckpt`` for X2, produced
     by ``motionbricks/scripts/build_x2_planner_clips.py``).

Both paths share:

  - The 4-frame past-context window (canonicalized + converted from MuJoCo qpos).
  - The implied target root xz / heading computed by integrating
    ``velocity_intent`` over ``TARGET_HORIZON_S``.
  - The post-prediction decode / uncanonicalize / FILTER_QPOS blend.

The 8-frame constraint window and the constraint-mask convention follow the
inference contract documented in
[`motion_inference.py`](motion_inference.py):

  - First 4 frames (numConstrainFrames=NUM_FRAMES_PER_TOKEN=4): past/current
    context; all has_* masks must be True for these.
  - Last 4 frames: target keyframe at the end of the predicted motion.
  - num_tokens drives how many tokens (=4 frames each) the model generates
    between the past context and the target keyframe.

Constructor takes the robot-specific MuJoCo converter (must implement
`convert_motion_features_to_mujoco_qpos` and
`convert_mujoco_qpos_to_motion_transforms`) plus the `motion_inference` wrapper.
Optionally takes a path to the pose-template clip library; if omitted, only
the velocity-only path is available.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch as t
from torch import nn

from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motionlib.core.utils.rotations import (
    angle_to_Y_rotation_matrix,
    matrix_to_cont6d,
    quaternion_to_matrix,
)


# ---------------------------------------------------------------------------
# Z-axis rotation helper (kept verbatim from full_agent.py for MuJoCo qpos
# canonicalization, where the gravity axis in the wire is +z).
# ---------------------------------------------------------------------------


def angle_to_Z_rotation_matrix(angle: t.Tensor) -> t.Tensor:
    """Rotation matrix around Z-axis for Z-up MuJoCo qpos canonicalization."""
    cos, sin = t.cos(angle), t.sin(angle)
    one, zero = t.ones_like(angle), t.zeros_like(angle)
    mat = t.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1)
    mat = mat.reshape(angle.shape + (3, 3))
    return mat


# ---------------------------------------------------------------------------
# X2 mode-indexed clip library (analogous to G1's `clip_holder_G1`).
#
# The on-disk ckpt is produced by
# ``motionbricks/scripts/build_x2_planner_clips.py`` and stores stacked
# per-mode buffers in motion-rep Y-up coordinates. Buffer layout matches
# what ``clip_holder._preprocess_clips_from_dataloader`` (G1) produces:
#
#   global_root_positions   [num_modes, max_frames, 3]   # Y=0 (gravity zeroed)
#   global_joint_positions  [num_modes, max_frames, J, 3] # root-XZ-relative
#   global_joint_rotations  [num_modes, max_frames, J, 3, 3]
#   global_headings         [num_modes, max_frames]
#   mujoco_qpos             [num_modes, max_frames, 38]
#   num_frames_per_clip     [num_modes]                   # int32
# ---------------------------------------------------------------------------


class _X2ClipLibrary(nn.Module):
    """Loaded clip library for pose-template inference.

    Modes are addressed by integer index; the binding contract between
    indices and mode names is defined by the bake script's ``DEFAULT_MODES``
    tuple (in ``motionbricks/scripts/build_x2_planner_clips.py``) and
    captured in the sidecar JSON next to the ckpt. The library does NOT
    read the sidecar — the caller is responsible for passing the right
    integer.

    Sampling: ``sample_target_segment(mode_idx, random_seed)`` returns a
    4-frame slice of the buffers, with frame-start chosen deterministically
    by ``random_seed % (num_frames - NUM_FRAMES_PER_TOKEN)``. This matches
    the segment-sampling logic in
    ``full_navigation_agent._generate_target_joint_transforms``
    (full_agent.py:328-338).
    """

    NUM_FRAMES_PER_TOKEN: int = 4

    def __init__(self, ckpt_path: Union[str, Path], device: str = "cuda") -> None:
        super().__init__()
        state = t.load(str(ckpt_path), map_location="cpu", weights_only=False)
        required = (
            "global_root_positions",
            "global_joint_positions",
            "global_joint_rotations",
            "global_headings",
            "mujoco_qpos",
            "num_frames_per_clip",
        )
        for key in required:
            if key not in state:
                raise KeyError(
                    f"X2 clip library at {ckpt_path} missing buffer {key!r}"
                )
            self.register_buffer(key, state[key].to(device))
        self._device = device
        self._num_modes = int(self.num_frames_per_clip.shape[0])
        self._num_joints = int(self.global_joint_positions.shape[2])

    @property
    def num_modes(self) -> int:
        return self._num_modes

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def clamp_mode_idx(self, mode_idx: int) -> int:
        """Clamp a mode index into ``[0, num_modes)``. Matches G1 demo safety."""
        if mode_idx < 0:
            return 0
        if mode_idx >= self._num_modes:
            return self._num_modes - 1
        return int(mode_idx)

    def sample_target_segment(
        self, mode_idx: int, random_seed: Optional[int] = None
    ) -> dict:
        """Return a 4-frame slice of buffers for ``mode_idx`` at a random start.

        Args:
            mode_idx: integer index into the mode table.
            random_seed: Python int. ``None`` -> torch.randint at call time.

        Returns:
            dict with keys (each tensor lives on ``self._device``):

                root_positions      [4, 3]       # Y=0
                joint_positions     [4, J, 3]    # root-XZ-relative, full Y
                joint_rotations     [4, J, 3, 3]
                headings            [4]
        """
        m = self.clamp_mode_idx(int(mode_idx))
        n = int(self.num_frames_per_clip[m].item())
        max_start = max(1, n - self.NUM_FRAMES_PER_TOKEN)
        if random_seed is None:
            seed = int(t.randint(0, 1_000_000, (1,)).item())
        else:
            seed = int(random_seed)
        start = seed % max_start
        sl = slice(start, start + self.NUM_FRAMES_PER_TOKEN)
        return {
            "root_positions": self.global_root_positions[m, sl].clone(),
            "joint_positions": self.global_joint_positions[m, sl].clone(),
            "joint_rotations": self.global_joint_rotations[m, sl].clone(),
            "headings": self.global_headings[m, sl].clone(),
        }


class NeuralPlannerCore(nn.Module):
    """Streaming neural locomotion planner driven by velocity intent.

    Lifecycle:

        core = NeuralPlannerCore(inferencer, converter)
        core.reset(init_mujoco_qpos)              # seed ring buffer with a stand frame
        while True:
            qpos = core.get_next_frame()           # pop one [38]-D qpos per tick
            if core.should_replan():
                core.replan_with_velocity(intent_vector_4d)
    """

    NUM_FRAMES_PER_TOKEN: int = 4
    DEFAULT_PRED_OFFSETS: int = 4
    NUM_MIN_FRAMES_IN_BUFFER: int = 64
    DEFAULT_PLANNING_HORIZON_S: float = 1.0
    # Default time-horizon over which to integrate velocity_intent into the
    # implied target_global_root_values constraint. ~2.13 s matches the
    # model's MASKED_NUM_TOKENS * NUM_FRAMES_PER_TOKEN / fps default at
    # 30 fps, i.e. the natural full-horizon length the model was
    # trained to predict per replan window.
    DEFAULT_TARGET_HORIZON_S: float = 2.0

    def __init__(
        self,
        inferencer: motion_inference,
        converter: nn.Module,
        device: str = "cuda",
        pred_offsets: int = DEFAULT_PRED_OFFSETS,
        filter_qpos: bool = True,
        force_canonicalization: bool = True,
        skip_ending_target_cond: bool = True,
        replan_threshold_frames: int = 16,
        use_implied_target_pos: bool = True,
        target_horizon_s: float = DEFAULT_TARGET_HORIZON_S,
        clip_library_ckpt: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self._inferencer = inferencer.eval().to(device)
        self._motion_rep = deepcopy(inferencer.motion_rep).to(device)
        self._converter = converter.to(device)
        self._device = device
        self._fps = self._motion_rep.fps

        self.PRED_OFFSETS = pred_offsets
        self.FILTER_QPOS = filter_qpos
        self.FORCE_CANONICALIZATION = force_canonicalization
        self.SKIP_ENDING_TARGET_COND = skip_ending_target_cond
        # When True, build target_global_root_values from
        # ``last_context_pos + velocity_intent * target_horizon_s`` and
        # mark has_global_root_values[:, -NUM_FT:] = True. This matches
        # the training distribution (the G1 demo's
        # ``target_root_realignment=True`` path) and is what unlocks
        # actual velocity tracking. Set to False only to reproduce the
        # legacy velocity-only behavior for ablation diagnostics.
        self.USE_IMPLIED_TARGET_POS = use_implied_target_pos
        self.TARGET_HORIZON_S = float(target_horizon_s)
        self.REPLAN_THRESHOLD_FRAMES = replan_threshold_frames

        # Optional clip library for pose-template inference. When loaded,
        # ``replan_with_pose_template(intent, mode_idx)`` becomes available.
        # The velocity-only path (``replan_with_velocity``) does not use it.
        self._clip_library: Optional[_X2ClipLibrary] = None
        if clip_library_ckpt is not None:
            self._clip_library = _X2ClipLibrary(clip_library_ckpt, device=device)

        self.frames: dict = {
            "model_features": None,  # [B, T, feat_dim]
            "mujoco_qpos": None,     # [B, T, qpos_dim]
        }
        self._current_frame_idx = 0
        self._qpos_dim: Optional[int] = None

        # ------------------------------------------------------------------
        # Output resampling (30 Hz model -> control-loop output_fps) + an
        # 8-frame cross-fade blend between successive planner outputs.
        #
        # This mirrors the G1 stock deployment stack (see
        # ``docs/source/references/planner_onnx.md`` "Output Resampling" and
        # "Animation Blending"). The plain integer ``get_next_frame`` path
        # advances +1 whole 30 Hz frame per call; when a 50 Hz publish loop
        # drives it that plays the 30 Hz content 1.67x too fast, so the
        # reference outruns the tracking policy. ``get_next_frame_resampled``
        # instead advances a FLOAT read cursor by ``model_fps / output_fps``
        # native frames per output tick and interpolates (lerp on
        # translation + dof, slerp on the root quaternion).
        #
        # State is inert until ``get_next_frame_resampled`` is first called
        # (``_resample_active`` stays False), so the legacy integer path and
        # the offline replay tooling are byte-for-byte unaffected.
        # ------------------------------------------------------------------
        self.BLEND_FRAMES: int = 8
        self._resample_active: bool = False
        self._resample_output_fps: float = 50.0
        # Fractional read cursor, in native (30 fps) frame units, into
        # ``self.frames["mujoco_qpos"]``.
        self._read_pos: float = 0.0
        # Cross-fade blend state (old buffer we are fading OUT of).
        self._blend_prev_buf: Optional[t.Tensor] = None  # [1, T, D]
        self._blend_prev_pos: float = 0.0                # native-frame cursor into old buf
        self._blend_remaining: int = 0                   # output ticks of blend left

        # Diagnostic hook. Optional callable invoked inside _predict after
        # the default has_* masks are constructed but before predict() is
        # called. Signature:
        #     fn(has_global_root_values, has_local_root_values,
        #        has_local_poses, NUM_FT) -> (has_g, has_l, has_p)
        # Use this from diagnostic scripts to simulate training-time
        # keyframe-density distributions and isolate train/infer
        # distribution mismatches. Do NOT set this in production code.
        self.diagnostic_mask_hook = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def fps(self) -> float:
        return float(self._fps)

    @property
    def current_frame_idx(self) -> int:
        return self._current_frame_idx

    @property
    def frames_remaining(self) -> int:
        """How many frames are left in the buffer after the current pop index.

        In resampled mode the read cursor is fractional (``_read_pos``); we
        report the remaining native (30 fps) frames so the replan threshold
        keeps the same "frames of lookahead" meaning across both paths.
        """
        if self.frames["mujoco_qpos"] is None:
            return 0
        T = int(self.frames["mujoco_qpos"].shape[1])
        if self._resample_active:
            return int(math.floor(T - self._read_pos))
        return T - self._current_frame_idx

    def reset(self, init_mujoco_qpos: t.Tensor) -> None:
        """Seed the ring buffer with NUM_MIN_FRAMES_IN_BUFFER copies of a stand pose.

        Args:
            init_mujoco_qpos: [qpos_dim] float tensor (e.g. [38] for X2).
                Will be tiled to fill the ring buffer.
        """
        if init_mujoco_qpos.ndim == 1:
            init_mujoco_qpos = init_mujoco_qpos[None, None, :]  # [1, 1, D]
        elif init_mujoco_qpos.ndim == 2:
            init_mujoco_qpos = init_mujoco_qpos[None, :, :]
        init_mujoco_qpos = init_mujoco_qpos.to(self._device).float()
        self._qpos_dim = int(init_mujoco_qpos.shape[-1])

        tile_n = self.NUM_MIN_FRAMES_IN_BUFFER
        if init_mujoco_qpos.shape[1] < tile_n:
            init_mujoco_qpos = t.cat(
                [init_mujoco_qpos]
                + [init_mujoco_qpos[:, -1:, :]] * (tile_n - init_mujoco_qpos.shape[1]),
                dim=1,
            )

        self.frames["mujoco_qpos"] = init_mujoco_qpos
        self.frames["model_features"] = None  # not used until first replan
        self._current_frame_idx = 0
        # Reset the resampled read cursor + drop any in-flight cross-fade.
        # ``_resample_active`` is intentionally left as-is: once the caller
        # opts into resampling it stays on across PLAYING re-entries.
        self._read_pos = 0.0
        self._blend_prev_buf = None
        self._blend_prev_pos = 0.0
        self._blend_remaining = 0

    def should_replan(self) -> bool:
        """True when the ring buffer's tail is closer than REPLAN_THRESHOLD_FRAMES.

        In resampled mode the decision is made against the FLOAT read
        position so it fires with enough lookahead given the fractional
        (``model_fps / output_fps``) consumption rate: replan when
        ``read_pos + REPLAN_THRESHOLD_FRAMES >= T``.
        """
        buf = self.frames["mujoco_qpos"]
        if buf is None:
            return False
        if self._resample_active:
            return (self._read_pos + self.REPLAN_THRESHOLD_FRAMES) >= float(buf.shape[1])
        return self.frames_remaining <= self.REPLAN_THRESHOLD_FRAMES

    def get_next_frame(self) -> t.Tensor:
        """Pop the next [qpos_dim] frame and advance the read cursor.

        Clamps at the buffer tail so we never index past the end; the daemon
        should call replan_with_velocity() before this returns a duplicate.
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError("NeuralPlannerCore.get_next_frame() called before reset()")
        buf = self.frames["mujoco_qpos"]
        idx = self._current_frame_idx
        # Clamp so the buffer's final frame repeats if we ran past replan threshold.
        idx_clamped = max(0, min(idx, buf.shape[1] - 1))
        self._current_frame_idx = min(idx + 1, buf.shape[1] - 1)
        return buf[0, idx_clamped].detach()

    def get_context_mujoco_qpos(self) -> t.Tensor:
        """Return last NUM_FRAMES_PER_TOKEN frames before the current cursor.

        Matches `full_navigation_agent.get_context_mujoco_qpos` exactly so the
        rest of the predict-and-decode pipeline transplants verbatim.
        """
        idx = self._current_frame_idx
        buf = self.frames["mujoco_qpos"]
        indices = [
            max(0, min(idx - self.NUM_FRAMES_PER_TOKEN + i + self.PRED_OFFSETS, buf.shape[1] - 1))
            for i in range(self.NUM_FRAMES_PER_TOKEN)
        ]
        return buf[:, indices, :].to(self._device)

    # ------------------------------------------------------------------
    # Output resampling (30 Hz model -> control-loop output_fps) + blend
    # ------------------------------------------------------------------

    @staticmethod
    def _slerp_wxyz(q0: t.Tensor, q1: t.Tensor, w: float) -> t.Tensor:
        """Spherical linear interpolation between two wxyz quaternions.

        Args:
            q0, q1: [4] tensors in ``[w, x, y, z]`` order.
            w: interpolation fraction in [0, 1] (0 -> q0, 1 -> q1).

        Returns a normalized [4] wxyz quaternion. Takes the shortest arc
        (flips ``q1`` when the dot product is negative) and falls back to a
        normalized lerp for nearly-parallel inputs to stay numerically safe.
        """
        q0 = q0 / (q0.norm() + 1e-8)
        q1 = q1 / (q1.norm() + 1e-8)
        dot = (q0 * q1).sum()
        if float(dot) < 0.0:
            q1 = -q1
            dot = -dot
        if float(dot) > 0.9995:
            out = q0 + w * (q1 - q0)
            return out / (out.norm() + 1e-8)
        theta0 = t.acos(dot.clamp(-1.0, 1.0))
        theta = theta0 * w
        sin0 = t.sin(theta0)
        s0 = t.sin(theta0 - theta) / sin0
        s1 = t.sin(theta) / sin0
        out = s0 * q0 + s1 * q1
        return out / (out.norm() + 1e-8)

    def _frame_at(self, buf: t.Tensor, pos: float) -> t.Tensor:
        """Interpolate a single [D] qpos frame at fractional index ``pos``.

        Layout is X2 ``mujoco_qpos`` ``[trans(3), root_quat_wxyz(4),
        dof(31)]`` (D=38): lerp indices 0:3 and 7:D, slerp the root quat
        at 3:7. Positions clamp to the valid ``[0, T-1]`` range.
        """
        T = int(buf.shape[1])
        if T <= 1:
            return buf[0, 0].detach().clone()
        pos = min(max(float(pos), 0.0), float(T - 1))
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, T - 1)
        frac = pos - i0
        f0 = buf[0, i0]
        if i1 == i0 or frac <= 0.0:
            return f0.detach().clone()
        f1 = buf[0, i1]
        out = f0.clone()
        out[:3] = f0[:3] * (1.0 - frac) + f1[:3] * frac
        out[7:] = f0[7:] * (1.0 - frac) + f1[7:] * frac
        out[3:7] = self._slerp_wxyz(f0[3:7], f1[3:7], frac)
        return out.detach()

    def _blend_frames(self, old: t.Tensor, new: t.Tensor, w_new: float) -> t.Tensor:
        """Cross-fade two [D] qpos frames: lerp trans/dof, slerp root quat."""
        w_old = 1.0 - w_new
        out = new.clone()
        out[:3] = old[:3] * w_old + new[:3] * w_new
        out[7:] = old[7:] * w_old + new[7:] * w_new
        out[3:7] = self._slerp_wxyz(old[3:7], new[3:7], w_new)
        return out.detach()

    def _resampled_output_frame(self, output_offset_ticks: float = 0.0) -> t.Tensor:
        """Return the resampled (+ optionally blended) [D] frame at a given
        offset (in OUTPUT ticks) ahead of the current read cursor.

        ``output_offset_ticks == 0`` is the frame about to be published;
        positive offsets are the future-window lookahead slots. Each output
        tick maps to ``model_fps / output_fps`` native (30 fps) frames.
        """
        buf = self.frames["mujoco_qpos"]
        step = self._fps / self._resample_output_fps
        new_pos = self._read_pos + step * output_offset_ticks
        frame = self._frame_at(buf, new_pos)
        if self._blend_prev_buf is not None:
            blend_left = self._blend_remaining - output_offset_ticks
            if blend_left > 0.0:
                old_pos = self._blend_prev_pos + step * output_offset_ticks
                old_frame = self._frame_at(self._blend_prev_buf, old_pos)
                # Linearly increasing weight on the NEW animation across the
                # 8-tick window: reaches 1.0 on the final blend tick.
                w_new = 1.0 - (blend_left - 1.0) / float(self.BLEND_FRAMES)
                w_new = min(max(w_new, 0.0), 1.0)
                frame = self._blend_frames(old_frame, frame, w_new)
        return frame

    def get_next_frame_resampled(
        self, output_fps: Optional[float] = None
    ) -> t.Tensor:
        """Pop one output frame at ``output_fps`` and advance the float cursor.

        This is the resampling counterpart to ``get_next_frame``. It advances
        a fractional read cursor by ``model_fps / output_fps`` native frames
        per call and interpolates between the two nearest 30 Hz frames
        (lerp on translation + dof, slerp on the root quaternion). When a
        replan installed a new buffer, the first ``BLEND_FRAMES`` outputs are
        cross-faded from the previous buffer for a seamless handoff.
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError(
                "NeuralPlannerCore.get_next_frame_resampled() called before reset()"
            )
        if output_fps is not None:
            self._resample_output_fps = float(output_fps)
        self._resample_active = True

        frame = self._resampled_output_frame(0.0)

        step = self._fps / self._resample_output_fps
        self._read_pos += step
        # Keep the integer cursor in sync so get_context_mujoco_qpos() (read
        # at replan time) samples the seam around the current read position.
        self._current_frame_idx = int(math.floor(self._read_pos))
        if self._blend_prev_buf is not None:
            self._blend_prev_pos += step
            self._blend_remaining -= 1
            if self._blend_remaining <= 0:
                self._blend_prev_buf = None
                self._blend_remaining = 0
        return frame

    def peek_output_frame(self, output_offset_ticks: float) -> t.Tensor:
        """Peek a resampled (+ blended) future frame WITHOUT advancing.

        ``output_offset_ticks`` is measured in OUTPUT ticks ahead of the
        frame that ``get_next_frame_resampled`` will next return. With the
        publish loop's ``step_ticks = 5`` at 50 Hz this yields +0.1 s real
        spacing (0.6 native frames/tick * 5 = 3 native frames = 0.1 s).
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError(
                "NeuralPlannerCore.peek_output_frame() called before reset()"
            )
        self._resample_active = True
        return self._resampled_output_frame(float(output_offset_ticks))

    def _arm_output_blend(self) -> None:
        """Snapshot the current buffer + read cursor to start an 8-tick blend.

        Called from the replan paths right before the buffer is swapped, but
        only while resampling is active. The old buffer keeps playing forward
        (from ``_blend_prev_pos``) and is faded out over ``BLEND_FRAMES``
        output ticks while the new buffer fades in from its frame 0.
        """
        if not self._resample_active:
            return
        if self.frames["mujoco_qpos"] is None:
            return
        self._blend_prev_buf = self.frames["mujoco_qpos"]
        self._blend_prev_pos = float(self._read_pos)
        self._blend_remaining = self.BLEND_FRAMES

    # ------------------------------------------------------------------
    # Velocity-intent replan
    # ------------------------------------------------------------------

    @t.no_grad()
    def replan_with_velocity(
        self,
        velocity_intent: t.Tensor,
        num_tokens: Optional[int] = None,
    ) -> tuple[t.Tensor, t.Tensor, int]:
        """Replan the upcoming motion conditioned on a 4-D velocity vector.

        Args:
            velocity_intent: [4] or [B, 4] tensor with channels
                ``[yaw_rate_rad_s, vel_x_m_s, vel_z_m_s, hip_height_m]``,
                broadcast across all 4 target frames in the constraint
                window.

                **Channel semantics** (verified against the X2 + G1
                mujoco converter via
                ``scripts/probe_root_constraint_modes.py``):

                * Y is the gravity axis in the local-motion-rep coordinate
                  frame.
                * ``vel_x`` = motion-rep X axis = MuJoCo Y =
                  **lateral** body-frame velocity (positive = robot's
                  left after canonicalization).
                * ``vel_z`` = motion-rep Z axis = MuJoCo X =
                  **forward** body-frame velocity (positive = robot's
                  forward after canonicalization).

                Earlier docs (incorrectly) called vel_z lateral; the
                converter rotation ``mujoco_to_motion`` maps mujoco-X
                (mjcf forward) -> motion-Z, so velocity_intent[2] is
                what walks the robot forward.
            num_tokens: Override the model's MASKED_NUM_TOKENS sentinel. Default
                None lets the root model predict the horizon length.

        Returns:
            model_features [1, T, D], mujoco_qpos [1, T, qpos_dim], num_pred_frames.
            The internal frame buffer is updated in place; callers can ignore
            the returned tensors and just keep popping via get_next_frame().
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError("NeuralPlannerCore.replan_with_velocity() called before reset()")

        velocity_intent = self._coerce_velocity_intent(velocity_intent)

        # Pull the last 4 frames of MuJoCo qpos and build the input dict the
        # legacy `_generate_inbetween_frames` consumes.
        context_mujoco_qpos = self.get_context_mujoco_qpos()
        input_state: dict = {
            "context_mujoco_qpos": context_mujoco_qpos,
            "raw_context_mujoco_qpos": context_mujoco_qpos.clone(),
            "velocity_intent": velocity_intent,
        }

        if self.FORCE_CANONICALIZATION:
            self._canonicalize_mujoco_qpos(input_state)

        # Convert canonicalized qpos to (joint_positions, joint_rotations).
        (
            input_state["context_global_joint_positions"],
            input_state["context_global_joint_rotations"],
        ) = self._converter.convert_mujoco_qpos_to_motion_transforms(
            input_state["context_mujoco_qpos"]
        )

        model_features, mujoco_qpos, num_pred_frames = self._predict_with_velocity(
            input_state, num_tokens=num_tokens
        )

        # Arm the cross-fade BEFORE swapping the buffer (snapshots the
        # still-playing old buffer + read cursor). No-op unless resampling
        # is active, so the legacy integer / offline paths are unaffected.
        self._arm_output_blend()
        self.frames["model_features"] = model_features[:, : int(num_pred_frames), :]
        self.frames["mujoco_qpos"] = mujoco_qpos[:, : int(num_pred_frames), :]
        self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS
        # New buffer read starts at frame 0 (the FILTER_QPOS-blended context
        # seam), mirroring the integer cursor reset above.
        if self._resample_active:
            self._read_pos = float(self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS)

        return model_features, mujoco_qpos, int(num_pred_frames)

    # ------------------------------------------------------------------
    # Pose-template-intent replan (SONIC / MotionBricks paper design)
    # ------------------------------------------------------------------

    @t.no_grad()
    def replan_with_pose_template(
        self,
        velocity_intent: t.Tensor,
        mode_idx: int,
        num_tokens: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[t.Tensor, t.Tensor, int]:
        """Replan with a 4-D velocity vector AND a discrete pose-template mode.

        This is the paper's "navigation keyframes from clip library" path.
        Unlike ``replan_with_velocity`` (which masks ``has_local_poses[:, -4:]``
        off and asks the in-betweener to free-sample target poses), this
        method samples a 4-frame segment from the mode's reference clip,
        realigns it to the implied target root xz / heading, and feeds it
        as the target keyframe to the in-betweener.

        Args:
            velocity_intent: same 4-tuple as ``replan_with_velocity``:
                ``[yaw_rate_rad_s, vel_x_m_s, vel_z_m_s, hip_height_m]``.
                Drives the spring-style target-root integration over
                ``TARGET_HORIZON_S``; the model still sees the broadcast
                velocity in ``target_local_root_values``.
            mode_idx: integer index into the loaded clip library's mode
                table. The bake script's ``DEFAULT_MODES`` defines the
                binding (e.g. 0=idle, 1=slow_walk, 2=walk, 3=run_proxy);
                see ``out/X2-clip.modes.json``.
            num_tokens: override the model's MASKED_NUM_TOKENS sentinel.
            random_seed: int controlling which 4-frame window of the
                mode's clip is sampled. ``None`` -> randomly drawn at
                call time (matches G1 demo). Pass a fixed value for
                deterministic replays (test fixtures etc.).

        Returns:
            ``(model_features [1, T, D], mujoco_qpos [1, T, qpos_dim],
            num_pred_frames)``. Same shape contract as ``replan_with_velocity``.
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError(
                "NeuralPlannerCore.replan_with_pose_template() called before reset()"
            )
        if self._clip_library is None:
            raise RuntimeError(
                "NeuralPlannerCore was constructed without a clip_library_ckpt; "
                "pose-template inference is unavailable. Pass clip_library_ckpt "
                "to the constructor (e.g. via load_x2_planner) or use "
                "replan_with_velocity()."
            )

        velocity_intent = self._coerce_velocity_intent(velocity_intent)
        context_mujoco_qpos = self.get_context_mujoco_qpos()
        input_state: dict = {
            "context_mujoco_qpos": context_mujoco_qpos,
            "raw_context_mujoco_qpos": context_mujoco_qpos.clone(),
            "velocity_intent": velocity_intent,
        }

        if self.FORCE_CANONICALIZATION:
            self._canonicalize_mujoco_qpos(input_state)

        (
            input_state["context_global_joint_positions"],
            input_state["context_global_joint_rotations"],
        ) = self._converter.convert_mujoco_qpos_to_motion_transforms(
            input_state["context_mujoco_qpos"]
        )

        model_features, mujoco_qpos, num_pred_frames = self._predict_with_pose_template(
            input_state,
            mode_idx=int(mode_idx),
            num_tokens=num_tokens,
            random_seed=random_seed,
        )

        # Arm the cross-fade BEFORE swapping the buffer (snapshots the
        # still-playing old buffer + read cursor). No-op unless resampling
        # is active, so the legacy integer / offline paths are unaffected.
        self._arm_output_blend()
        self.frames["model_features"] = model_features[:, : int(num_pred_frames), :]
        self.frames["mujoco_qpos"] = mujoco_qpos[:, : int(num_pred_frames), :]
        self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS
        # New buffer read starts at frame 0 (the FILTER_QPOS-blended context
        # seam), mirroring the integer cursor reset above.
        if self._resample_active:
            self._read_pos = float(self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS)

        return model_features, mujoco_qpos, int(num_pred_frames)

    # ------------------------------------------------------------------
    # Pose-to-pose in-betweening replan (last-mile precision docking)
    # ------------------------------------------------------------------

    # num_tokens sizing for replan_with_target_pose: ~90 frames per meter
    # of translation (caps the reference speed at ~0.33 m/s) plus ~24
    # frames per 90 deg of heading change (~15 frames/rad), clamped to
    # the model's trained token range [6, 16] (= 24..64 frames @ 30 fps).
    # SPEED CALIBRATION (MuJoCo+SONIC, 2026-08-18): the deployed SONIC
    # under-tracks brisk in-betweened references — 0.70 m in 36 frames
    # (0.58 m/s) executed 32 cm short, while the SAME target in 64
    # frames (0.33 m/s) executed to 3.0 cm. The original 48 f/m sizing
    # produced the brisk regime; 90 f/m keeps single calls in the
    # well-tracked band (one call covers <= ~0.7 m; longer docks chain).
    TARGET_POSE_FRAMES_PER_METER: float = 90.0
    TARGET_POSE_FRAMES_PER_RAD: float = 15.0
    TARGET_POSE_MIN_TOKENS: int = 6
    TARGET_POSE_MAX_TOKENS: int = 16
    # Calibrated arrival-yaw aim-past. The frozen X2 core delivers the
    # standing arrival ~7.5 deg clockwise of the conditioned heading
    # (constant across commanded yaw / translation direction; verified NOT
    # to be a template hip-twist — hip-vector heading == root yaw on the
    # idle template — so it is a decode-side artifact of the frozen
    # core). Aiming past by +7.5 deg cancels it to <=1.2 deg (measured
    # MuJoCo-kinematic, 2026-08-18, micro suite). Applied internally to
    # the template rotation + heading conditioning only; ``target_se2``
    # remains the true commanded arrival.
    TARGET_POSE_YAW_COMP_RAD: float = math.radians(7.5)

    @t.no_grad()
    def replan_with_target_pose(
        self,
        target_se2,
        target_pose_source: str = "standing",
        num_tokens: Optional[int] = None,
        random_seed: Optional[int] = None,
        target_hip_m: Optional[float] = None,
        ending_target_cond: bool = True,
        yaw_comp_rad: Optional[float] = None,
        target_segment_override: Optional[dict] = None,
    ) -> tuple[t.Tensor, t.Tensor, int]:
        """Replan a pose-to-pose transition that ENDS at a commanded SE(2).

        This is the third replan variant (after ``replan_with_velocity`` and
        ``replan_with_pose_template``): true in-betweening. The context is
        the current 4 frames as usual; the target 4 frames are a standing
        keyframe segment from the clip library rigidly placed at the
        commanded world SE(2) (position + heading, height = standing hip).
        Both ``has_local_poses`` and ``has_global_root_values`` are True for
        the target window, so the frozen core generates a stepping
        transition that arrives AND stands at the target. Designed as a
        LAST-MILE precision docking / micro-step primitive; for moves
        > ~1.5 m chain calls (pop the buffer, call again toward the same
        target).

        Args:
            target_se2: ``(x, y, yaw_rad)`` in the MuJoCo WORLD frame (the
                same frame as the qpos ring buffer / ``reset`` seed).
            target_pose_source: which clip-library mode supplies the target
                keyframe. ``"standing"`` (default) uses mode 0 (idle) of
                the loaded library; an integer-valued string or int selects
                an explicit mode index.
            num_tokens: transition length override in tokens (4 frames
                each). Default None sizes it from the SE(2) distance (see
                ``TARGET_POSE_FRAMES_PER_METER`` / ``_PER_RAD``), clamped
                to [6, 16] -> 24..64 frames @ 30 fps.
            random_seed: deterministic 4-frame window selection within the
                source clip (None -> random, matches the sibling paths).
            target_hip_m: world-frame hip (pelvis) height at the target.
                Default None uses the current context's last-frame hip
                height (start standing -> end standing at the same height).
            ending_target_cond: keep the ending target-pose conditioning in
                the VQVAE decoder (config ``skip_ending_target_cond`` is
                the INVERSE of this). Unlike the velocity paths (which skip
                it because their ending target is imprecise), the whole
                point here is a precise ending, so default True.
            yaw_comp_rad: arrival-yaw aim-past compensation. Default None
                uses the calibrated ``TARGET_POSE_YAW_COMP_RAD``; pass 0.0
                to disable (raw conditioning).
            target_segment_override: supply the 4-frame target keyframe
                segment directly instead of sampling it from the clip
                library (same dict shape ``sample_target_segment``
                returns: ``root_positions``/``joint_positions``/
                ``joint_rotations``/``headings``). When given, the clip
                library is not consulted at all and ``target_pose_source``
                / ``random_seed`` no longer select the keyframe. Build one
                from an arbitrary MuJoCo qpos with
                ``_target_segment_from_mujoco_qpos`` — or just call the
                sibling ``replan_with_target_keyframe``.

        Returns:
            ``(model_features [1, T, D], mujoco_qpos [1, T, qpos_dim],
            num_pred_frames)`` — same contract as the sibling replans; the
            internal frame buffer is swapped in place.
        """
        if self.frames["mujoco_qpos"] is None:
            raise RuntimeError(
                "NeuralPlannerCore.replan_with_target_pose() called before reset()"
            )
        if self._clip_library is None and target_segment_override is None:
            raise RuntimeError(
                "NeuralPlannerCore was constructed without a clip_library_ckpt; "
                "target-pose inference needs the standing keyframe library. "
                "Pass clip_library_ckpt to the constructor (e.g. via "
                "load_x2_planner) — or supply target_segment_override."
            )
        if not self.FORCE_CANONICALIZATION:
            raise RuntimeError(
                "replan_with_target_pose requires FORCE_CANONICALIZATION "
                "(the world->canonical SE(2) transform reuses the "
                "canonicalization anchors)."
            )

        if target_pose_source == "standing":
            mode_idx = 0
        else:
            mode_idx = int(target_pose_source)

        target_se2_t = t.as_tensor(target_se2, dtype=t.float32).reshape(3)
        tx, ty, tyaw = (float(v) for v in target_se2_t)

        context_mujoco_qpos = self.get_context_mujoco_qpos()
        input_state: dict = {
            "context_mujoco_qpos": context_mujoco_qpos,
            "raw_context_mujoco_qpos": context_mujoco_qpos.clone(),
        }
        self._canonicalize_mujoco_qpos(input_state)
        (
            input_state["context_global_joint_positions"],
            input_state["context_global_joint_rotations"],
        ) = self._converter.convert_mujoco_qpos_to_motion_transforms(
            input_state["context_mujoco_qpos"]
        )

        # ----- world SE(2) -> canonicalized-context frame -------------
        # Canonicalization put context frame 0 at the XY origin with zero
        # heading; map the world target into that frame, then into the
        # motion rep's axes (motion X = MuJoCo Y, motion Z = MuJoCo X,
        # motion Y = MuJoCo Z (up); heading angle carries over 1:1).
        p0 = input_state["first_frame_position"][0]          # [3] (z zeroed)
        th0 = float(input_state["first_frame_heading_angle"][0])
        dx, dy = tx - float(p0[0]), ty - float(p0[1])
        c0, s0 = math.cos(th0), math.sin(th0)
        x_c = c0 * dx + s0 * dy    # R_z(-th0) @ [dx, dy]
        y_c = -s0 * dx + c0 * dy
        if yaw_comp_rad is None:
            yaw_comp_rad = self.TARGET_POSE_YAW_COMP_RAD
        yaw_c = float(
            (tyaw + yaw_comp_rad - th0 + math.pi) % (2.0 * math.pi) - math.pi
        )
        target_x_mo, target_z_mo, target_heading_mo = y_c, x_c, yaw_c

        # Hip height: current context last-frame hip (world Z == motion Y).
        raw_ctx = input_state["raw_context_mujoco_qpos"]
        if target_hip_m is None:
            target_hip = float(raw_ctx[0, -1, 2])
        else:
            target_hip = float(target_hip_m)

        # ----- token budget from the CURRENT pose -> target SE(2) -----
        if num_tokens is None:
            cur_x, cur_y = float(raw_ctx[0, -1, 0]), float(raw_ctx[0, -1, 1])
            qw, qx, qy, qz = (float(v) for v in raw_ctx[0, -1, 3:7])
            cur_yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            dist = math.hypot(tx - cur_x, ty - cur_y)
            dyaw = abs((tyaw - cur_yaw + math.pi) % (2.0 * math.pi) - math.pi)
            frames = (
                self.TARGET_POSE_FRAMES_PER_METER * dist
                + self.TARGET_POSE_FRAMES_PER_RAD * dyaw
            )
            num_tokens = int(math.ceil(frames / self.NUM_FRAMES_PER_TOKEN))
            num_tokens = max(
                self.TARGET_POSE_MIN_TOKENS,
                min(self.TARGET_POSE_MAX_TOKENS, num_tokens),
            )

        model_features, mujoco_qpos, num_pred_frames = self._predict_with_target_pose(
            input_state,
            target_x_mo=target_x_mo,
            target_z_mo=target_z_mo,
            target_heading_mo=target_heading_mo,
            target_hip=target_hip,
            mode_idx=mode_idx,
            num_tokens=int(num_tokens),
            random_seed=random_seed,
            ending_target_cond=ending_target_cond,
            target_segment=target_segment_override,
        )

        self._arm_output_blend()
        self.frames["model_features"] = model_features[:, : int(num_pred_frames), :]
        self.frames["mujoco_qpos"] = mujoco_qpos[:, : int(num_pred_frames), :]
        self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS
        if self._resample_active:
            self._read_pos = float(self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS)

        return model_features, mujoco_qpos, int(num_pred_frames)

    # ------------------------------------------------------------------
    # Sibling entry: arbitrary target KEYFRAME (not a clip-library mode).
    # ------------------------------------------------------------------

    @t.no_grad()
    def _target_segment_from_mujoco_qpos(self, keyframe_qpos) -> dict:
        """Bake a MuJoCo qpos keyframe into a target segment dict.

        ``keyframe_qpos``: ``[qpos_dim]`` (a single frame, repeated to fill
        the 4-frame target window -> a STATIC keyframe) or
        ``[NUM_FRAMES_PER_TOKEN, qpos_dim]`` (an explicit 4-frame window).

        The transform mirrors the offline clip bake
        (``build_x2_planner_clips._compute_clip_buffers``) exactly, so a
        segment produced here is interchangeable with one from
        ``_X2ClipLibrary.sample_target_segment``: motion-rep transforms via
        the converter, root Y (height) zeroed in ``root_positions``, joint
        positions made root-XZ-relative, heading read off the root's
        forward (+Z) axis projected into XZ.
        """
        NUM_FT = self.NUM_FRAMES_PER_TOKEN
        q = t.as_tensor(keyframe_qpos, dtype=t.float32, device=self._device)
        if q.ndim == 1:
            q = q[None].repeat(NUM_FT, 1)
        if q.ndim != 2 or q.shape[0] != NUM_FT:
            raise ValueError(
                f"keyframe_qpos must be [qpos_dim] or [{NUM_FT}, qpos_dim], "
                f"got {tuple(q.shape)}"
            )
        gjp, gjr = self._converter.convert_mujoco_qpos_to_motion_transforms(
            q[None]
        )
        gjp, gjr = gjp[0], gjr[0]                    # [4, J, 3], [4, J, 3, 3]
        root = gjp[:, 0, :].clone()
        root[:, 1] = 0.0                            # motion Y == world up
        fwd = t.matmul(
            gjr[:, 0],
            t.tensor([0.0, 0.0, 1.0], device=self._device).view(1, 3, 1),
        ).view(-1, 3)
        fwd_xz = fwd * t.tensor([1.0, 0.0, 1.0], device=self._device).view(1, 3)
        fwd_xz = fwd_xz / fwd_xz.norm(dim=1, keepdim=True)
        return {
            "root_positions": root,                       # [4, 3]
            "joint_positions": gjp - root[:, None, :],    # [4, J, 3]
            "joint_rotations": gjr,                       # [4, J, 3, 3]
            "headings": t.atan2(fwd_xz[:, 0], fwd_xz[:, 2]),  # [4]
        }

    @t.no_grad()
    def replan_with_target_keyframe(
        self,
        target_se2,
        target_keyframe_qpos,
        num_tokens: Optional[int] = None,
        random_seed: Optional[int] = None,
        target_hip_m: Optional[float] = None,
        ending_target_cond: bool = True,
        yaw_comp_rad: Optional[float] = 0.0,
    ) -> tuple[t.Tensor, t.Tensor, int]:
        """In-between to an ARBITRARY full-body keyframe at a commanded SE(2).

        The fourth replan variant, and the only one that takes a POSTURE
        rather than a clip-library mode (``replan_with_pose_template``) or a
        velocity (``replan_with_velocity``). ``replan_with_target_pose``
        pins the target window to a standing keyframe sampled from the clip
        library; this pins it to a caller-supplied one, so the arrival
        posture — arms, waist lean, anything the converter round-trips —
        becomes part of the CONDITIONING rather than something spliced over
        the planner's output afterwards.

        The frozen core takes its constraints per FRAME, not per joint:
        ``has_local_poses`` is all-or-nothing for the target window, so the
        keyframe must be a COMPLETE body pose. Compose it upstream (e.g.
        standing legs/pelvis from the idle template, waist + arms from an
        IK solve) and pass the whole qpos.

        Args:
            target_se2: ``(x, y, yaw_rad)`` in the MuJoCo WORLD frame. Pass
                the CURRENT SE(2) for a stationary re-posture (zero
                displacement, arms/waist move, base holds).
            target_keyframe_qpos: ``[qpos_dim]`` or
                ``[NUM_FRAMES_PER_TOKEN, qpos_dim]`` MuJoCo qpos. The
                keyframe's own root XY/heading are IGNORED — the segment is
                rigidly re-placed at ``target_se2`` exactly as the library
                keyframe is — but its joint angles and root height (via
                ``target_hip_m``) are honoured.
            num_tokens: transition length in tokens (4 frames each). The
                distance-based default sizing is meaningless for a
                zero-displacement re-posture (it floors at
                ``TARGET_POSE_MIN_TOKENS``), so pass this explicitly to
                control the pace; the trained range is 6..16 tokens
                (24..64 frames @ 30 fps) and slower has consistently
                tracked tighter through the deployed SONIC.
            random_seed: unused for keyframe selection (the keyframe is
                supplied); forwarded for signature parity with the sibling.
            target_hip_m: world-frame pelvis height at the target. Default
                None keeps the current context's hip height.
            ending_target_cond: as in ``replan_with_target_pose``.
            yaw_comp_rad: arrival-yaw aim-past. Defaults to **0.0** here,
                not to ``TARGET_POSE_YAW_COMP_RAD``: the +7.5 deg constant
                was calibrated on the IDLE template and the decode yaw bias
                is a property of the target STANCE (composed non-idle
                stances measured ~zero, KEYFRAME_GRAMMAR.md). Calibrate per
                keyframe stance if precision yaw matters; pass None to fall
                back to the idle-template constant.

        Returns:
            ``(model_features, mujoco_qpos, num_pred_frames)`` — same
            contract as the sibling replans.
        """
        segment = self._target_segment_from_mujoco_qpos(target_keyframe_qpos)
        return self.replan_with_target_pose(
            target_se2,
            num_tokens=num_tokens,
            random_seed=random_seed,
            target_hip_m=target_hip_m,
            ending_target_cond=ending_target_cond,
            yaw_comp_rad=yaw_comp_rad,
            target_segment_override=segment,
        )

    # ------------------------------------------------------------------
    # Internal: target-pose prediction path. Mirrors
    # `_predict_with_pose_template` (kept side-by-side per the same
    # rationale). Differences are confined to the target window:
    #   - target_global_root_values = the commanded SE(2) (static; the
    #     standing segment's tiny root drift rigidly placed at the target,
    #     LAST target frame pinned exactly on the command).
    #   - target_local_root_values = zero velocity + target hip height
    #     (arrive and STAND).
    #   - target_local_poses = the standing segment rotated to the
    #     commanded heading; has_local_poses[:, -4:] stays True.
    #   - num_tokens is always explicit (sized from distance upstream).
    #   - skip_ending_target_cond defaults OFF (precise ending wanted).
    # ------------------------------------------------------------------

    @t.no_grad()
    def _predict_with_target_pose(
        self,
        input_state: dict,
        target_x_mo: float,
        target_z_mo: float,
        target_heading_mo: float,
        target_hip: float,
        mode_idx: int,
        num_tokens: int,
        random_seed: Optional[int] = None,
        ending_target_cond: bool = True,
        target_segment: Optional[dict] = None,
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        assert self._clip_library is not None or target_segment is not None
        batch_size = 1
        fps = self._inferencer.local_motion_rep.fps
        root_joint_idx = 0
        device = self._device
        NUM_FT = self.NUM_FRAMES_PER_TOKEN

        ctx_joint_pos = input_state["context_global_joint_positions"]   # [B, 4, J, 3]
        ctx_joint_rot = input_state["context_global_joint_rotations"]   # [B, 4, J, 3, 3]

        # ----------------------------------------------------------------
        # Past 4 frames — identical to _predict_with_pose_template.
        # ----------------------------------------------------------------
        context_global_root_pos = ctx_joint_pos[:, :, root_joint_idx, :]  # [B, 4, 3]
        context_rotation_angle = t.atan2(
            ctx_joint_rot[:, :, root_joint_idx, 0, 2],
            ctx_joint_rot[:, :, root_joint_idx, 2, 2],
        )  # [B, 4]
        context_global_root_values = t.cat(
            [
                context_global_root_pos,
                t.cos(context_rotation_angle)[..., None],
                t.sin(context_rotation_angle)[..., None],
            ],
            dim=-1,
        )  # [B, 4, 5]

        context_local_root_values = t.zeros([batch_size, NUM_FT, 4], device=device)
        context_local_root_values[:, : NUM_FT - 1, 0] = (
            ((context_rotation_angle[:, 1:] - context_rotation_angle[:, :-1] + t.pi) % (2 * t.pi))
            - t.pi
        ) * fps
        context_local_root_values[:, : NUM_FT - 1, 1:3] = (
            context_global_root_pos[:, 1:, [0, 2]] - context_global_root_pos[:, :-1, [0, 2]]
        ) * fps
        context_local_root_values[:, : NUM_FT - 1, 3] = (
            context_global_root_values[:, : NUM_FT - 1, 1]
        )
        context_local_root_values[:, NUM_FT - 1, :] = context_local_root_values[:, NUM_FT - 2, :]

        joint_positions_ctx = ctx_joint_pos[:, :, 1:, :].clone()
        joint_positions_ctx[..., 0] = ctx_joint_pos[:, :, 1:, 0] - ctx_joint_pos[:, :, :1, 0]
        joint_positions_ctx[..., 2] = ctx_joint_pos[:, :, 1:, 2] - ctx_joint_pos[:, :, :1, 2]
        joint_rotation_ortho6d_ctx = matrix_to_cont6d(ctx_joint_rot)
        context_local_poses = t.cat(
            [
                joint_positions_ctx.view([batch_size, NUM_FT, -1]),
                joint_rotation_ortho6d_ctx.view([batch_size, NUM_FT, -1]),
            ],
            dim=-1,
        )

        # ----------------------------------------------------------------
        # Target 4 frames: standing segment rigidly placed at the command.
        # The LAST target frame is pinned exactly on the commanded SE(2);
        # earlier target frames carry the segment's own (tiny, for idle)
        # root drift, rotated into the commanded heading.
        #
        # ``target_segment`` (supplied by replan_with_target_keyframe) is an
        # arbitrary baked keyframe in the SAME shape the library returns;
        # everything downstream is identical either way.
        # ----------------------------------------------------------------
        if target_segment is None:
            segment = self._clip_library.sample_target_segment(
                mode_idx=mode_idx, random_seed=random_seed
            )
        else:
            segment = target_segment
        clip_joint_pos = segment["joint_positions"][None].to(device)   # [1, 4, J, 3]
        clip_joint_rot = segment["joint_rotations"][None].to(device)   # [1, 4, J, 3, 3]
        clip_headings = segment["headings"][None].to(device)           # [1, 4]
        clip_root_pos = segment["root_positions"][None].to(device)     # [1, 4, 3] (Y=0)

        target_heading_t = t.tensor([target_heading_mo], device=device, dtype=t.float32)
        # Align the segment's LAST frame heading with the command.
        diff_heading_single = (
            (target_heading_t - clip_headings[:, -1] + t.pi) % (2 * t.pi) - t.pi
        )  # [B]
        corrective_mat = angle_to_Y_rotation_matrix(diff_heading_single).to(device).float()  # [B, 3, 3]
        corrective_mat_jnt = corrective_mat[:, None, None, :, :]

        aligned_joint_rot = t.matmul(corrective_mat_jnt, clip_joint_rot)
        aligned_joint_pos = t.matmul(
            corrective_mat_jnt, clip_joint_pos[..., None]
        )[..., 0]

        # Root path: rotate the segment's root drift (relative to its last
        # frame) and translate the last frame onto the commanded target.
        root_offsets = clip_root_pos - clip_root_pos[:, -1:, :]        # [1, 4, 3]
        root_offsets = t.matmul(
            corrective_mat[:, None, :, :], root_offsets[..., None]
        )[..., 0]
        target_root_pos = root_offsets + t.tensor(
            [[[target_x_mo, 0.0, target_z_mo]]], device=device, dtype=t.float32
        )
        target_root_pos[..., 1] = target_hip

        target_headings_4 = clip_headings + diff_heading_single[:, None]  # [B, 4]
        target_global_root_values = t.cat(
            [
                target_root_pos,
                t.cos(target_headings_4)[..., None],
                t.sin(target_headings_4)[..., None],
            ],
            dim=-1,
        )  # [B, 4, 5]

        # Arrive and STAND: zero velocity / yaw-rate, hip at target height.
        target_local_root_values = t.zeros([batch_size, NUM_FT, 4], device=device)
        target_local_root_values[..., 3] = target_hip

        target_joint_positions = aligned_joint_pos[:, :, 1:, :]
        target_joint_rotation_ortho6d = matrix_to_cont6d(aligned_joint_rot)
        target_local_poses = t.cat(
            [
                target_joint_positions.reshape([batch_size, NUM_FT, -1]),
                target_joint_rotation_ortho6d.reshape([batch_size, NUM_FT, -1]),
            ],
            dim=-1,
        )

        # ----------------------------------------------------------------
        # Concatenate -> 8-frame constraint window. Both masks TRUE on the
        # target window (position/heading AND pose fully constrained).
        # ----------------------------------------------------------------
        local_root_values = t.cat([context_local_root_values, target_local_root_values], dim=1)
        global_root_values = t.cat([context_global_root_values, target_global_root_values], dim=1)
        local_poses = t.cat([context_local_poses, target_local_poses], dim=1)

        has_global_root_values = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
        has_local_root_values = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
        has_local_poses = t.ones_like(local_poses[:, :, 0], dtype=t.bool)
        has_local_root_values[:, NUM_FT - 1] = False

        if self.diagnostic_mask_hook is not None:
            has_global_root_values, has_local_root_values, has_local_poses = (
                self.diagnostic_mask_hook(
                    has_global_root_values,
                    has_local_root_values,
                    has_local_poses,
                    NUM_FT,
                )
            )

        num_tokens_t = t.full([batch_size, 1], int(num_tokens), dtype=t.int, device=device)

        config = {
            "num_inference_step": 1,
            "smooth_root_traj": False,
            "allow_pred_out_of_reach_num_tokens": False,
            "pose_token_sampling_use_argmax": True,
            "skip_ending_target_cond": not ending_target_cond,
        }
        info: dict = {}
        pred_global_motions, num_pred_tokens = self._inferencer.predict(
            global_root_values,
            has_global_root_values,
            local_root_values,
            has_local_root_values,
            local_poses,
            has_local_poses,
            num_tokens_t,
            config=config,
            info=info,
        )

        model_features = pred_global_motions
        num_pred_frames = NUM_FT * num_pred_tokens

        mujoco_qpos = self._converter.convert_motion_features_to_mujoco_qpos(
            model_features, self._motion_rep, False
        )
        root_rot = mujoco_qpos[:, :, 3:7].clone()
        mujoco_qpos[:, :, 3:7] = root_rot[:, :, [3, 0, 1, 2]]

        if self.FORCE_CANONICALIZATION:
            input_state["mujoco_qpos"] = mujoco_qpos
            mujoco_qpos = self._uncanonicalize_mujoco_qpos(input_state)

        if self.FILTER_QPOS:
            self.frames["raw_mujoco_qpos"] = mujoco_qpos.clone()
            ctx = input_state["raw_context_mujoco_qpos"]
            num_ctx = ctx.shape[1]
            blend = t.linspace(0.3, 0.7, num_ctx)[None, :, None].to(ctx.device)
            mujoco_qpos[:, :num_ctx, :3] = (
                ctx[:, :, :3] * (1 - blend) + mujoco_qpos[:, :num_ctx, :3] * blend
            )
            mujoco_qpos[:, :num_ctx, 7:] = (
                ctx[:, :, 7:] * (1 - blend) + mujoco_qpos[:, :num_ctx, 7:] * blend
            )

        return model_features, mujoco_qpos, num_pred_frames

    # ------------------------------------------------------------------
    # Internal: predict + decode (forked from full_navigation_agent
    # ._generate_inbetween_frames). The differences from G1 demo:
    #   - target_local_root_values is constructed from velocity_intent, not
    #     from a target keyframe pose (no clip_holder).
    #   - target_local_poses is zero + masked off.
    #   - target_global_root_values is zero + masked off.
    # ------------------------------------------------------------------

    @t.no_grad()
    def _predict_with_velocity(
        self, input_state: dict, num_tokens: Optional[int] = None
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        # PoC G2: derive batch from input instead of hardcoding 1 — the
        # tensor plumbing below is already [B, ...]-shaped throughout.
        batch_size = int(input_state["velocity_intent"].shape[0]) \
            if "velocity_intent" in input_state else 1
        MASKED_NUM_TOKENS = self._inferencer._root_model.backbone_net.MASKED_NUM_TOKENS
        fps = self._inferencer.local_motion_rep.fps
        root_joint_idx = 0
        device = self._device
        NUM_FT = self.NUM_FRAMES_PER_TOKEN

        ctx_joint_pos = input_state["context_global_joint_positions"]   # [B, 4, J, 3]
        ctx_joint_rot = input_state["context_global_joint_rotations"]   # [B, 4, J, 3, 3]
        velocity_intent = input_state["velocity_intent"]                # [B, 4]

        # ----------------------------------------------------------------
        # Past 4 frames: global + local root values from context positions.
        # Layout matches LocalRootLocalBody.compute_root_rep_from_root_pos_and_rot:
        #   channel 0 = yaw_rate (rad/s, * fps internally)
        #   channels 1-2 = vel_x, vel_z (m/s, * fps internally)
        #   channel 3 = hip height (m, world-frame Y-up)
        # ----------------------------------------------------------------
        context_global_root_pos = ctx_joint_pos[:, :, root_joint_idx, :]  # [B, 4, 3]
        context_rotation_angle = t.atan2(
            ctx_joint_rot[:, :, root_joint_idx, 0, 2],
            ctx_joint_rot[:, :, root_joint_idx, 2, 2],
        )  # [B, 4] heading angle
        context_global_root_values = t.cat(
            [
                context_global_root_pos,
                t.cos(context_rotation_angle)[..., None],
                t.sin(context_rotation_angle)[..., None],
            ],
            dim=-1,
        )  # [B, 4, 5]

        context_local_root_values = t.zeros([batch_size, NUM_FT, 4], device=device)
        # yaw-rate (channel 0) from differenced headings
        context_local_root_values[:, : NUM_FT - 1, 0] = (
            ((context_rotation_angle[:, 1:] - context_rotation_angle[:, :-1] + t.pi) % (2 * t.pi))
            - t.pi
        ) * fps
        # vel_x, vel_z (channels 1-2) from differenced xz positions in body frame
        context_local_root_values[:, : NUM_FT - 1, 1:3] = (
            context_global_root_pos[:, 1:, [0, 2]] - context_global_root_pos[:, :-1, [0, 2]]
        ) * fps
        # hip height (channel 3) directly from Y
        context_local_root_values[:, : NUM_FT - 1, 3] = (
            context_global_root_values[:, : NUM_FT - 1, 1]
        )
        # Last context frame has no t+1; copy the previous frame's velocity.
        context_local_root_values[:, NUM_FT - 1, :] = context_local_root_values[:, NUM_FT - 2, :]

        # Joint positions relative to the root, plus 6D rotations.
        joint_positions = ctx_joint_pos[:, :, 1:, :].clone()
        joint_positions[..., 0] = ctx_joint_pos[:, :, 1:, 0] - ctx_joint_pos[:, :, :1, 0]
        joint_positions[..., 2] = ctx_joint_pos[:, :, 1:, 2] - ctx_joint_pos[:, :, :1, 2]
        joint_rotation_ortho6d = matrix_to_cont6d(ctx_joint_rot)
        context_local_poses = t.cat(
            [
                joint_positions.view([batch_size, NUM_FT, -1]),
                joint_rotation_ortho6d.view([batch_size, NUM_FT, -1]),
            ],
            dim=-1,
        )

        # ----------------------------------------------------------------
        # Target 4 frames.
        #
        # ``target_local_root_values`` is broadcast from velocity_intent
        # (per-frame velocity / yaw_rate / hip_height the model should
        # see across the target window).
        #
        # ``target_global_root_values``:
        #   * In ``USE_IMPLIED_TARGET_POS`` mode (default, matches the
        #     G1 demo's ``target_root_realignment=True`` training-time
        #     distribution): set the target world-frame xz / heading
        #     to the *implied* position after integrating velocity_intent
        #     forward by ``TARGET_HORIZON_S`` seconds, and KEEP
        #     ``has_global_root_values[:, -NUM_FT:] = True``. Empirically
        #     this jumps the per-token forward-tracking slope from
        #     ~0.07 to ~0.95 on both X2 and G1 reference checkpoints
        #     (see ``scripts/probe_root_constraint_modes.py``).
        #   * In legacy mode (``use_implied_target_pos=False``): zero +
        #     mask off, reproducing the original velocity-only behavior.
        # ----------------------------------------------------------------
        target_local_root_values = velocity_intent[:, None, :].expand([batch_size, NUM_FT, 4]).contiguous()

        if self.USE_IMPLIED_TARGET_POS:
            # Progressive target trajectory across the NUM_FT target frames.
            # See ``_predict_with_pose_template`` for the rationale (TL;DR:
            # broadcasting one end-of-horizon waypoint to all 4 target
            # frames makes the model "park" at the waypoint, suppressing
            # locomotion). Frame time offsets:
            #     t_3 = TARGET_HORIZON_S        t_2 = TARGET_HORIZON_S - 1/fps
            #     t_1 = TARGET_HORIZON_S - 2/fps  t_0 = TARGET_HORIZON_S - 3/fps
            _MOTION_FPS = 30.0
            _frame_dt = 1.0 / _MOTION_FPS
            t_offsets = t.tensor(
                [self.TARGET_HORIZON_S - (NUM_FT - 1 - k) * _frame_dt for k in range(NUM_FT)],
                device=device,
                dtype=t.float32,
            )  # [NUM_FT]
            implied_x = (
                context_global_root_pos[:, -1:, 0]
                + velocity_intent[:, 1:2] * t_offsets[None, :]
            )  # [B, NUM_FT]
            implied_z = (
                context_global_root_pos[:, -1:, 2]
                + velocity_intent[:, 2:3] * t_offsets[None, :]
            )  # [B, NUM_FT]
            implied_y = velocity_intent[:, 3:4].expand([batch_size, NUM_FT])  # hip_height
            implied_target_pos = t.stack([implied_x, implied_y, implied_z], dim=-1)  # [B, NUM_FT, 3]
            implied_target_heading = (
                context_rotation_angle[:, -1:]
                + velocity_intent[:, 0:1] * t_offsets[None, :]
            )  # [B, NUM_FT]
            target_global_root_values = t.cat(
                [
                    implied_target_pos,
                    t.cos(implied_target_heading)[..., None],
                    t.sin(implied_target_heading)[..., None],
                ],
                dim=-1,
            )
        else:
            target_global_root_values = t.zeros([batch_size, NUM_FT, 5], device=device)
        target_local_poses = t.zeros_like(context_local_poses)

        # ----------------------------------------------------------------
        # Concatenate -> 8-frame constraint window.
        # ----------------------------------------------------------------
        local_root_values = t.cat([context_local_root_values, target_local_root_values], dim=1)
        global_root_values = t.cat([context_global_root_values, target_global_root_values], dim=1)
        local_poses = t.cat([context_local_poses, target_local_poses], dim=1)

        has_global_root_values = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
        has_local_root_values = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
        has_local_poses = t.ones_like(local_poses[:, :, 0], dtype=t.bool)
        # Last differenced velocity in the context window is invalid (no t+1
        # to differentiate against). Matches full_agent.py:457.
        has_local_root_values[:, NUM_FT - 1] = False
        if not self.USE_IMPLIED_TARGET_POS:
            # Legacy velocity-only mode: drop the world-frame target so the
            # network is free to integrate however it wants. Empirically
            # this collapses forward-tracking slope to ~0.07; kept only
            # for ablation diagnostics.
            has_global_root_values[:, -NUM_FT:] = False
        # No target keyframe pose: free-sample locomotion poses consistent
        # with the velocity + target_pos constraints. (Including a
        # target keyframe pose adds only ~3% tracking improvement on the
        # probe, and would require shipping a robot-specific stand-pose
        # template here; not worth the coupling.)
        has_local_poses[:, -NUM_FT:] = False

        if self.diagnostic_mask_hook is not None:
            has_global_root_values, has_local_root_values, has_local_poses = (
                self.diagnostic_mask_hook(
                    has_global_root_values,
                    has_local_root_values,
                    has_local_poses,
                    NUM_FT,
                )
            )

        if num_tokens is None:
            num_tokens_t = t.full([batch_size, 1], MASKED_NUM_TOKENS, dtype=t.int, device=device)
        else:
            num_tokens_t = t.full([batch_size, 1], int(num_tokens), dtype=t.int, device=device)

        config = {
            "num_inference_step": 1,
            "smooth_root_traj": False,
            "allow_pred_out_of_reach_num_tokens": False,
            "pose_token_sampling_use_argmax": True,
            "skip_ending_target_cond": self.SKIP_ENDING_TARGET_COND,
        }
        info: dict = {}
        pred_global_motions, num_pred_tokens = self._inferencer.predict(
            global_root_values,
            has_global_root_values,
            local_root_values,
            has_local_root_values,
            local_poses,
            has_local_poses,
            num_tokens_t,
            config=config,
            info=info,
        )

        model_features = pred_global_motions
        num_pred_frames = NUM_FT * num_pred_tokens

        mujoco_qpos = self._converter.convert_motion_features_to_mujoco_qpos(
            model_features, self._motion_rep, False
        )
        # Converter emits xyzw; flip to wxyz so the wire matches the heuristic
        # planner's convention (build_pose_payload reads root_quat_xyzw, but
        # the X2 converter has root_quat_w_first=False by default in the demo
        # path -- match that here).
        root_rot = mujoco_qpos[:, :, 3:7].clone()
        mujoco_qpos[:, :, 3:7] = root_rot[:, :, [3, 0, 1, 2]]

        if self.FORCE_CANONICALIZATION:
            input_state["mujoco_qpos"] = mujoco_qpos
            mujoco_qpos = self._uncanonicalize_mujoco_qpos(input_state)

        if self.FILTER_QPOS:
            self.frames["raw_mujoco_qpos"] = mujoco_qpos.clone()
            ctx = input_state["raw_context_mujoco_qpos"]
            num_ctx = ctx.shape[1]
            blend = t.linspace(0.3, 0.7, num_ctx)[None, :, None].to(ctx.device)
            mujoco_qpos[:, :num_ctx, :3] = (
                ctx[:, :, :3] * (1 - blend) + mujoco_qpos[:, :num_ctx, :3] * blend
            )
            mujoco_qpos[:, :num_ctx, 7:] = (
                ctx[:, :, 7:] * (1 - blend) + mujoco_qpos[:, :num_ctx, 7:] * blend
            )

        return model_features, mujoco_qpos, num_pred_frames

    # ------------------------------------------------------------------
    # Internal: pose-template prediction path. Mirrors
    # `_predict_with_velocity` (kept side-by-side rather than refactored
    # into a shared helper so the velocity-only fallback stays
    # byte-equivalent to its tested form). The only differences are in
    # the target window:
    #   - target_local_poses is filled from the realigned clip segment.
    #   - has_local_poses[:, -NUM_FT:] = True (not False).
    # `target_global_root_values` still comes from USE_IMPLIED_TARGET_POS
    # integration of velocity_intent.
    # ------------------------------------------------------------------

    @t.no_grad()
    def _predict_with_pose_template(
        self,
        input_state: dict,
        mode_idx: int,
        num_tokens: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        assert self._clip_library is not None
        batch_size = 1
        MASKED_NUM_TOKENS = self._inferencer._root_model.backbone_net.MASKED_NUM_TOKENS
        fps = self._inferencer.local_motion_rep.fps
        root_joint_idx = 0
        device = self._device
        NUM_FT = self.NUM_FRAMES_PER_TOKEN

        ctx_joint_pos = input_state["context_global_joint_positions"]   # [B, 4, J, 3]
        ctx_joint_rot = input_state["context_global_joint_rotations"]   # [B, 4, J, 3, 3]
        velocity_intent = input_state["velocity_intent"]                # [B, 4]

        # ----------------------------------------------------------------
        # Past 4 frames — identical to _predict_with_velocity.
        # ----------------------------------------------------------------
        context_global_root_pos = ctx_joint_pos[:, :, root_joint_idx, :]  # [B, 4, 3]
        context_rotation_angle = t.atan2(
            ctx_joint_rot[:, :, root_joint_idx, 0, 2],
            ctx_joint_rot[:, :, root_joint_idx, 2, 2],
        )  # [B, 4]
        context_global_root_values = t.cat(
            [
                context_global_root_pos,
                t.cos(context_rotation_angle)[..., None],
                t.sin(context_rotation_angle)[..., None],
            ],
            dim=-1,
        )  # [B, 4, 5]

        context_local_root_values = t.zeros([batch_size, NUM_FT, 4], device=device)
        context_local_root_values[:, : NUM_FT - 1, 0] = (
            ((context_rotation_angle[:, 1:] - context_rotation_angle[:, :-1] + t.pi) % (2 * t.pi))
            - t.pi
        ) * fps
        context_local_root_values[:, : NUM_FT - 1, 1:3] = (
            context_global_root_pos[:, 1:, [0, 2]] - context_global_root_pos[:, :-1, [0, 2]]
        ) * fps
        context_local_root_values[:, : NUM_FT - 1, 3] = (
            context_global_root_values[:, : NUM_FT - 1, 1]
        )
        context_local_root_values[:, NUM_FT - 1, :] = context_local_root_values[:, NUM_FT - 2, :]

        joint_positions_ctx = ctx_joint_pos[:, :, 1:, :].clone()
        joint_positions_ctx[..., 0] = ctx_joint_pos[:, :, 1:, 0] - ctx_joint_pos[:, :, :1, 0]
        joint_positions_ctx[..., 2] = ctx_joint_pos[:, :, 1:, 2] - ctx_joint_pos[:, :, :1, 2]
        joint_rotation_ortho6d_ctx = matrix_to_cont6d(ctx_joint_rot)
        context_local_poses = t.cat(
            [
                joint_positions_ctx.view([batch_size, NUM_FT, -1]),
                joint_rotation_ortho6d_ctx.view([batch_size, NUM_FT, -1]),
            ],
            dim=-1,
        )

        # ----------------------------------------------------------------
        # Target 4 frames (pose-template path):
        #   1. Compute the SCALAR implied target xz / heading from velocity
        #      intent (current pos + velocity * TARGET_HORIZON_S).
        #   2. Sample a 4-frame segment from the mode's clip.
        #   3. Compute a SINGLE diff_heading from the segment's FIRST frame
        #      heading and rotate ALL 4 frames by that same Y-rotation.
        #      Mirrors the uniform-rotation branch in
        #      ``full_navigation_agent._generate_target_joint_transforms``
        #      (full_agent.py:362-375). The clip's natural per-frame heading
        #      drift survives the alignment, so the model sees a consistent
        #      pose evolution within the target window.
        #   4. ``target_global_root_values`` broadcasts the implied target
        #      xz + hip + heading across all 4 frames (G1's spring model
        #      uses critical damping with ~0.4s halflife, so by t=1s the
        #      4-frame spring trajectory has effectively converged to the
        #      steady-state target; broadcasting matches that limit and
        #      avoids hand-rolling a spring here).
        #   5. ``target_local_root_values`` broadcasts ``velocity_intent``
        #      across all 4 frames (same convention as
        #      ``_predict_with_velocity``). Both the spatial target and
        #      the velocity target then agree.
        #   6. ``target_local_poses`` is built from the rotated clip
        #      joint positions (J-1 root-relative xyz, Y absolute) and
        #      the rotated joint rotations (cont6d). ``has_local_poses[:, -4:] = True``.
        # ----------------------------------------------------------------
        target_local_root_values = (
            velocity_intent[:, None, :].expand([batch_size, NUM_FT, 4]).contiguous()
        )

        # Progressive target trajectory across the 4 target frames.
        # The 4 target frames are the LAST NUM_FT frames of a TARGET_HORIZON_S
        # prediction horizon at the motion-rep native fps (30 Hz; matches
        # clip data fps and the model's training distribution). Frame
        # offsets from "now":
        #     t_3 = TARGET_HORIZON_S         (final target frame)
        #     t_2 = TARGET_HORIZON_S - 1/fps
        #     t_1 = TARGET_HORIZON_S - 2/fps
        #     t_0 = TARGET_HORIZON_S - 3/fps
        # Broadcasting the same end-of-horizon position to all 4 frames
        # (the previous MVP) forced the root to "stand still at the
        # waypoint" while the target joint poses showed a stepping gait;
        # the model resolved that conflict by walking-in-place. With a
        # progressive trajectory the spatial diff between adjacent
        # target frames matches the gait's forward push, so the model
        # can commit to actual locomotion at the commanded velocity.
        _MOTION_FPS = 30.0
        _frame_dt = 1.0 / _MOTION_FPS
        t_offsets = t.tensor(
            [self.TARGET_HORIZON_S - (NUM_FT - 1 - k) * _frame_dt for k in range(NUM_FT)],
            device=device,
            dtype=t.float32,
        )  # [NUM_FT]

        implied_x_4 = (
            context_global_root_pos[:, -1:, 0]
            + velocity_intent[:, 1:2] * t_offsets[None, :]
        )  # [B, NUM_FT]
        implied_z_4 = (
            context_global_root_pos[:, -1:, 2]
            + velocity_intent[:, 2:3] * t_offsets[None, :]
        )  # [B, NUM_FT]
        implied_hip_4 = velocity_intent[:, 3:4].expand([batch_size, NUM_FT])  # [B, NUM_FT]
        implied_heading_4 = (
            context_rotation_angle[:, -1:]
            + velocity_intent[:, 0:1] * t_offsets[None, :]
        )  # [B, NUM_FT]

        implied_target_pos_4 = t.stack(
            [implied_x_4, implied_hip_4, implied_z_4], dim=-1
        )  # [B, NUM_FT, 3]

        # Final-frame heading is the alignment reference for the clip rotation step below.
        implied_target_heading = implied_heading_4[:, -1]  # [B]

        target_global_root_values = t.cat(
            [
                implied_target_pos_4,
                t.cos(implied_heading_4)[..., None],
                t.sin(implied_heading_4)[..., None],
            ],
            dim=-1,
        )  # [B, NUM_FT, 5]

        # Sample a clip segment and apply a uniform Y-rotation.
        segment = self._clip_library.sample_target_segment(
            mode_idx=mode_idx, random_seed=random_seed
        )
        clip_joint_pos = segment["joint_positions"][None].to(device)   # [1, 4, J, 3]
        clip_joint_rot = segment["joint_rotations"][None].to(device)   # [1, 4, J, 3, 3]
        clip_headings = segment["headings"][None].to(device)           # [1, 4]

        # Single diff_heading: target_heading vs clip's FIRST-frame heading.
        diff_heading_single = (
            (implied_target_heading - clip_headings[:, 0] + t.pi) % (2 * t.pi) - t.pi
        )  # [B]
        corrective_mat = angle_to_Y_rotation_matrix(diff_heading_single).to(device).float()  # [B, 3, 3]
        corrective_mat_jnt = corrective_mat[:, None, None, :, :]  # broadcast over 4 frames, J joints

        aligned_joint_rot = t.matmul(corrective_mat_jnt, clip_joint_rot)
        aligned_joint_pos = t.matmul(
            corrective_mat_jnt, clip_joint_pos[..., None]
        )[..., 0]

        # Build target_local_poses with the realigned segment. joint_positions[:, 1:, :]
        # is root-XZ-relative (Y absolute, from the clip). Root rotation (joint 0)
        # IS included in the cont6d block.
        target_joint_positions = aligned_joint_pos[:, :, 1:, :]
        target_joint_rotation_ortho6d = matrix_to_cont6d(aligned_joint_rot)
        target_local_poses = t.cat(
            [
                target_joint_positions.reshape([batch_size, NUM_FT, -1]),
                target_joint_rotation_ortho6d.reshape([batch_size, NUM_FT, -1]),
            ],
            dim=-1,
        )

        # ----------------------------------------------------------------
        # Concatenate -> 8-frame constraint window.
        # ----------------------------------------------------------------
        local_root_values = t.cat([context_local_root_values, target_local_root_values], dim=1)
        global_root_values = t.cat([context_global_root_values, target_global_root_values], dim=1)
        local_poses = t.cat([context_local_poses, target_local_poses], dim=1)

        has_global_root_values = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
        has_local_root_values = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
        has_local_poses = t.ones_like(local_poses[:, :, 0], dtype=t.bool)
        has_local_root_values[:, NUM_FT - 1] = False
        # Target keyframe pose IS provided -> keep has_local_poses[:, -NUM_FT:] True.

        if self.diagnostic_mask_hook is not None:
            has_global_root_values, has_local_root_values, has_local_poses = (
                self.diagnostic_mask_hook(
                    has_global_root_values,
                    has_local_root_values,
                    has_local_poses,
                    NUM_FT,
                )
            )

        if num_tokens is None:
            num_tokens_t = t.full([batch_size, 1], MASKED_NUM_TOKENS, dtype=t.int, device=device)
        else:
            num_tokens_t = t.full([batch_size, 1], int(num_tokens), dtype=t.int, device=device)

        config = {
            "num_inference_step": 1,
            "smooth_root_traj": False,
            "allow_pred_out_of_reach_num_tokens": False,
            "pose_token_sampling_use_argmax": True,
            "skip_ending_target_cond": self.SKIP_ENDING_TARGET_COND,
        }
        info: dict = {}
        pred_global_motions, num_pred_tokens = self._inferencer.predict(
            global_root_values,
            has_global_root_values,
            local_root_values,
            has_local_root_values,
            local_poses,
            has_local_poses,
            num_tokens_t,
            config=config,
            info=info,
        )

        model_features = pred_global_motions
        num_pred_frames = NUM_FT * num_pred_tokens

        mujoco_qpos = self._converter.convert_motion_features_to_mujoco_qpos(
            model_features, self._motion_rep, False
        )
        root_rot = mujoco_qpos[:, :, 3:7].clone()
        mujoco_qpos[:, :, 3:7] = root_rot[:, :, [3, 0, 1, 2]]

        if self.FORCE_CANONICALIZATION:
            input_state["mujoco_qpos"] = mujoco_qpos
            mujoco_qpos = self._uncanonicalize_mujoco_qpos(input_state)

        if self.FILTER_QPOS:
            self.frames["raw_mujoco_qpos"] = mujoco_qpos.clone()
            ctx = input_state["raw_context_mujoco_qpos"]
            num_ctx = ctx.shape[1]
            blend = t.linspace(0.3, 0.7, num_ctx)[None, :, None].to(ctx.device)
            mujoco_qpos[:, :num_ctx, :3] = (
                ctx[:, :, :3] * (1 - blend) + mujoco_qpos[:, :num_ctx, :3] * blend
            )
            mujoco_qpos[:, :num_ctx, 7:] = (
                ctx[:, :, 7:] * (1 - blend) + mujoco_qpos[:, :num_ctx, 7:] * blend
            )

        return model_features, mujoco_qpos, num_pred_frames

    # ------------------------------------------------------------------
    # Canonicalize / uncanonicalize (lifted from full_agent.py, robot-agnostic).
    # ------------------------------------------------------------------

    def _canonicalize_mujoco_qpos(self, input_state: dict) -> None:
        """In-place canonicalize the context qpos to first-frame-origin + zero-heading."""
        mujoco_qpos = input_state["context_mujoco_qpos"]
        input_state["raw_context_mujoco_qpos"] = mujoco_qpos.clone()

        first_frame_position = (
            mujoco_qpos[:, 0, :3].clone() * t.tensor([[1.0, 1.0, 0.0]]).to(mujoco_qpos.device)
        )
        first_frame_rot = quaternion_to_matrix(mujoco_qpos[:, 0, 3:7].clone())
        first_frame_heading_angle = t.atan2(first_frame_rot[:, 1, 0], first_frame_rot[:, 0, 0])
        first_frame_heading_angle[first_frame_heading_angle.isnan()] = 0.0
        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)
        inverse_first_frame_rot_heading = first_frame_rot_heading.transpose(-2, -1)

        canonicalized_root_position = t.matmul(
            inverse_first_frame_rot_heading[:, None, :, :],
            (mujoco_qpos[:, :, :3].clone() - first_frame_position)[..., None],
        )[..., 0]
        canonicalized_rot_matrix = t.matmul(
            inverse_first_frame_rot_heading[:, None, :, :],
            quaternion_to_matrix(mujoco_qpos[:, :, 3:7]),
        )
        mujoco_qpos[:, :, 3:7] = matrix_to_quaternion(canonicalized_rot_matrix)
        mujoco_qpos[:, :, :3] = canonicalized_root_position

        input_state["first_frame_heading_angle"] = first_frame_heading_angle
        input_state["first_frame_position"] = first_frame_position
        input_state["context_mujoco_qpos"] = mujoco_qpos

    def _uncanonicalize_mujoco_qpos(self, input_state: dict) -> t.Tensor:
        """Undo `_canonicalize_mujoco_qpos`. Returns the world-frame qpos."""
        mujoco_qpos = input_state["mujoco_qpos"]
        first_frame_heading_angle = input_state["first_frame_heading_angle"]
        first_frame_position = input_state["first_frame_position"]

        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)
        current_first_frame_rotation = quaternion_to_matrix(mujoco_qpos[:, :1, 3:7])
        current_first_frame_heading_angle = t.atan2(
            current_first_frame_rotation[:, :, 1, 0],
            current_first_frame_rotation[:, :, 0, 0],
        )
        current_first_frame_rot_heading = angle_to_Z_rotation_matrix(current_first_frame_heading_angle)
        rot_matrix = quaternion_to_matrix(mujoco_qpos[:, :, 3:7])
        rot_matrix = t.matmul(
            first_frame_rot_heading[:, None, :, :],
            t.matmul(current_first_frame_rot_heading.transpose(-2, -1), rot_matrix),
        )
        root_positions = t.matmul(
            first_frame_rot_heading[:, None, :, :],
            t.matmul(
                current_first_frame_rot_heading.transpose(-2, -1),
                mujoco_qpos[:, :, :3, None],
            ),
        )[..., 0]
        root_positions = (
            root_positions
            - root_positions[:, :1, :] * t.tensor([[[1.0, 1.0, 0.0]]]).to(mujoco_qpos.device)
            + first_frame_position
        )
        mujoco_qpos[:, :, 3:7] = matrix_to_quaternion(rot_matrix)
        mujoco_qpos[:, :, :3] = root_positions
        return mujoco_qpos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_velocity_intent(self, velocity_intent: t.Tensor) -> t.Tensor:
        """Accept [4] or [B, 4]; return [B, 4] on self._device with float dtype."""
        if not isinstance(velocity_intent, t.Tensor):
            velocity_intent = t.as_tensor(velocity_intent, dtype=t.float32)
        if velocity_intent.ndim == 1:
            velocity_intent = velocity_intent[None, :]
        if velocity_intent.shape[-1] != 4:
            raise ValueError(
                f"velocity_intent last-dim must be 4 (yaw_rate, vel_x, vel_z, hip_h); "
                f"got shape {tuple(velocity_intent.shape)}"
            )
        return velocity_intent.to(self._device).float()
