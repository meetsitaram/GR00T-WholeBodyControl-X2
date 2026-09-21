"""FrozenCore T2 actor module: frozen GEAR-SONIC G1 core + Phi codec, batched.

Drop-in ``actor_module`` for :class:`gear_sonic.trl.modules.
actor_critic_modules.Actor` (``input_obs_dict=True``): consumes the X2
env's obs dict, runs the codec + frozen G1 encoder/FSQ/decoder inside
the policy graph, and emits X2-space action means (B, 31).

Design decisions (ledger: the frozen-core design ledger (not shipped), T2 sections):
  * STATELESS: the G1-native last-action history the decoder expects is
    recovered by EXACT inverse affine from the X2 action history already
    in actor_obs (roundtrip error ~1e-16; only env-side clipping at 20
    can perturb it, rare). No per-env buffers, no reset masking.
  * Actor obs must be RAW (running_mean_std=false, incumbent convention)
    — the codec algebra is in physical units.
  * LoRA (optional cfg) injects into the g1_dyn decoder only; encoder +
    FSQ stay frozen (preserves the shared token space). With lora cfg
    absent the module is EXACTLY the T1 zero-shot policy.
  * Gradients: encoder is frozen and FSQ rounds, so the only gradient
    path is decoder LoRA — no straight-through estimator needed.

Smoke test (no Isaac):  python -m gear_sonic.trl.modules.frozen_core_g1_actor
cross-checks batched torch output against the T1 numpy wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

NUM_G1, NUM_X2, NFRAMES = 29, 31, 10
SMPL_OBS_DIM = 840          # 10 frames x (72 joints + 6 root-ori + 6 wrist)
SMPL_PER_FRAME = 84
G1_WRIST_SLICE = slice(23, 29)   # G1 IL wrist rows
X2_PROP_DIM = 990
G1_ACTION_CLIP = 20.0

# Default G1 SONIC checkpoint: the ``sonic_release/last.pt`` that
# ``download_from_hf.py --training`` places under ``$SONIC_HOME/g1``
# (public ``nvidia/GEAR-SONIC``). Override with ``g1_checkpoint=`` in the
# actor config or ``$SONIC_G1_CHECKPOINT``.
_HF_CKPT = Path(
    os.environ.get("SONIC_G1_CHECKPOINT")
    or os.path.join(
        os.environ.get("SONIC_HOME", os.path.join(os.path.expanduser("~"), ".cache", "sonic")),
        "g1", "sonic_release", "last.pt",
    )
)


class FrozenCoreG1ActorModule(nn.Module):
    def __init__(self, env_config=None, algo_config=None,
                 g1_checkpoint: str | None = None,
                 lora: dict | None = None,
                 cmd_obs_name: str = "command_multi_future_nonflat",
                 ori_obs_name: str = "motion_anchor_ori_b_mf_nonflat",
                 proprio_key: str = "actor_obs",
                 tokenizer_key: str = "tokenizer",
                 # --- SMPL encoder mode (curriculum resume; off by default) ---
                 enable_smpl: bool = False,
                 remap_wrist_ranges: bool = False,
                 remap_waist_ranges: bool = False,
                 smpl_joints_obs_name: str = "smpl_joints_multi_future_local_nonflat",
                 smpl_ori_obs_name: str = "smpl_root_ori_heading_multi_future",
                 smpl_wrist_obs_name: str = "joint_pos_multi_future_wrist_for_smpl",
                 encoder_index_obs_name: str = "encoder_index",
                 **_ignored):
        super().__init__()
        from gear_sonic.trl.modules.onnx_helpers import (
            FsqQuantizer, _mlp_from_state_dict, tolerant_torch_load)
        from gear_sonic.scripts.frozen_core_sonic_codec import (
            SonicPhiCodec, harvest_g1_params, harvest_x2_params)

        sd = tolerant_torch_load(str(g1_checkpoint or _HF_CKPT))["policy_state_dict"]
        self.encoder, enc_dims = _mlp_from_state_dict(
            sd, "actor_module.encoders.g1.module.")
        self.decoder, dec_dims = _mlp_from_state_dict(
            sd, "actor_module.decoders.g1_dyn.module.")
        assert enc_dims[0] == 640 and dec_dims[-1] == NUM_G1
        self.fsq = FsqQuantizer(levels=32)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        codec = SonicPhiCodec(harvest_g1_params(), harvest_x2_params())

        # ---- TRAIN/DEPLOY WRIST MAPPING ---------------------------------
        # DEFAULT false ON PURPOSE. The codec tensors below are registered
        # persistent=False -- they are NOT in any checkpoint and are rebuilt
        # from THIS code on every load. So flipping this unconditionally would
        # retroactively change how every existing checkpoint behaves, silently
        # (same shapes, different values). Every pre-2026-08-22 config omits
        # the key and therefore keeps the exact affine it trained with.
        #
        # What it fixes when enabled: remap_wrist_ranges() is called in
        # FrozenCoreSmplActor (the DEPLOY composite actor) but never here, so a
        # model trained without it is executed with it. Measured gap:
        #     wrist pitch  enc_a 0.943 -> 2.893   (3.07x)
        #     wrist roll   enc_a 0.973 -> 1.407   (1.45x), b -64.7deg -> +34.1deg
        #     wrist yaw    unchanged (X2 yaw range exceeds its G1 source)
        # i.e. training passes G1's full pitch range into an X2 joint with ~35%
        # of that travel, then deploy corrects it. Enable for NEW runs so both
        # sides derive wrist coefficients identically.
        if remap_wrist_ranges:
            from gear_sonic.scripts.frozen_core_sonic_codec import (
                remap_wrist_ranges as _remap_wrist_ranges)
            _remap_wrist_ranges(codec)
            print("[frozen-core-actor] wrist range-map APPLIED (train==deploy)",
                  flush=True)
        if remap_waist_ranges:
            # S1 lineage: fixes the range-blind waist_pitch fit (dec_a
            # +1.287 into a joint with 67% of G1's travel — the S0 robot
            # snap-back root cause). Opt-in; see remap_waist_ranges().
            from gear_sonic.scripts.frozen_core_sonic_codec import (
                remap_waist_ranges as _remap_waist_ranges)
            _remap_waist_ranges(codec)
            print("[frozen-core-actor] waist range-map APPLIED (train==deploy)",
                  flush=True)

        phi = codec.phi
        f64 = dict(dtype=torch.float64)

        # persistent=False: these are CODEC CONSTANTS, not learned state. If
        # they persist into state_dict, a checkpoint load silently restores
        # the training-time affine (b=0 wrists) over any later codec fix —
        # cost one bit-identical "re-eval" on 2026-08-15 before diagnosis.
        def reg(name, t):
            self.register_buffer(name, t, persistent=False)
        reg("enc_src", torch.as_tensor(phi["enc_src"]))
        reg("enc_a", torch.as_tensor(phi["enc_a"], **f64))
        reg("enc_b", torch.as_tensor(phi["enc_b"], **f64))
        dec_rows = torch.as_tensor(codec.dec_rows)
        reg("dec_rows", dec_rows)
        reg("dec_gsrc", torch.as_tensor(codec.dec_gsrc))
        reg("dec_a", torch.as_tensor(phi["dec_a"][codec.dec_rows], **f64))
        reg("dec_b", torch.as_tensor(phi["dec_b"][codec.dec_rows], **f64))
        reg("d_g1", torch.as_tensor(codec.d_g1, **f64))
        reg("s_g1", torch.as_tensor(codec.s_g1, **f64))
        reg("d_x2", torch.as_tensor(codec.d_x2, **f64))
        reg("s_x2", torch.as_tensor(codec.s_x2, **f64))

        # ---- SMPL branch -----------------------------------------------
        # The release/v1.1 encoders share ONE aligned latent before FSQ, so an
        # smpl-encoder token decodes through the same g1_dyn. The 840-D smpl
        # obs is 10 frames x [72 SMPL joint coords | 6D root ori | 6 wrist dofs]
        # -- 78 of 84 values per frame are HUMAN-frame and embodiment-neutral.
        # Only the 6 wrist dofs are robot-specific, and the codec already holds
        # their X2->G1 affine.
        self.enable_smpl = bool(enable_smpl)
        self.smpl_encoder = None
        if self.enable_smpl:
            self.smpl_encoder, smpl_dims = _mlp_from_state_dict(
                sd, "actor_module.encoders.smpl.module.")
            assert smpl_dims[0] == SMPL_OBS_DIM and smpl_dims[-1] == enc_dims[-1], (
                f"smpl encoder {smpl_dims} incompatible with g1 {enc_dims}")
            for p_ in self.smpl_encoder.parameters():
                p_.requires_grad_(False)
            print(f"[frozen-core-actor] smpl encoder loaded {smpl_dims} (FROZEN)")

        # G1 wrist rows 23..28 come from X2 rows enc_src[23:29] with a per-row
        # affine. VERIFIED 2026-08-21: enc_src[23:29] == [25,26,27,28,29,30]
        # and a = [-1.0,-1.0,0.943,0.929,0.973,0.960],
        #         b = [-0.17,0,0,+0.70,-1.13,+0.18]
        # -- TWO SIGN FLIPS and offsets up to 1.13 rad. Feeding RAW X2 wrist
        # dofs to the frozen G1 encoder is badly wrong; those b terms ARE the
        # wrist-saga b-fix. The env obs term MUST use joints_idx=enc_src[23:29]
        # so the 6 values arrive in G1 wrist-row order.
        reg("wrist_src", torch.as_tensor(phi["enc_src"])[G1_WRIST_SLICE])
        reg("wrist_a", torch.as_tensor(phi["enc_a"], **f64)[G1_WRIST_SLICE])
        reg("wrist_b", torch.as_tensor(phi["enc_b"], **f64)[G1_WRIST_SLICE])

        self.proprio_key, self.tokenizer_key = proprio_key, tokenizer_key
        self.cmd_obs_name, self.ori_obs_name = cmd_obs_name, ori_obs_name
        self.smpl_joints_obs_name = smpl_joints_obs_name
        self.smpl_ori_obs_name = smpl_ori_obs_name
        self.smpl_wrist_obs_name = smpl_wrist_obs_name
        self.encoder_index_obs_name = encoder_index_obs_name
        self._tok_slices = None
        if env_config is not None and getattr(env_config.obs, "group_obs_dims", None):
            self._build_tok_slices(env_config.obs.group_obs_dims["tokenizer"],
                                   env_config.obs.group_obs_names["tokenizer"])

        if lora:
            from gear_sonic.trl.modules.lora import inject_lora, mark_only_lora_trainable
            inject_lora(self.decoder, lora.get("targets", ["*"]),
                        rank=int(lora.get("rank", 16)),
                        alpha=float(lora.get("alpha", 16.0)))
            tr, tot = mark_only_lora_trainable(self)
            print(f"[frozen-core-actor] LoRA on decoder: {tr/1e3:.1f}k trainable "
                  f"/ {tot/1e6:.1f}M total")

    _CODEC_CONST_KEYS = ("enc_src", "enc_a", "enc_b", "dec_rows", "dec_gsrc",
                         "dec_a", "dec_b", "d_g1", "s_g1", "d_x2", "s_x2")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before 2026-08-15 persisted the codec constants as
        # buffers. They are rebuilt fresh in __init__ (persistent=False now);
        # drop the stale copies so strict loading neither restores them over a
        # newer codec calibration nor errors on the unexpected keys.
        for name in self._CODEC_CONST_KEYS:
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _build_tok_slices(self, dims_map, names):
        off, slices = 0, {}
        for name in names:
            d = int(np.prod(dims_map[name]))
            slices[name] = (off, d)
            off += d
        want_dims = [(self.cmd_obs_name, 620), (self.ori_obs_name, 60)]
        if self.enable_smpl:
            # 10 frames x 24 joints x 3 = 720 | 10 x 6 = 60 | 10 x 6 = 60  -> 840
            want_dims += [(self.smpl_joints_obs_name, 720),
                          (self.smpl_ori_obs_name, 60),
                          (self.smpl_wrist_obs_name, 60),
                          (self.encoder_index_obs_name, None)]
        for want, exp in want_dims:
            if want not in slices:
                raise KeyError(f"tokenizer group lacks '{want}' (has {list(slices)})")
            if exp is not None and slices[want][1] != exp:
                raise ValueError(f"{want}: dim {slices[want][1]} != {exp}")
        self._tok_slices = slices

    # ------------------------------------------------------------------
    def _phi_q(self, q_x2):          # absolute joint positions (B,*,31)->29
        return (q_x2[..., self.enc_src] - self.enc_b) / self.enc_a

    def _phi_v(self, v_x2):
        return v_x2[..., self.enc_src] / self.enc_a

    def _act_x2_to_g1(self, act_x2):  # exact inverse of the decode affine
        q_x2 = self.d_x2 + act_x2 * self.s_x2
        q_g1_from = (q_x2[..., self.dec_rows.long()] - self.dec_b) / self.dec_a
        act = torch.zeros(*act_x2.shape[:-1], NUM_G1,
                          dtype=act_x2.dtype, device=act_x2.device)
        act[..., self.dec_gsrc.long()] = ((q_g1_from - self.d_g1[self.dec_gsrc.long()])
                                          / self.s_g1[self.dec_gsrc.long()])
        return act

    def _act_g1_to_x2(self, act_g1):
        q_g1 = self.d_g1 + act_g1 * self.s_g1
        q_x2 = self.d_x2.expand(*act_g1.shape[:-1], NUM_X2).clone()
        q_x2[..., self.dec_rows.long()] = (
            self.dec_a * q_g1[..., self.dec_gsrc.long()] + self.dec_b)
        return (q_x2 - self.d_x2) / self.s_x2

    def _slice(self, tok, name):
        o, d = self._tok_slices[name]
        return tok[:, o:o + d]

    def _blend_smpl_latent(self, tok, latent_g1, B, f32):
        """Replace the g1 latent with an smpl-encoder latent on smpl envs.

        `encoder_index` is MULTI-HOT (commands.py uses |=), so an episode can
        have g1 AND smpl active at once. This module emits ONE action, so the
        action path must pick a single encoder: smpl wins where it is set,
        because that is the distribution the Pico/teleop deployment actually
        drives. g1 remains the fallback everywhere else.

        Only 6 of the 84 values per frame are robot-specific; the 72 SMPL joint
        coords and the 6D root orientation are HUMAN-frame and pass through
        untouched. The wrists are mapped X2->G1 by the codec affine (sign flips
        and up-to-1.13rad offsets -- see the note in __init__).
        """
        idx = self._slice(tok, self.encoder_index_obs_name)
        smpl_col = self._smpl_index_col(idx.shape[1])
        if smpl_col is None:
            return latent_g1
        use = idx[:, smpl_col] > 0.5
        if not bool(use.any()):
            return latent_g1

        joints = self._slice(tok, self.smpl_joints_obs_name)           # (B, 720)
        ori = self._slice(tok, self.smpl_ori_obs_name)                 # (B, 60)
        wr_x2 = self._slice(tok, self.smpl_wrist_obs_name)             # (B, 60)
        wr_g1 = ((wr_x2.reshape(B, NFRAMES, 6) - self.wrist_b) / self.wrist_a)

        per = torch.cat([joints.reshape(B, NFRAMES, 72),
                         ori.reshape(B, NFRAMES, 6),
                         wr_g1], dim=2)                                # (B,10,84)
        smpl_obs = per.reshape(B, SMPL_OBS_DIM)
        assert smpl_obs.shape[1] == SMPL_OBS_DIM, smpl_obs.shape
        latent_smpl = self.smpl_encoder(smpl_obs.to(f32))
        return torch.where(use.unsqueeze(-1), latent_smpl, latent_g1)

    def _smpl_index_col(self, ncols):
        """Column of `encoder_index` holding the smpl flag.

        Order comes from encoder_sample_probs (an ORDERED dict). Cached on the
        module; env_config carries the names when available.
        """
        if getattr(self, "_smpl_col", "unset") != "unset":
            return self._smpl_col
        col = 2 if ncols >= 3 else None      # {g1, teleop, smpl} config order
        names = getattr(self, "_encoder_names", None)
        if names and "smpl" in names:
            col = list(names).index("smpl")
        self._smpl_col = col
        return col

    # ------------------------------------------------------------------
    def forward(self, obs_dict, **_kw):
        # All manager-env obs tensors have leading (B, S, ...) — S=1 at
        # rollout (UniversalTokenModule.forward docstring). Normalize to
        # (B*S, D) and restore the leading shape on the action output.
        prop_in = obs_dict[self.proprio_key]
        assert prop_in.shape[-1] == X2_PROP_DIM, (
            f"{self.proprio_key} last dim {prop_in.shape[-1]} != {X2_PROP_DIM}")
        lead = prop_in.shape[:-1]                    # (B,) or (B, S)
        prop = prop_in.double().reshape(-1, X2_PROP_DIM)
        B = prop.shape[0]

        tok_in = obs_dict[self.tokenizer_key]
        if self._tok_slices is None:
            raise RuntimeError("tokenizer slices not built (env_config missing)")
        spec_total = sum(d for _, d in self._tok_slices.values())
        assert tok_in.shape[-1] == spec_total, (
            f"tokenizer last dim {tok_in.shape[-1]} != spec {spec_total}; "
            f"slices={self._tok_slices}")
        tok = tok_in.double().reshape(-1, spec_total)
        co, cd = self._tok_slices[self.cmd_obs_name]
        oo, od = self._tok_slices[self.ori_obs_name]
        cmd = tok[:, co:co + cd]                     # (B, 620) = [jp310|jv310]
        ori = tok[:, oo:oo + od].reshape(B, NFRAMES, 6)
        assert cmd.shape[1] == 2 * NFRAMES * NUM_X2, cmd.shape
        jp = cmd[:, :NFRAMES * NUM_X2].reshape(B, NFRAMES, NUM_X2)
        jv = cmd[:, NFRAMES * NUM_X2:].reshape(B, NFRAMES, NUM_X2)

        jp_g1 = self._phi_q(jp)                             # (B, 10, 29) absolute
        jv_g1 = self._phi_v(jv)
        cmd_g1 = torch.cat([jp_g1.reshape(B, -1), jv_g1.reshape(B, -1)], dim=1)
        tok_g1 = torch.cat([cmd_g1.reshape(B, NFRAMES, 2 * NUM_G1), ori],
                           dim=2).reshape(B, 640)           # T0-probed layout

        n = NFRAMES * NUM_X2
        angvel, rest = prop[:, :30], prop[:, 30:]
        jp_rel = rest[:, :n].reshape(B, NFRAMES, NUM_X2)
        jv_p = rest[:, n:2 * n].reshape(B, NFRAMES, NUM_X2)
        act_hist = rest[:, 2 * n:3 * n].reshape(B, NFRAMES, NUM_X2)
        grav = rest[:, 3 * n:3 * n + 30]
        jp_g1_rel = self._phi_q(jp_rel + self.d_x2) - self.d_g1
        act_g1_hist = self._act_x2_to_g1(act_hist)
        prop_g1 = torch.cat([
            angvel, jp_g1_rel.reshape(B, -1), self._phi_v(jv_p).reshape(B, -1),
            act_g1_hist.reshape(B, -1), grav], dim=1)       # (B, 930)

        f32 = obs_dict[self.proprio_key].dtype
        latent = self.encoder(tok_g1.to(f32))

        if self.enable_smpl:
            latent = self._blend_smpl_latent(tok, latent, B, f32)

        token = self.fsq(latent)
        act_g1 = self.decoder(torch.cat([token, prop_g1.to(f32)], dim=-1))
        act_g1 = act_g1.clamp(-G1_ACTION_CLIP, G1_ACTION_CLIP)
        act_x2 = self._act_g1_to_x2(act_g1.double()).to(f32)
        return act_x2.reshape(*lead, NUM_X2)         # (B[,S], 31), head = 0


def _smoke() -> int:
    """Cross-parity vs the T1 numpy wrapper on identical random obs."""
    import types
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    names = ["command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"]
    dims = {"command_multi_future_nonflat": (10, 62),
            "motion_anchor_ori_b_mf_nonflat": (10, 6)}
    env_config = types.SimpleNamespace(obs=types.SimpleNamespace(
        group_obs_dims={"tokenizer": dims}, group_obs_names={"tokenizer": names}))
    mod = FrozenCoreG1ActorModule(env_config=env_config).eval()

    from gear_sonic.scripts.frozen_core_sonic_codec import FrozenCoreG1SonicActor
    ref = FrozenCoreG1SonicActor()

    B = 8
    prop = rng.normal(0, 0.3, (B, 990)).astype(np.float32)
    jp = rng.normal(0, 0.5, (B, 310)).astype(np.float32)
    jv = rng.normal(0, 1.0, (B, 310)).astype(np.float32)
    ori = rng.normal(0, 0.6, (B, 60)).astype(np.float32)
    tok_x2_flat = np.concatenate([np.concatenate([jp, jv], 1), ori], 1)

    with torch.no_grad():
        out = mod({"actor_obs": torch.from_numpy(prop),
                   "tokenizer": torch.from_numpy(tok_x2_flat)}).numpy()
        # the histories the torch path derived (validates the inverse map)
        hist_g1 = mod._act_x2_to_g1(
            torch.from_numpy(prop[:, 650:960]).double().reshape(B, 10, 31)
        ).float().numpy()

    # reference path: numpy wrapper expects the 680 harness layout and a
    # G1-NATIVE history buffer — inject the inverse-mapped one so both
    # sides see identical decoder inputs.
    worst = 0.0
    for i in range(B):
        blk = np.concatenate([
            np.concatenate([jp[i], jv[i]]).reshape(10, 62),
            ori[i].reshape(10, 6)], axis=1).reshape(-1)
        ref.reset()
        ref._g1_action_hist = hist_g1[i].copy()
        r = ref(prop[i], blk).numpy().ravel()
        worst = max(worst, float(np.abs(r - out[i]).max()))
    head_cols = [c for c in range(NUM_X2)
                 if c not in mod.dec_rows.long().tolist()]
    print(f"[frozen-core-actor] batched-vs-numpy parity max diff {worst:.2e} "
          f"(B={B}); head cols {head_cols} max |a| "
          f"{np.abs(out[:, head_cols]).max():.1e}")
    ok = worst < 1e-4
    print(f"[frozen-core-actor] SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
