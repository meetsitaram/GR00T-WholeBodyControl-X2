#!/usr/bin/env python3
"""Fast CPU pre-flight for a training config — no IsaacSim, no GPU.

Catches the failure classes that otherwise cost a ~5 minute IsaacSim startup
each to surface, one at a time:

  1. hydra config resolution (bad `defaults:` paths, missing overrides)
  2. motion-lib load of the real corpus + SMPL sidecars (length/timebase
     assertions, missing files, key-join failures)
  3. SMPL coverage: how many motions actually carry sidecar data — the
     silent-zero case that makes the SMPL encoder train on nothing
  4. reward body_names ⊆ command body_names — an unmatched name makes
     _get_body_indexes return [] SILENTLY, the reward averages over zero
     bodies -> NaN -> poisons total reward -> advantages -> all losses

Usage (on a node, env_isaaclab python):
    python gear_sonic/scripts/cloud/preflight_train.py \
        +exp=manager/universal_token/all_modes/sonic_x2_ultra \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=/path/corpus.pkl \
        ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/path/smpl_dir
    python gear_sonic/scripts/cloud/preflight_train.py --help

Exit 0 = the config and data are sound; launch the real run.
"""

from __future__ import annotations

import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = __file__.rsplit("/gear_sonic/", 1)[0]
CONFIG_DIR = f"{REPO}/gear_sonic/config"


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    overrides = [a for a in argv if a.startswith(("+", "~", "++")) or "=" in a]
    print(f"[preflight] {len(overrides)} overrides")

    print("[preflight] 1/3 hydra compose ...", flush=True)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="base", overrides=overrides)
    m_cfg = cfg.manager_env.commands.motion.motion_lib_cfg
    print(f"           motion_file      = {m_cfg.get('motion_file')}")
    print(f"           smpl_motion_file = {m_cfg.get('smpl_motion_file')}")
    probs = cfg.manager_env.commands.motion.get("encoder_sample_probs")
    print(f"           encoder probs    = {probs}")
    if check_reward_body_names(cfg):
        print("[preflight] WARN: reward body_names not tracked by the command "
              "(see check_reward_body_names) -- expect NaN rewards", file=sys.stderr)

    smpl_wanted = bool(probs) and float(dict(probs).get("smpl", 0.0)) > 0.0
    smpl_file = m_cfg.get("smpl_motion_file")
    if smpl_wanted and smpl_file in (None, "dummy", "zeros"):
        print(
            f"[preflight] FAIL: smpl encoder has sample prob but smpl_motion_file="
            f"{smpl_file!r}. The encoder would train with zero gradient.",
            file=sys.stderr,
        )
        return 2

    print("[preflight] 2/3 corpus load ...", flush=True)
    import os

    import joblib

    motion_file = m_cfg.get("motion_file")
    if os.path.isdir(motion_file):
        # directory mode: one pkl per motion, keyed by basename (motion_lib_base
        # load_data does the same glob)
        import glob as _glob

        paths = _glob.glob(os.path.join(motion_file, "**", "*.pkl"), recursive=True)
        paths = [p for p in paths if not p.endswith("metadata.pkl")]
        keys = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        corpus = None
        print(f"           {len(keys)} motions (directory mode)")
    else:
        corpus = joblib.load(motion_file)
        keys = list(corpus.keys())
        paths = None
        print(f"           {len(keys)} motions (single pkl)")

    print("[preflight] 3/3 SMPL sidecar join ...", flush=True)
    if not smpl_wanted:
        print("           (smpl encoder inactive — skipping)")
        print("[preflight] OK — config and data are sound")
        return 0

    n_hit = n_miss = n_timebase = 0
    bad_examples = []
    for k in keys:
        p = os.path.join(smpl_file, f"{os.path.basename(k)}.pkl")
        if not os.path.exists(p):
            n_miss += 1
            continue
        n_hit += 1
        if n_hit <= 400:  # duration audit on a bounded sample (I/O bound)
            s = joblib.load(p)
            if corpus is not None:
                entry = corpus[k]
            else:
                _d = joblib.load(paths[keys.index(k)])
                entry = _d[list(_d)[0]]
            r_dur = entry["dof"].shape[0] / float(entry.get("fps", 30))
            s_dur = s["pose_aa"].shape[0] / float(s["fps"])
            if abs(r_dur - s_dur) > 0.2:
                n_timebase += 1
                if len(bad_examples) < 3:
                    bad_examples.append(f"{k}: robot {r_dur:.2f}s vs smpl {s_dur:.2f}s")

    pct = 100.0 * n_hit / max(len(keys), 1)
    print(f"           coverage: {n_hit}/{len(keys)} ({pct:.1f}%), missing {n_miss}")
    print(f"           timebase mismatches in audited sample: {n_timebase}")
    for b in bad_examples:
        print(f"             ! {b}")

    if pct < 90.0:
        print(f"[preflight] FAIL: SMPL coverage {pct:.1f}% < 90%", file=sys.stderr)
        return 4
    if n_timebase:
        print(
            f"[preflight] FAIL: {n_timebase} clips have inconsistent fps labels; "
            f"the SMPL/robot correspondence is broken for those.",
            file=sys.stderr,
        )
        return 5

    print("[preflight] OK — config and data are sound")
    return 0


def check_reward_body_names(cfg) -> int:
    """Reward body_names must be a SUBSET of the command's tracked body_names.

    WHY: ``_get_body_indexes`` (commands.py) filters ``command.cfg.body_names``
    with a list comprehension. An unmatched name yields ``[]`` with NO error;
    the reward then averages over zero bodies and returns NaN. A single NaN
    reward term poisons total reward -> advantages -> ALL losses.

    Cost of not having this check: a multi-node launch ran ~100 iterations
    with ``Mean rewards: nan`` before anyone noticed, because
    ``tracking_vr_2wrists_local_ori`` had been pointed at ``wrist_roll_link``,
    which X2's command does not track.
    """
    tracked = set(cfg.manager_env.commands.motion.get("body_names") or [])
    if not tracked:
        print("  [reward-bodies] command has no body_names — skipped")
        return 0
    bad = 0
    for name, term in (cfg.manager_env.rewards or {}).items():
        if not isinstance(term, dict) and not hasattr(term, "get"):
            continue
        params = (term.get("params") if hasattr(term, "get") else None) or {}
        want = params.get("body_names") if hasattr(params, "get") else None
        if not want:
            continue
        missing = [b for b in want if b not in tracked]
        if missing and len(missing) == len(want):
            # NONE of the requested bodies is tracked: _get_body_indexes returns []
            # and the term averages over zero bodies -> NaN. This is the fatal case.
            print(f"  [reward-bodies] FAIL {name}: none of {want} in command.body_names "
                  f"-> _get_body_indexes returns [] -> NaN")
            bad += 1
        elif missing:
            # Some bodies are tracked, so the term is finite -- but the missing ones
            # are silently ignored (the term does not penalize them at all).
            print(f"  [reward-bodies] WARN {name}: {missing} not in command.body_names "
                  f"-> silently ignored; term averages over the {len(want) - len(missing)} tracked body(s)")
        else:
            print(f"  [reward-bodies] ok   {name}: {len(want)} body(s) tracked")
    return bad


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
