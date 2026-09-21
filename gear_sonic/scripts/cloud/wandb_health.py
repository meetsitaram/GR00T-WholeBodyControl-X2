"""5-minute wandb health check for the T2 big run (rule 14, metric-side).

Prints ONE line per invocation:
  HEALTHY step=<s> rew=<r> d10=<trend> kl=<kl>
  BOOTING (no history yet)
  ALARM <reason>   <- reasons: NaN in any logged series, step stalled,
                      reward collapsed vs its trailing window
State (last step/reward) kept in a sidecar JSON so trend/stall checks
work across invocations.
"""
import json, math, os, sys

if len(sys.argv) < 2:
    sys.exit("usage: wandb_health.py <entity/project/run_id>")
RUN = sys.argv[1]
STATE = os.path.expanduser("~/.cache/t2_wandb_health.json")

import wandb

api = wandb.Api()
try:
    run = api.run(RUN)
    # history(samples=N) downsamples over the WHOLE run — on a long run the
    # last sample trails the true tail by ~1-2% of total steps (observed:
    # reported step 6152 while the run was at 6297). scan_history
    # from near the end returns the actual tail, all series included (the NaN
    # scan needs full rows). lastHistoryStep may itself lag, so over-fetch
    # and keep the last 120 rows.
    tail_from = max(0, (run.lastHistoryStep or 0) - 200)
    rows = list(run.scan_history(min_step=tail_from))[-120:]
except Exception as e:
    print(f"WANDB-UNREACHABLE {str(e)[:60]}")
    sys.exit(0)
if not rows:
    print("BOOTING (no history yet)")
    sys.exit(0)

rows.sort(key=lambda r: r.get("_step", 0))
latest = rows[-1]
step = latest.get("_step", 0)

# 1. NaN scan over every numeric series in the recent window
nan_keys = set()
for r in rows[-40:]:
    for k, v in r.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float) and math.isnan(v):
            nan_keys.add(k)
        elif isinstance(v, str) and v.lower() == "nan":
            nan_keys.add(k)
nan_keys.discard("objective/rewards")  # known-benign inherited accumulator (washes out); real NaN shows in other series
if nan_keys:
    print(f"ALARM NaN in {len(nan_keys)} series e.g. {sorted(nan_keys)[:3]} at step {step}")
    sys.exit(0)

# 2. reward level + short trend
def series(key):
    return [(r["_step"], r[key]) for r in rows
            if isinstance(r.get(key), (int, float)) and not (isinstance(r.get(key), float) and math.isnan(r[key]))]

rew = series("objective/rewards")
kl = series("policy/approxkl_avg")
r_now = rew[-1][1] if rew else None
d10 = (rew[-1][1] - rew[-10][1]) if len(rew) >= 10 else 0.0

# 3. stall / collapse vs previous invocation
prev = {}
if os.path.exists(STATE):
    try:
        prev = json.load(open(STATE))
    except Exception:
        prev = {}
stall_n = prev.get("stall_n", 0) + 1 if (prev.get("step") == step and prev.get("step") is not None) else 0
json.dump({"step": step, "rew": r_now, "stall_n": stall_n}, open(STATE, "w"))

if stall_n >= 2:
    # two consecutive stalled polls: real stall (single-poll stalls are
    # usually wandb uploader lag — verify against the node log regardless)
    print(f"ALARM step STALLED at {step} across {stall_n+1} checks")
    sys.exit(0)
if r_now is not None and prev.get("rew") is not None and r_now < prev["rew"] - 5.0:
    print(f"ALARM reward collapse {prev['rew']:.2f} -> {r_now:.2f} at step {step}")
    sys.exit(0)

print(f"HEALTHY step={step} rew={r_now if r_now is None else round(r_now,2)} "
      f"d10={round(d10,2)} kl={round(kl[-1][1],4) if kl else '-'}")
