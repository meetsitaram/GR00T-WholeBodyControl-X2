#!/usr/bin/env python3
"""Pre-ignition contract check: does the ROBOT's binary agree with the MODEL?

WHY THIS EXISTS (2026-08-24 incident). sim_onnx_planner's identity gate compares
only FILE IDENTITY -- sonic md5, planner md5s, handoff marker count. It never
inspects the deploy BINARY. So it reported "GATE CLEAN" while:

  1. the robot's binary carried LEGACY action scales and the pushed ONNX sidecar
     carried VENDOR ones -- 29/31 scales disagreed, waist x1.736 -> waist pitch
     AND roll pinned at the +-0.45 clamp on 26/45 ticks; and
  2. X2_ANCHOR_ORI_MODE was 'b' (from kplanner_profile.env) while the model
     declares 'heading' -- a silent convention mismatch that is invisible while
     upright and diverges exactly as the robot leans.

Both are SEMANTIC compatibility, which md5s cannot express. This checks it.

    python preflight_deploy_contract.py --sidecar <onnx>.phi.json \
        --header <policy_parameters.hpp> [--ori-mode b|heading]

Exit 0 = compatible. Non-zero = DO NOT IGNITE.
"""
from __future__ import annotations
import argparse, json, re, sys, pathlib

def arr_double(src, name):
    m = re.search(rf'const std::array<double, 31> {name} = \{{(.*?)\n\}};', src, re.S)
    if not m: raise SystemExit(f"could not find {name} in header")
    return [float(x) for x in re.findall(r'(-?\d+\.?\d*(?:e-?\d+)?)\s*,', m.group(1))]

def arr_int(src, name):
    m = re.search(rf'const std::array<int, 31> {name} = \{{(.*?)\}};', src, re.S)
    if not m: raise SystemExit(f"could not find {name} in header")
    return [int(x) for x in re.findall(r'(-?\d+)', m.group(1))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True, help="<model>.onnx.phi.json")
    ap.add_argument("--header",  required=True, help="policy_parameters.hpp the ROBOT's binary was built from")
    ap.add_argument("--ori-mode", default=None, help="X2_ANCHOR_ORI_MODE the robot will run with")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    src = pathlib.Path(a.header).read_text()
    binv = arr_double(src, "x2_action_scale")
    mj2il = arr_int(src, "mujoco_to_isaaclab")
    side = json.load(open(a.sidecar))["action_scales"]["x2"]

    fails = []
    # the sidecar is ISAACLAB-ordered, the binary MuJoCo-ordered. Permute before
    # comparing -- comparing raw looks like a 16/31 mismatch that is not real.
    perm = [side[i] for i in mj2il]
    bad = [(i, perm[i], binv[i]) for i in range(31) if abs(perm[i] - binv[i]) > a.tol]
    if bad:
        fails.append(f"ACTION SCALES: {len(bad)}/31 disagree between sidecar and binary")
        for i, s, b in bad[:5]:
            fails.append(f"    mj[{i:2d}] sidecar={s:.7f} binary={b:.7f}  x{b/s:.3f}")
        worst = max(bad, key=lambda t: abs(t[2]/t[1] - 1))
        fails.append(f"    worst amplification x{worst[2]/worst[1]:.3f} -> commands scaled wrong on the ROBOT")
    else:
        print(f"  action scales : 31/31 agree (after mujoco_to_isaaclab permutation)  OK")

    if a.ori_mode is not None:
        want = None
        for k in ("ori_obs_name", "convention", "format"):
            v = json.load(open(a.sidecar)).get(k)
            if isinstance(v, str) and "heading" in v: want = "heading"; break
            if isinstance(v, str) and "anchor_ori_b" in v: want = "b"; break
        if want is None:
            print(f"  ori mode      : model does not declare one; running '{a.ori_mode}'  (UNVERIFIED)")
        elif want != a.ori_mode:
            fails.append(f"ORI MODE: model requires '{want}' but robot runs '{a.ori_mode}' "
                         f"-- silent; invisible upright, diverges with tilt")
        else:
            print(f"  ori mode      : model wants '{want}', robot runs '{a.ori_mode}'  OK")

    if fails:
        print("\n  *** CONTRACT VIOLATION -- DO NOT IGNITE ***")
        for f in fails: print(f"  {f}")
        return 1
    print("\n  contract OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
