#!/usr/bin/env python3
"""Generate policy_parameters.hpp's plant tables from a plant YAML.

WHY THIS EXISTS. policy_parameters.hpp hardcodes kps[], kds[], x2_action_scale[]
and default_angles[] and applies them to WHATEVER model the binary is handed.
That made it a seventh, uncoordinated copy of the plant (armature_investigation
§9) and the direct cause of the §12 deploy blocker: a vendor-trained policy fed
legacy-armature gains is 1.16-2.86x too soft and drifts 1.4 m while told to
stand still.

This regenerates those tables from the single source of truth so the binary and
the trainer cannot disagree by hand-edit.

    python gen_policy_parameters.py --plant vendor_20260823 [--write]

Without --write it prints a diff against the current header and exits non-zero
if they differ, which makes it usable as a CI/pre-deploy gate.

!! THE BINARY IMPLEMENTS EXACTLY ONE PLANT AT A TIME. Regenerating for vendor
!! makes it correct for vendor-trained models (v13+) and WRONG for legacy-trained
!! ones (v0, v12) -- which would then run waist_yaw 2.86x TOO STIFF. Over-stiff
!! is the dangerous direction. Do not ship a binary whose plant does not match
!! the model being deployed. The real fix is plant_loader.hpp (runtime YAML).
"""
from __future__ import annotations
import argparse, importlib.util, pathlib, re, sys

REPO = pathlib.Path(__file__).resolve().parents[2]
HDR = (REPO / "gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include"
             / "policy_parameters.hpp")

def _load(name):
    spec = importlib.util.spec_from_file_location(
        "plant_config", REPO / "gear_sonic/envs/manager_env/robots/plant_config.py")
    pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
    return pc.load_plant(name)

def _joint_names():
    sys.path.insert(0, str(REPO / "gear_sonic/scripts"))
    from eval_x2_mujoco import MUJOCO_JOINT_NAMES
    return list(MUJOCO_JOINT_NAMES)

def table(name, vals, names, comment):
    body = "\n".join(f"    {v:.8f}, // {n}" for v, n in zip(vals, names))
    # `inline` (not const): the tables are runtime-overridable by plant_loader
    # (X2_PLANT / X2_PLANT_YAML) since 2026-09-08.
    tail = "          // runtime-overridable (plant_loader, X2_PLANT) since 2026-09-08" if name == "kps" else ""
    return f"// {comment}\ninline std::array<double, 31> {name} = {{{tail}\n{body}\n}};"

def build(plant):
    names = _joint_names()
    p = _load(plant)
    return {
        "kps": (
            [p.kp_for(j) for j in names],
            f"kp[i] = armature * (2*pi*10 Hz)^2   [plant: {plant}]"),
        "kds": (
            [p.kd_for(j) for j in names],
            f"kd[i] = 2 * zeta * armature * (2*pi*10 Hz)   [plant: {plant}]"),
        "x2_action_scale": (
            [p.action_scale_for(j) for j in names],
            f"0.25 * effort / (armature * w^2)   [plant: {plant}]"),
        "default_angles": (
            [p.default_pos_for(j) for j in names],
            f"default joint positions (rad)   [plant: {plant}]"),
    }, names

def splice(src, arr, text):
    pat = re.compile(rf"(//[^\n]*\n)*(const|inline) std::array<double, 31> {arr} = \{{.*?\n\}};",
                     re.S)
    if not pat.search(src):
        raise SystemExit(f"could not locate array {arr} in the header")
    return pat.sub(lambda _: text, src, count=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plant", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", default="", help="comma list, e.g. kps,kds")
    a = ap.parse_args()
    tables, names = build(a.plant)
    keep = [x for x in a.only.split(",") if x] or list(tables)
    src = HDR.read_text(); out = src
    for arr in keep:
        vals, comment = tables[arr]
        out = splice(out, arr, table(arr, vals, names, comment))
    if out == src:
        print(f"  header already matches plant '{a.plant}' for {keep}"); return 0
    if a.write:
        HDR.write_text(out); print(f"  WROTE {HDR.relative_to(REPO)}  [{a.plant}] {keep}")
        return 0
    print(f"  header DIFFERS from plant '{a.plant}' for {keep} (use --write)")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
