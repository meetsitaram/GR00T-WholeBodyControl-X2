#!/usr/bin/env python3
"""Verify a deploy ONNX's observation contract against what a consumer builds.

THE BUG THIS EXISTS TO PREVENT. The anchor-orientation term has two variants of
IDENTICAL SHAPE (10 frames x 6 = 60 values):

  motion_anchor_ori_b_mf_nonflat        normalize by the robot's FULL
                                        orientation (incl. pitch/roll).
                                        Built by the C++ deploy tokenizer;
                                        what PRE-v1.1 models were trained on.
  motion_anchor_ori_heading_mf_nonflat  normalize by the robot's YAW ONLY,
                                        preserving the reference's pitch/roll
                                        vs gravity. What every V1.1-LINEAGE
                                        model expects.

Feeding the wrong one raises nothing. Measured divergence: the orientation
error handed to the policy equals the robot's own tilt, 1:1 --

    tilt   0deg ->  0.0deg error      tilt  20deg -> 20.0deg error
    tilt   5deg ->  5.0deg error      tilt  30deg -> 30.0deg error
    tilt  10deg -> 10.0deg error      tilt  45deg -> 44.9deg error

So it is EXACTLY ZERO while the robot stands upright and grows as it tilts.
Every gentle bring-up rung (static nudge, idle stand, slow walk) passes, and
the error appears once the robot is already leaning -- i.e. when it can least
afford a wrong reference. A related ordering bug in this same term once made
the policy "correct" a fictitious yaw error by spinning ~180deg in 1-2 s
(see eval_x2_mujoco.py's note on row-major vs column-major 6D).

Usage:
    onnx_obs_contract.py MODEL.onnx                 # report what it declares
    onnx_obs_contract.py MODEL.onnx --require heading   # exit 1 on mismatch
    onnx_obs_contract.py MODEL.onnx --require b
"""

import argparse
import sys

HEADING = "motion_anchor_ori_heading_mf_nonflat"
BODY = "motion_anchor_ori_b_mf_nonflat"


def read_contract(path):
    """Metadata via onnxruntime first -- the ROBOT has onnxruntime (the deploy
    needs it) but NOT the `onnx` package, and a check you cannot run where it
    matters is not a check. Falls back to `onnx` on machines that have it."""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return dict(s.get_modelmeta().custom_metadata_map or {})
    except ImportError:
        pass
    import onnx
    m = onnx.load(path, load_external_data=False)
    return {q.key: q.value for q in m.metadata_props}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--require", choices=("heading", "b"), default=None,
                    help="what the CONSUMER builds; exit 1 if the model "
                         "disagrees")
    a = ap.parse_args()

    meta = read_contract(a.onnx)
    declared = meta.get("ori_obs_name", "")

    print(f"model    : {a.onnx}")
    if not meta:
        print("contract : *** NOT DECLARED *** (exported before contract "
              "stamping)")
    else:
        for k in ("ori_obs_name", "obs_dim", "action_dim", "source_checkpoint",
                  "phi_sha256"):
            if meta.get(k):
                print(f"{k:9s}: {meta[k]}")

    if a.require is None:
        return 0

    want = HEADING if a.require == "heading" else BODY
    if not declared:
        print(f"\nRESULT   : CANNOT VERIFY -- model declares no contract, "
              f"consumer builds {want}.")
        print("           An undeclared model is NOT evidence of a match. "
              "Re-export it,")
        print("           or confirm its lineage by hand before deploying.")
        return 1
    if declared == want:
        print(f"\nRESULT   : OK -- model wants {declared}, consumer builds it.")
        return 0

    print(f"\nRESULT   : *** MISMATCH ***")
    print(f"           model    wants : {declared}")
    print(f"           consumer builds: {want}")
    print("           Same shape, so nothing will error. The policy will read")
    print("           its orientation reference in the wrong frame; the error")
    print("           equals the robot's tilt and is ZERO while it stands")
    print("           upright, so bring-up checks will not reveal it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
