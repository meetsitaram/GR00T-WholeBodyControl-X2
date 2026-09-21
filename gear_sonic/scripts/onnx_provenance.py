#!/usr/bin/env python3
"""ONNX provenance: SHOW what an export is, CHECK that a set belongs together.

Every X2 SONIC export carries metadata_props stamped by its exporter
(`reexport_x2_g1_onnx` + `native_token_onnx_export.py --stamp-pose-graph`,
`native_dual_head_onnx_export.py`): `checkpoint` / `source_checkpoint`
(the .pt file name), `codec_fingerprint` (`native-3enc:<md5 prefix of the
.pt>`), `graph_kind`, and (newer exports) `source_md5`, `export_utc`,
`git_sha`. The fingerprint is the identity that matters: two files with the
same fingerprint came from the SAME .pt, whatever they are named.

    onnx_provenance.py show  A.onnx [B.onnx ...]
    onnx_provenance.py check --pose <name>_g1.onnx [--dual <name>_dual.onnx]
                             [--token <name>_g1_token.onnx] [--tokenizer <name>_smpl_tokenizer.onnx]

`check` exits 3 when any two files disagree on the fingerprint or the
checkpoint name, when a graph has the wrong kind / obs width for its slot,
or when a file that must be stamped is not. The sim stack and the PC2
ritual call it before the deploy is started; the operator gets one line
per file that says exactly which .pt each graph came from.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

EXPECT = {
    # slot: (graph_kind or None for the plain pose graph, obs width)
    "pose": (None, 1670),
    "dual": ("native_dual_head", 2511),
    "token": ("any2any_token_deploy", 1670),
    "tokenizer": (None, 840),
}


def _md5(p: Path, n: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()


def info(path: str | Path) -> dict:
    import onnx
    p = Path(path).expanduser()
    m = onnx.load(str(p), load_external_data=False)
    meta = {e.key: e.value for e in m.metadata_props}
    inp = m.graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    return {
        "path": str(p), "name": p.name, "md5": _md5(p),
        "input": inp.name, "obs": dims[-1] if dims else -1,
        "graph_kind": meta.get("graph_kind", ""),
        "checkpoint": meta.get("checkpoint") or meta.get("source_checkpoint") or "",
        "fingerprint": meta.get("codec_fingerprint", ""),
        "source_md5": meta.get("source_md5", ""),
        "export_utc": meta.get("export_utc", ""),
        "git_sha": meta.get("git_sha", ""),
        "ori_mode": meta.get("ori_mode", ""),
    }


def line(i: dict) -> str:
    kind = i["graph_kind"] or "pose-graph"
    ck = i["checkpoint"] or "UNSTAMPED"
    fp = i["fingerprint"] or "-"
    extra = f" ori={i['ori_mode']}" if i["ori_mode"] else ""
    when = f" exported {i['export_utc']}" if i["export_utc"] else ""
    return (f"{i['name']:40s} md5 {i['md5'][:8]}  obs {i['obs']:5d}  {kind:22s} "
            f"from {ck}  fp {fp}{extra}{when}")


def check(slots: dict[str, str]) -> int:
    infos = {s: info(p) for s, p in slots.items() if p}
    bad = []
    for s, i in infos.items():
        kind, obs = EXPECT[s]
        if kind is not None and i["graph_kind"] != kind:
            bad.append(f"{s}: {i['name']} has graph_kind='{i['graph_kind']}', expected '{kind}'")
        if kind is None and s == "pose" and i["graph_kind"] == "any2any_token_deploy":
            bad.append(f"pose: {i['name']} is a TOKEN graph, not the pose graph")
        if i["obs"] != obs:
            bad.append(f"{s}: {i['name']} obs width {i['obs']}, expected {obs}")
        if s == "dual" and not i["fingerprint"]:
            bad.append(f"{s}: {i['name']} carries no codec_fingerprint (re-export with native_dual_head_onnx_export.py)")
    # the dual exporter stamped an 8-hex prefix until 2026-09-04, the token
    # exporter the full md5: compare on the shared prefix, print in full
    def _fp_key(fp: str) -> str:
        lin, _, h = fp.partition(":")
        return f"{lin}:{h[:8]}"
    fps = {s: _fp_key(i["fingerprint"]) for s, i in infos.items() if i["fingerprint"]}
    # checkpoint names are compared only among fingerprinted (native) graphs:
    # the frozen-G1 frozen-core lineage pairs a token graph from a merged .pt
    # with a different base pose graph BY DESIGN
    # basename only: the frozen-core token exporter stamps the FULL path of the merged .pt, the pose
    # exporter the bare name (s1ft16000: "s1ft_it16000_merged.pt" vs "/home/.../s1ft_it16000_merged.pt"
    # refused a set whose fingerprints matched, 2026-09-06)
    cks = {s: os.path.basename(i["checkpoint"]) for s, i in infos.items() if i["checkpoint"] and i["fingerprint"]}
    if len(set(fps.values())) > 1:
        bad.append("codec_fingerprint MISMATCH -- these graphs come from DIFFERENT checkpoints: "
                   + ", ".join(f"{s}={v}" for s, v in fps.items()))
    if len(set(cks.values())) > 1:
        bad.append("checkpoint name MISMATCH: " + ", ".join(f"{s}={v}" for s, v in cks.items()))
    unst = [s for s, i in infos.items() if not i["fingerprint"]]
    for s, i in infos.items():
        print("[provenance] " + line(i))
    if unst and fps:
        print(f"[provenance] WARNING: {', '.join(unst)} unstamped -- cannot prove it pairs with the "
              f"stamped graphs (stamp it: native_token_onnx_export.py --stamp-pose-graph)")
    if bad:
        for b in bad:
            print(f"[provenance] FAIL: {b}")
        print("[provenance] REFUSING: this ONNX set does not belong together.")
        return 3
    if fps:
        print(f"[provenance] OK: {len(infos)} graph(s), one checkpoint: "
              f"{next(iter(cks.values()), '?')} ({next(iter(fps.values()))})")
    else:
        print("[provenance] OK (no lineage stamps to compare -- pre-native export)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("files", nargs="+")
    c = sub.add_parser("check")
    c.add_argument("--pose"); c.add_argument("--dual"); c.add_argument("--token"); c.add_argument("--tokenizer")
    a = ap.parse_args()
    if a.cmd == "show":
        for f in a.files:
            print(line(info(f)))
        return 0
    slots = {"pose": a.pose, "dual": a.dual, "token": a.token, "tokenizer": a.tokenizer}
    if not any(slots.values()):
        ap.error("check needs at least one of --pose/--dual/--token/--tokenizer")
    return check(slots)


if __name__ == "__main__":
    sys.exit(main())
