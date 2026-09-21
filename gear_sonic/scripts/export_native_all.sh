#!/bin/bash
# ONE command per trained native 3-encoder checkpoint -> the complete, cross-gated,
# provenance-stamped ONNX set the X2 stacks consume:
#
#   <out>/<name>_g1.onnx              pose-ref graph (kplanner / pad)      obs 1670
#   <out>/<name>_smpl.onnx            fused smpl graph (gate input only)   obs 1830
#   <out>/<name>_smpl_tokenizer.onnx  smpl_obs -> token                    obs  840
#   <out>/<name>_g1_token.onnx        token-input graph (legacy overlay)   obs 1670
#   <out>/<name>_dual.onnx            NATIVE DUAL-HEAD (one session)       obs 2511
#   <out>/<name>.manifest.txt         md5 + provenance line per file
#
# Every file is stamped with the SAME codec_fingerprint (md5 of the .pt), the
# checkpoint name, export time and git sha; the dual export is parity-gated
# against the single-head graphs (gates A/B) and onnx_provenance.py must pass
# on the set before this script reports success. The launchers refuse a set
# that does not pass the same check, so a wrong swap between two trained
# models cannot reach the deploy.
#
#   ./gear_sonic/scripts/export_native_all.sh <ckpt.pt> <name> [out-dir]
#     e.g. ./gear_sonic/scripts/export_native_all.sh \
#            $CKPT_ROOT/<run>/last_it39000.pt x2_sonic_39000
#   out-dir defaults to <ckpt dir>/exported. config.yaml must sit next to the .pt
#   Run LOCALLY. Interpreters:
#     ISAACLAB_PYTHON  python of the IsaacLab conda env (steps 1-3; default:
#                      `python` of the active conda env, i.e. run inside it)
#     PY (auto)        repo .venv python for the provenance check, else python3
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_ISAAC="${ISAACLAB_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PY_ISAAC="${PY_ISAAC:-python}"
if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"; else PY="$(command -v python3)"; fi
CKPT="${1:?ckpt.pt}"; NAME="${2:?name, e.g. x2_sonic_39000}"; OUT="${3:-$(dirname "$CKPT")/exported}"
CKPT="${CKPT/#\~/$HOME}"; OUT="${OUT/#\~/$HOME}"
[[ -f "$CKPT" ]] || { echo "missing checkpoint $CKPT" >&2; exit 2; }
[[ -f "$(dirname "$CKPT")/config.yaml" ]] || echo "WARNING: no config.yaml next to $CKPT (eval_agent_trl needs it)" >&2
mkdir -p "$OUT"
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
echo "[export-all] $CKPT -> $OUT/$NAME_*  (md5 $(md5sum "$CKPT" | cut -c1-8)...)"

step() { echo; echo "[export-all] --- $* ---"; }

step "1/4 pose graph (g1 encoder) + fused smpl graph, IsaacLab fidelity-checked"
DUMPS="$OUT/.step0_dumps"; mkdir -p "$DUMPS"
for ENC in g1 smpl; do
  # fresh dump PER ENCODER and PER CHECKPOINT (a stale /tmp dump from an
  # earlier export fails the fidelity check with a misleading message)
  "$PY_ISAAC" -m gear_sonic.scripts.dump_isaaclab_step0 checkpoint="$CKPT" +headless=True \
    ++dump_path="$DUMPS/${NAME}_step0_${ENC}.pt" ++encoder_name="$ENC"
  "$PY_ISAAC" -m gear_sonic.scripts.reexport_x2_g1_onnx --run-dir "$(dirname "$CKPT")" --checkpoint "$CKPT" \
    --encoder-name "$ENC" --decoder-name g1_dyn --dump "$DUMPS/${NAME}_step0_${ENC}.pt" \
    --output "$OUT/${NAME}_${ENC}.onnx"
done

step "2/4 smpl tokenizer + token graph, pose graph stamped with the lineage"
"$PY_ISAAC" gear_sonic/scripts/native_token_onnx_export.py --checkpoint "$CKPT" \
  --fused-smpl "$OUT/${NAME}_smpl.onnx" --name "$NAME" --out-dir "$OUT" \
  --stamp-pose-graph "$OUT/${NAME}_g1.onnx" --force

step "3/4 native dual-head graph (gated against the single-head graphs)"
"$PY_ISAAC" gear_sonic/scripts/native_dual_head_onnx_export.py --checkpoint "$CKPT" --name "$NAME" --out-dir "$OUT"

step "4/4 provenance check + manifest"
"$PY" gear_sonic/scripts/onnx_provenance.py check --pose "$OUT/${NAME}_g1.onnx" --dual "$OUT/${NAME}_dual.onnx" \
  --token "$OUT/${NAME}_g1_token.onnx" --tokenizer "$OUT/${NAME}_smpl_tokenizer.onnx"
{
  echo "# $NAME exported $(date -u +%Y-%m-%dT%H:%M:%SZ) from $CKPT (md5 $(md5sum "$CKPT" | cut -d' ' -f1)) at $(git -C "$REPO" rev-parse --short HEAD)"
  (cd "$OUT" && md5sum "${NAME}"_*.onnx)
  "$PY" gear_sonic/scripts/onnx_provenance.py show "$OUT/${NAME}"_*.onnx
} > "$OUT/${NAME}.manifest.txt"
cat "$OUT/${NAME}.manifest.txt"
echo
echo "[export-all] DONE. Sim:   MODEL=$OUT/${NAME}_dual.onnx  (dual)   or   MODEL=$OUT/${NAME}_g1.onnx  (legacy)"
echo "[export-all]       Robot: scp ${NAME}_g1.onnx ${NAME}_dual.onnx (+ manifest) to PC2 policies/; X2_RITUAL_MODEL=policies/${NAME}_dual.onnx (the ritual re-checks the pairing)."
