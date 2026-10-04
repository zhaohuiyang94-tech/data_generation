#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/miniconda3/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/pipeline.v42.json}"
DECOMPOSITIONS="${DECOMPOSITIONS:-$PROJECT_ROOT/data/cwq/decompose_test_pred_cwq_25328_vllm.json}"
QUESTION_WORKERS="${QUESTION_WORKERS:-4}"
RUN_NAME="${CWQ_RUN_NAME:-cwq_v42_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${CWQ_RUN_DIR:-$PROJECT_ROOT/outputs/$RUN_NAME}"
NO_DECOMPOSITION_REVIEW="${NO_DECOMPOSITION_REVIEW:-0}"

if [[ "$NO_DECOMPOSITION_REVIEW" != "1" && -z "${GLM_API_KEY:-}" ]]; then
  echo "GLM_API_KEY is required unless NO_DECOMPOSITION_REVIEW=1" >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/verify_bundle.py" >/dev/null
mkdir -p "$RUN_DIR/artifacts"

ARGS=(
  --config "$CONFIG"
  --decompositions "$DECOMPOSITIONS"
  --start 0
  --limit 0
  --question-workers "$QUESTION_WORKERS"
  --output "$RUN_DIR/results.json"
  --errors-output "$RUN_DIR/errors.json"
  --artifacts-dir "$RUN_DIR/artifacts"
  --allow-incomplete-resume
  --no-write-only-nonperfect-json
)
if [[ "$NO_DECOMPOSITION_REVIEW" == "1" ]]; then
  ARGS+=(--no-decomposition-review)
fi

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src" PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" -u -m semantic_guided_kbqa.cli "${ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/run.log"

echo "run_dir=$RUN_DIR"
