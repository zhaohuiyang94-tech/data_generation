#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTORY_ROOT="${FACTORY_ROOT:-/workspace/LlamaFactory-main}"
FACTORY_CLI="${FACTORY_CLI:-/workspace/conda-envs/finetune/bin/llamafactory-cli}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/miniconda3/bin/python3}"
GENERATOR="${DECOMPOSITION_GENERATOR:-$PROJECT_ROOT/scripts/generate_cwq_decompose_predictions.py}"
WEBQSP_QUESTIONS="${WEBQSP_QUESTIONS:-$PROJECT_ROOT/data/webqsp/decompose_test_questions.json}"
DECOMPOSITION_OUTPUT="${WEBQSP_DECOMPOSITION_OUTPUT:-$PROJECT_ROOT/data/webqsp/decompose_test_pred_webqsp_from_cwq_model.json}"
PIPELINE_CONFIG="${WEBQSP_CONFIG:-$PROJECT_ROOT/configs/pipeline.webqsp_cwq_transfer.glm_review.json}"
RUN_NAME="${WEBQSP_RUN_NAME:-webqsp_cwq_model_glm_review_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${WEBQSP_RUN_DIR:-$PROJECT_ROOT/outputs/$RUN_NAME}"
QUESTION_WORKERS="${QUESTION_WORKERS:-4}"
DECOMPOSITION_WORKERS="${DECOMPOSITION_WORKERS:-8}"
START_INDEX="${WEBQSP_START:-0}"
LIMIT="${WEBQSP_LIMIT:-0}"
SERVICE_TIMEOUT="${SERVICE_TIMEOUT:-900}"

SEMANTIC_CONFIG="$PROJECT_ROOT/configs/model_services/llama3_1_8b_lora_semantic_goldfirst_vllm.yaml"
DECOMPOSITION_CONFIG="$PROJECT_ROOT/configs/model_services/llama3_1_8b_lora_decompose_goldfirst_vllm.yaml"

if [[ -z "${GLM_API_KEY:-}" ]]; then
  echo "GLM_API_KEY is required for WebQSP GLM decomposition review." >&2
  exit 2
fi
for path in \
  "$FACTORY_CLI" "$GENERATOR" "$WEBQSP_QUESTIONS" "$PIPELINE_CONFIG" \
  "$SEMANTIC_CONFIG" "$DECOMPOSITION_CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "required path is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/service_logs" "$(dirname "$DECOMPOSITION_OUTPUT")"

service_ready() {
  local port="$1"
  curl --noproxy '*' --silent --show-error --fail --max-time 5 \
    "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

wait_ready() {
  local name="$1" port="$2" pid="$3"
  local deadline=$((SECONDS + SERVICE_TIMEOUT))
  while ! service_ready "$port"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name process exited before port $port became ready" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "timed out waiting for $name on port $port" >&2
      return 1
    fi
    sleep 5
  done
  echo "$name healthy on port $port (pid=$pid)"
}

wait_down() {
  local name="$1" port="$2"
  local deadline=$((SECONDS + 180))
  while service_ready "$port"; do
    if ((SECONDS >= deadline)); then
      echo "timed out waiting for $name port $port to stop" >&2
      return 1
    fi
    sleep 2
  done
}

stop_group() {
  local name="$1" pid="$2"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local pgid
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  if [[ -z "$pgid" || "$pgid" == "1" ]]; then
    echo "refusing to stop invalid $name process group: $pgid" >&2
    return 1
  fi
  echo "stopping $name pid=$pid pgid=$pgid"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 2
  done
  echo "$name did not stop after TERM; sending KILL" >&2
  kill -KILL -- "-$pgid" 2>/dev/null || true
}

start_service() {
  local name="$1" port="$2" alias="$3" config="$4" log="$5"
  setsid env \
    CUDA_VISIBLE_DEVICES=1,2 \
    API_HOST=127.0.0.1 \
    API_PORT="$port" \
    API_MODEL_NAME="$alias" \
    API_VERBOSE=0 \
    VLLM_USE_V1=0 \
    VLLM_ATTENTION_BACKEND=XFORMERS \
    PYTHONPATH="$FACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$FACTORY_CLI" api "$config" >"$log" 2>&1 < /dev/null &
  STARTED_PID=$!
  echo "$STARTED_PID" >"${log%.log}.pid"
  wait_ready "$name" "$port" "$STARTED_PID"
}

semantic_pid() {
  pgrep -f 'llamafactory-cli api .*/llama3_1_8b_lora_semantic_goldfirst_vllm.yaml$' \
    | head -1 || true
}

decomposition_complete() {
  [[ -f "$DECOMPOSITION_OUTPUT" ]] || return 1
  jq -e '
    length == 1639 and
    all(.[];
      (.prediction | type) == "array" and
      (.prediction | length) > 0 and
      (.validation.json_valid == true) and
      (.validation.schema_valid == true)
    )
  ' "$DECOMPOSITION_OUTPUT" >/dev/null
}

DECOMPOSITION_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$DECOMPOSITION_PID" ]] && kill -0 "$DECOMPOSITION_PID" 2>/dev/null; then
    stop_group decomposition "$DECOMPOSITION_PID" || true
    wait_down decomposition 18001 || true
  fi
  if ! service_ready 18002; then
    echo "restoring CWQ Semantic service on GPUs 1,2"
    start_service semantic 18002 semantic "$SEMANTIC_CONFIG" \
      "$RUN_DIR/service_logs/semantic_vllm.log" || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if decomposition_complete; then
  echo "reuse complete CWQ-model WebQSP decomposition: $DECOMPOSITION_OUTPUT"
else
  CURRENT_SEMANTIC_PID="$(semantic_pid)"
  if service_ready 18002 && [[ -z "$CURRENT_SEMANTIC_PID" ]]; then
    echo "port 18002 is healthy but its Semantic PID cannot be resolved; aborting safely" >&2
    exit 3
  fi
  if [[ -n "$CURRENT_SEMANTIC_PID" ]]; then
    stop_group semantic "$CURRENT_SEMANTIC_PID"
    wait_down semantic 18002
  fi

  echo "starting CWQ Decomposition model on former Semantic GPUs 1,2"
  start_service decomposition 18001 decomposition "$DECOMPOSITION_CONFIG" \
    "$RUN_DIR/service_logs/decomposition_vllm.log"
  DECOMPOSITION_PID="$STARTED_PID"

  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    "$PYTHON_BIN" -u "$GENERATOR" \
      --input "$WEBQSP_QUESTIONS" \
      --output "$DECOMPOSITION_OUTPUT" \
      --base-url http://127.0.0.1:18001/v1 \
      --model decomposition \
      --workers "$DECOMPOSITION_WORKERS" \
      --timeout 300 \
      --retries 2 \
      --checkpoint-every 25 \
      --resume \
      2>&1 | tee "$RUN_DIR/generate_decomposition.log"

  if ! decomposition_complete; then
    echo "generated decomposition is incomplete or invalid: $DECOMPOSITION_OUTPUT" >&2
    jq -r '
      to_entries[] |
      select(
        (.value.prediction | type) != "array" or
        (.value.prediction | length) == 0 or
        .value.validation.json_valid != true or
        .value.validation.schema_valid != true
      ) | .key
    ' "$DECOMPOSITION_OUTPUT" | head -50 >&2 || true
    exit 4
  fi

  stop_group decomposition "$DECOMPOSITION_PID"
  wait_down decomposition 18001
  DECOMPOSITION_PID=""

  echo "restarting CWQ Semantic model on GPUs 1,2"
  start_service semantic 18002 semantic "$SEMANTIC_CONFIG" \
    "$RUN_DIR/service_logs/semantic_vllm.log"
fi

for port in 18002 18003 18004 18005; do
  if ! service_ready "$port"; then
    echo "required model service is unavailable on port $port" >&2
    exit 5
  fi
done
if ! curl --noproxy '*' --silent --show-error --fail --max-time 10 \
    http://127.0.0.1:8008/health >/dev/null; then
  echo "BGE embedding service is unavailable on port 8008" >&2
  exit 5
fi
if ! curl --noproxy '*' --silent --show-error --fail --max-time 10 \
    'http://127.0.0.1:3005/sparql?query=ASK%20%7B%20%3Fs%20%3Fp%20%3Fo%20%7D&format=application%2Fsparql-results%2Bjson' \
    >/dev/null; then
  echo "Freebase SPARQL service is unavailable on port 3005" >&2
  exit 5
fi

echo "running WebQSP with CWQ-trained models and CWQ-family GLM review"
env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
  -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  PYTHONPATH="$PROJECT_ROOT/src" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" -u -m semantic_guided_kbqa.cli \
    --config "$PIPELINE_CONFIG" \
    --decompositions "$DECOMPOSITION_OUTPUT" \
    --start "$START_INDEX" \
    --limit "$LIMIT" \
    --question-workers "$QUESTION_WORKERS" \
    --output "$RUN_DIR/results.json" \
    --errors-output "$RUN_DIR/errors.json" \
    --artifacts-dir "$RUN_DIR/artifacts" \
    --no-write-only-nonperfect-json \
    2>&1 | tee "$RUN_DIR/run.log"

echo "webqsp_run_dir=$RUN_DIR"
echo "webqsp_decomposition=$DECOMPOSITION_OUTPUT"
