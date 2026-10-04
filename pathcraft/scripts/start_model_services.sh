#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTORY_ROOT="${FACTORY_ROOT:-/workspace/LlamaFactory-main}"
FACTORY_CLI="${FACTORY_CLI:-/workspace/conda-envs/finetune/bin/llamafactory-cli}"
LOG_DIR="${MODEL_LOG_DIR:-$PROJECT_ROOT/logs/model_services}"
START_TIMEOUT="${MODEL_START_TIMEOUT:-900}"

if [[ ! -x "$FACTORY_CLI" ]]; then
  echo "LlamaFactory CLI is not executable: $FACTORY_CLI" >&2
  exit 2
fi
if [[ ! -d "$FACTORY_ROOT/src" ]]; then
  echo "LlamaFactory source directory is missing: $FACTORY_ROOT/src" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

service_is_ready() {
  local port="$1"
  curl --silent --show-error --fail --max-time 5 \
    "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

start_service() {
  local name="$1"
  local port="$2"
  local devices="$3"
  local model_alias="$4"
  local config="$5"

  if service_is_ready "$port"; then
    echo "$name already healthy on port $port"
    return
  fi
  if [[ ! -f "$config" ]]; then
    echo "missing model service config: $config" >&2
    exit 2
  fi

  echo "starting $name on port $port (CUDA devices: $devices)"
  (
    cd "$FACTORY_ROOT"
    CUDA_VISIBLE_DEVICES="$devices" \
    API_HOST=127.0.0.1 \
    API_PORT="$port" \
    API_MODEL_NAME="$model_alias" \
    API_VERBOSE=0 \
    VLLM_USE_V1=0 \
    VLLM_ATTENTION_BACKEND=XFORMERS \
    PYTHONPATH="$FACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      nohup "$FACTORY_CLI" api "$config" \
      >"$LOG_DIR/${name}.log" 2>&1 &
    echo "$!" >"$LOG_DIR/${name}.pid"
  )
}

CONFIG_DIR="$PROJECT_ROOT/configs/model_services"
start_service \
  semantic 18002 "${SEMANTIC_CUDA_DEVICES:-1,2}" semantic \
  "$CONFIG_DIR/llama3_1_8b_lora_semantic_goldfirst_vllm.yaml"
start_service \
  operator 18004 "${OPERATOR_CUDA_DEVICES:-3}" operator \
  "$CONFIG_DIR/llama3_1_8b_lora_operator_goldfirst_vllm.yaml"
start_service \
  compose 18003 "${COMPOSE_CUDA_DEVICES:-4,5,6,7}" compose \
  "$CONFIG_DIR/llama3_1_8b_lora_compose_goldfirst_vllm.yaml"
start_service \
  selector 18005 "${SELECTOR_CUDA_DEVICES:-8}" llama3.1-8b-instruct \
  "$CONFIG_DIR/llama3_1_8b_graph_selector_vllm.yaml"

deadline=$((SECONDS + START_TIMEOUT))
for spec in "semantic:18002" "compose:18003" "operator:18004" "selector:18005"; do
  name="${spec%%:*}"
  port="${spec##*:}"
  while ! service_is_ready "$port"; do
    if ((SECONDS >= deadline)); then
      echo "$name did not become healthy on port $port" >&2
      echo "inspect $LOG_DIR/${name}.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "$name healthy on port $port"
done

echo "all CWQ model services are healthy"
