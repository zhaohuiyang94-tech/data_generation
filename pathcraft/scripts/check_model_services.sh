#!/usr/bin/env bash
set -euo pipefail

failed=0
for spec in "semantic:18002" "compose:18003" "operator:18004" "selector:18005"; do
  name="${spec%%:*}"
  port="${spec##*:}"
  if curl --silent --show-error --fail --max-time 10 \
      "http://127.0.0.1:${port}/v1/models" >/dev/null; then
    echo "$name healthy on port $port"
  else
    echo "$name unavailable on port $port" >&2
    failed=1
  fi
done
exit "$failed"
