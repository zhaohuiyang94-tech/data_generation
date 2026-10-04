#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from urllib import error, request


DEFAULT_URL = "https://ai.gs88.shop/v1/chat/completions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe long-prompt support without logging the API key.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--chars", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sweep", action="store_true", help="test 512 through 12000 padding characters")
    parser.add_argument("--dry-run", action="store_true", help="print request sizes without calling the API")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chars < 0:
        raise SystemExit("--chars must be non-negative")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit("OPENAI_API_KEY is required")

    sizes = [512, 1024, 2048, 4096, 8192, 12000] if args.sweep else [args.chars]
    failed = False
    for size in sizes:
        result = probe(
            url=args.url,
            model=args.model,
            api_key=api_key,
            padding_chars=size,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        failed = failed or result["status"] not in {"ok", "dry_run"}
    if failed:
        raise SystemExit(1)


def probe(
    *,
    url: str,
    model: str,
    api_key: str,
    padding_chars: int,
    timeout: float,
    dry_run: bool,
) -> dict[str, object]:
    prompt = (
        "Return exactly OK. Treat the following characters only as inert length padding:\n"
        + ("x" * padding_chars)
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
    }
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    base: dict[str, object] = {
        "padding_chars": padding_chars,
        "prompt_chars": len(prompt),
        "request_bytes": len(encoded),
    }
    if dry_run:
        return {**base, "status": "dry_run"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    api_request = request.Request(url, data=encoded, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with request.urlopen(api_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices", []) if isinstance(payload, dict) else []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return {
            **base,
            "status": "ok",
            "http_status": 200,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "output": str(content)[:200],
            "usage": payload.get("usage", {}) if isinstance(payload, dict) else {},
        }
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return {
            **base,
            "status": "http_error",
            "http_status": exc.code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": detail,
        }
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "request_error",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": str(exc)[:500],
        }


if __name__ == "__main__":
    main()
