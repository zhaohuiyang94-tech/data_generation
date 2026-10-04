from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .openai_client import DEFAULT_RESPONSES_URL, KaedeResponsesClient, ResponsesClient
from .pipeline import (
    Flywheel,
    FlywheelConfig,
    load_source_pairs,
    materialize,
    source_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOME_ROOT = PROJECT_ROOT.parent
DEFAULT_SEMANTIC = PROJECT_ROOT / "data/webqsp/semantic_path_train.json"
DEFAULT_COMPOSE = PROJECT_ROOT / "data/webqsp/compose_train.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/generated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt54-flywheel",
        description="Rewrite and verify KAeDe-VQG semantic/compose SFT data with GPT-5.4.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="pair source rows without calling the API")
    _add_source_arguments(preflight)

    generate = subparsers.add_parser("generate", help="run the verified generation loop")
    _add_source_arguments(generate)
    generate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--split-name", default="train")
    generate.add_argument(
        "--model",
        default=os.environ.get("WEBQSP_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-5.4",
    )
    generate.add_argument(
        "--responses-url",
        "--base-url",
        dest="responses_url",
        default=(
            os.environ.get("WEBQSP_RESPONSES_URL")
            or os.environ.get("OPENAI_RESPONSES_URL")
            or DEFAULT_RESPONSES_URL
        ),
        help=(
            "full Responses API endpoint; --base-url remains as a compatibility alias "
            "and appends /responses when needed"
        ),
    )
    generate.add_argument(
        "--reasoning-effort",
        choices=_EFFORTS,
        default=(
            os.environ.get("WEBQSP_REASONING_EFFORT")
            or os.environ.get("OPENAI_REASONING_EFFORT")
            or "low"
        ),
    )
    generate.add_argument("--verifier-reasoning-effort", choices=_EFFORTS, default="high")
    generate.add_argument("--max-attempts", type=int, default=5)
    generate.add_argument("--network-retries", type=int, default=5)
    generate.add_argument("--timeout", type=float, default=180.0)
    generate.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help=(
            "optional output cap; 0 omits it for Responses and uses the kaede_vqg "
            "Chat default of 2048"
        ),
    )
    generate.add_argument(
        "--json-mode",
        choices=("auto", "schema", "prompt"),
        default="auto",
        help="structured-output mode; prompt avoids gateways that fail on text.format.json_schema",
    )
    generate.add_argument(
        "--api-mode",
        choices=("kaede", "responses", "chat"),
        default="kaede",
        help="API transport; kaede directly reuses webqsp_mas.llm_client",
    )
    generate.add_argument(
        "--pipeline-profile",
        choices=("auto", "standard", "lean", "micro"),
        default="auto",
        help=(
            "auto uses lean for the KAEDE proxy, micro for chat, and standard for "
            "direct Responses"
        ),
    )
    generate.add_argument("--checkpoint-every", type=int, default=20)
    generate.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-row semantic/compose/verifier progress (disable with --no-progress)",
    )
    generate.add_argument("--start", type=int, default=0)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument(
        "--operator-only",
        action="store_true",
        help="process rows with semantic AND or an executable source operator",
    )
    generate.add_argument(
        "--external-validator-command",
        default="",
        help="optional command reading a candidate JSON object on stdin",
    )
    generate.add_argument("--external-validator-timeout", type=float, default=120.0)

    render = subparsers.add_parser("materialize", help="rebuild JSON arrays from accepted.jsonl")
    render.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    render.add_argument("--split-name", default="train")
    return parser


_EFFORTS = ("none", "low", "medium", "high", "xhigh")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--semantic", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--compose", type=Path, default=DEFAULT_COMPOSE)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        manifest = materialize(args.output, split_name=args.split_name)
        print(json.dumps(manifest.get("counts", {}), ensure_ascii=False, sort_keys=True))
        return

    pairs = load_source_pairs(args.semantic, args.compose)
    summary = source_summary(pairs)
    if args.command == "preflight":
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.start < 0 or args.limit < 0:
        raise SystemExit("--start and --limit must be non-negative")
    selected = pairs
    if args.operator_only:
        selected = [pair for pair in selected if pair.has_semantic_operator]
    selected = selected[args.start :]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no source rows selected")

    api_key = os.environ.get("WEBQSP_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("WEBQSP_API_KEY or OPENAI_API_KEY is required for generate")
    if args.api_mode == "kaede":
        client = KaedeResponsesClient(
            api_key=api_key,
            model=args.model,
            responses_url=args.responses_url,
            timeout=args.timeout,
            max_network_retries=args.network_retries,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        client = ResponsesClient(
            api_key=api_key,
            model=args.model,
            responses_url=args.responses_url,
            timeout=args.timeout,
            max_network_retries=args.network_retries,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            json_mode=args.json_mode,
            api_mode=args.api_mode,
        )
    pipeline_profile = args.pipeline_profile
    if pipeline_profile == "auto":
        pipeline_profile = {
            "kaede": "lean",
            "chat": "micro",
            "responses": "standard",
        }[args.api_mode]
    flywheel = Flywheel(
        client=client,
        config=FlywheelConfig(
            output_dir=args.output,
            split_name=args.split_name,
            max_attempts=args.max_attempts,
            checkpoint_every=args.checkpoint_every,
            verifier_reasoning_effort=args.verifier_reasoning_effort,
            external_validator_command=args.external_validator_command,
            external_validator_timeout=args.external_validator_timeout,
            pipeline_profile=pipeline_profile,
            show_progress=args.progress,
        ),
        semantic_source=args.semantic,
        compose_source=args.compose,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "api_mode": client.api_mode,
                "api_url": client.api_url,
                "json_mode": client.json_mode,
                "pipeline_profile": pipeline_profile,
                "responses_url": client.responses_url,
                "selected_rows": len(selected),
                "source_summary": summary,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    result = flywheel.run(selected)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
