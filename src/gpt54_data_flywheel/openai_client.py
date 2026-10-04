from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib import error, request


DEFAULT_RESPONSES_URL = "https://ai.gs88.shop/v1/responses"


class OpenAIRequestError(RuntimeError):
    """Raised when a Responses API request cannot be completed safely."""


@dataclass(frozen=True)
class Generation:
    value: dict[str, Any]
    response_id: str
    usage: dict[str, Any]
    transport_mode: str = "json_schema"


class ResponsesClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        responses_url: str = DEFAULT_RESPONSES_URL,
        timeout: float = 180.0,
        max_network_retries: int = 5,
        reasoning_effort: str = "low",
        max_output_tokens: int = 0,
        json_mode: str = "auto",
        api_mode: str = "responses",
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is empty")
        self.api_key = api_key.strip()
        self.model = model
        self.responses_url = _normalize_responses_url(responses_url)
        self.timeout = timeout
        self.max_network_retries = max_network_retries
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if json_mode not in {"auto", "schema", "prompt"}:
            raise ValueError("json_mode must be auto, schema, or prompt")
        self.json_mode = json_mode
        if api_mode not in {"responses", "chat"}:
            raise ValueError("api_mode must be responses or chat")
        self.api_mode = api_mode
        self.api_url = (
            _chat_completions_url(self.responses_url)
            if api_mode == "chat"
            else self.responses_url
        )

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
        reasoning_effort: str | None = None,
        gateway_instructions: str | None = None,
        output_tokens: int | None = None,
    ) -> Generation:
        if self.api_mode == "chat":
            return self._generate_chat_json(
                schema_name=schema_name,
                schema=schema,
                instructions=instructions,
                payload=payload,
                gateway_instructions=gateway_instructions,
                output_tokens=output_tokens,
            )

        effort = reasoning_effort or self.reasoning_effort
        input_messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": instructions}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                            "text": json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                    }
                ],
            },
        ]
        schema_body: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        attempts: list[tuple[str, dict[str, Any]]] = []
        if self.json_mode in {"auto", "schema"}:
            if effort != "none":
                with_reasoning = dict(schema_body)
                with_reasoning["reasoning"] = {"effort": effort}
                attempts.append(("json_schema+reasoning", with_reasoning))
            attempts.append(("json_schema", schema_body))
        if self.json_mode in {"auto", "prompt"}:
            prompt_instructions = gateway_instructions or _prompt_json_instructions(
                instructions, schema_name=schema_name, schema=schema
            )
            prompt_input = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            prompt_body: dict[str, Any] = {
                "model": self.model,
                "instructions": prompt_instructions,
                "input": prompt_input,
            }
            if effort != "none":
                prompt_with_reasoning = dict(prompt_body)
                prompt_with_reasoning["reasoning"] = {"effort": effort}
                attempts.append(("prompt_json+reasoning", prompt_with_reasoning))
            attempts.append(("prompt_json", prompt_body))

            inline_body: dict[str, Any] = {
                "model": self.model,
                "input": f"{prompt_instructions}\n\nInput JSON:\n{prompt_input}",
            }
            if effort != "none":
                inline_with_reasoning = dict(inline_body)
                inline_with_reasoning["reasoning"] = {"effort": effort}
                attempts.append(("inline_prompt_json+reasoning", inline_with_reasoning))
            attempts.append(("inline_prompt_json", inline_body))
        output_cap = self.max_output_tokens if output_tokens is None else output_tokens
        if output_cap > 0:
            for _, candidate in attempts:
                candidate["max_output_tokens"] = output_cap

        errors: list[str] = []
        for mode, candidate in attempts:
            try:
                raw = self._post_json(candidate)
                text = _extract_output_text(raw)
                value = _decode_json_object(text)
                return Generation(
                    value=value,
                    response_id=str(raw.get("id", "")),
                    usage=raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {},
                    transport_mode=mode,
                )
            except (OpenAIRequestError, json.JSONDecodeError) as exc:
                errors.append(f"{mode}: {exc}")
        raise OpenAIRequestError("; ".join(errors))

    def _generate_chat_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
        gateway_instructions: str | None,
        output_tokens: int | None,
    ) -> Generation:
        prompt_instructions = gateway_instructions or _prompt_json_instructions(
            instructions, schema_name=schema_name, schema=schema
        )
        prompt = (
            f"{prompt_instructions}\n\nInput JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": (
                output_tokens
                if output_tokens is not None
                else (self.max_output_tokens if self.max_output_tokens > 0 else 2048)
            ),
        }
        raw = self._post_json(body)
        text = _extract_chat_output_text(raw)
        return Generation(
            value=_decode_json_object(text),
            response_id=str(raw.get("id", "")),
            usage=raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {},
            transport_mode="chat_prompt_json",
        )

    def _post_json(self, body: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for network_attempt in range(self.max_network_retries + 1):
            api_request = request.Request(
                self.api_url,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(api_request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                if not isinstance(raw, dict):
                    raise OpenAIRequestError("Responses API returned a non-object payload")
                if raw.get("error"):
                    raise OpenAIRequestError(
                        f"Responses API error: {json.dumps(raw['error'], ensure_ascii=False)}"
                    )
                return raw
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code in {401, 403, 404}:
                    raise OpenAIRequestError(
                        f"OpenAI request failed with HTTP {exc.code}: {detail}"
                    ) from exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise OpenAIRequestError(
                        f"OpenAI request failed with HTTP {exc.code}: {detail}"
                    ) from exc
                last_error = f"HTTP {exc.code}: {detail}"
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)

            if network_attempt >= self.max_network_retries:
                break
            delay = min(30.0, 2.0**network_attempt) + random.uniform(0.0, 0.5)
            time.sleep(delay)

        raise OpenAIRequestError(
            f"OpenAI request failed after {self.max_network_retries + 1} attempts: {last_error}"
        )


def _normalize_responses_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return DEFAULT_RESPONSES_URL
    if url.endswith("/responses"):
        return url
    return f"{url}/responses"


def _chat_completions_url(responses_url: str) -> str:
    if responses_url.endswith("/responses"):
        return f"{responses_url[:-len('/responses')]}/chat/completions"
    return f"{responses_url.rstrip('/')}/chat/completions"


def _prompt_json_instructions(
    instructions: str,
    *,
    schema_name: str,
    schema: dict[str, Any],
) -> str:
    return (
        f"{instructions.rstrip()}\n\n"
        "Return only one valid JSON object. Do not use Markdown fences or add commentary. "
        f"The object must satisfy this {schema_name} JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}"
    )


def _decode_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise OpenAIRequestError("model output was not a JSON object")
    return value


def _extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    texts: list[str] = []
    refusals: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
            elif content.get("type") == "refusal":
                refusals.append(str(content.get("refusal", "model refusal")))
    if texts:
        return "".join(texts)
    if refusals:
        raise OpenAIRequestError(f"model refused the request: {'; '.join(refusals)}")
    raise OpenAIRequestError("Responses API returned no output_text")


def _extract_chat_output_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise OpenAIRequestError("Chat Completions API returned no choices")
    first = choices[0]
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        texts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        if any(texts):
            return "".join(texts)
    raise OpenAIRequestError("Chat Completions API returned no message content")


class KaedeResponsesClient:
    """Thin adapter around the exact Responses client used by kaede_vqg."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        responses_url: str = DEFAULT_RESPONSES_URL,
        timeout: float = 180.0,
        max_network_retries: int = 3,
        reasoning_effort: str = "low",
    ) -> None:
        if not api_key.strip():
            raise ValueError("API key is empty")
        self.api_key = api_key.strip()
        self.model = model
        self.responses_url = _normalize_responses_url(responses_url)
        self.api_url = self.responses_url
        self.api_mode = "kaede"
        self.json_mode = "schema"
        self.timeout = timeout
        self.max_network_retries = max(0, max_network_retries)
        self.reasoning_effort = reasoning_effort
        self._responses_config, self._responses_client = _kaede_client_types()

    def generate_json(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
        reasoning_effort: str | None = None,
        gateway_instructions: str | None = None,
        output_tokens: int | None = None,
    ) -> Generation:
        del output_tokens
        effective_instructions = gateway_instructions or instructions
        effort = reasoning_effort or self.reasoning_effort
        errors: list[str] = []
        for attempt in range(self.max_network_retries + 1):
            config = self._responses_config.from_args(
                responses_url=self.responses_url,
                api_key=self.api_key,
                model=self.model,
                reasoning_effort=effort,
                timeout=max(1, int(self.timeout)),
                max_retries=1,
                retry_delay=0.1,
                provider="openai",
            )
            client = self._responses_client(config)
            try:
                value, raw = client.create_structured_json(
                    system_prompt=effective_instructions,
                    user_prompt=json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    schema_name=schema_name,
                    schema=schema,
                )
                return Generation(
                    value=value,
                    response_id=str(raw.get("id", "")),
                    usage=raw.get("usage", {}) if isinstance(raw.get("usage"), dict) else {},
                    transport_mode="kaede_responses",
                )
            except Exception as exc:  # noqa: BLE001 - preserve KAEDE integration errors
                errors.append(f"attempt {attempt + 1}: {exc}")
                if attempt < self.max_network_retries:
                    time.sleep(min(30.0, 2.0**attempt))
        raise OpenAIRequestError("KAEDE Responses request failed: " + "; ".join(errors))


def _kaede_client_types():
    source_root = Path(
        os.environ.get("WEBQSP_MAS_SRC", "/home/yangzhaohui/webqsp_mas/src")
    ).resolve()
    if not (source_root / "webqsp_mas/llm_client.py").is_file():
        raise RuntimeError(
            f"KAEDE Responses client not found under {source_root}; set WEBQSP_MAS_SRC"
        )
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from webqsp_mas.llm_client import ResponsesClient as OriginalResponsesClient
        from webqsp_mas.llm_client import ResponsesConfig
    except Exception as exc:  # noqa: BLE001 - provide a useful local integration error
        raise RuntimeError(f"Could not import KAEDE Responses client from {source_root}: {exc}") from exc
    return ResponsesConfig, OriginalResponsesClient
