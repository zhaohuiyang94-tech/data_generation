from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import math
import re
import time
from typing import Any, Protocol
from urllib import error, parse, request

from .config import ChatConfig, EmbeddingConfig, FreebaseConfig
from .contracts import EntityCandidate


_LABEL_BATCH_SIZE = 128


def _open_url(req: request.Request, *, timeout: float):
    """Contact loopback services directly, independently of shell proxy settings."""
    hostname = (parse.urlsplit(req.full_url).hostname or "").rstrip(".").casefold()
    loopback = hostname == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    if loopback:
        return request.build_opener(request.ProxyHandler({})).open(req, timeout=timeout)
    return request.urlopen(req, timeout=timeout)


class SparqlRequestError(RuntimeError):
    """A Freebase failure with enough request context for persisted traces."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        method: str,
        query: str,
        elapsed_seconds: float,
        status_code: int | None = None,
        response_detail: str = "",
        url_length: int | None = None,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.method = method
        self.query = query
        self.elapsed_seconds = elapsed_seconds
        self.status_code = status_code
        self.response_detail = response_detail
        self.url_length = url_length

    def trace_details(self) -> dict[str, Any]:
        query_limit = 20_000
        query = self.query
        return {
            "type": type(self).__name__,
            "message": str(self),
            "request": {
                "endpoint": self.endpoint,
                "method": self.method,
                "query": query[:query_limit],
                "query_truncated": len(query) > query_limit,
                "query_length": len(query),
                "url_length": self.url_length,
                "elapsed_seconds": round(self.elapsed_seconds, 6),
            },
            "response": {
                "status_code": self.status_code,
                "detail": self.response_detail,
            },
        }


class ChatClient(Protocol):
    def generate_json(
        self,
        *,
        instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any] | None = None,
        count: int = 1,
    ) -> list[dict[str, Any]]: ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]: ...


class KnowledgeGraph(Protocol):
    def search_entities(self, surface: str, *, limit: int) -> list[EntityCandidate]: ...

    def expand_hop(
        self,
        node_id: str | dict[str, Any],
        direction: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def first_hop(self, entity_id: str, *, limit: int) -> list[dict[str, Any]]: ...

    def second_hop(
        self,
        entity_id: str,
        first_hop: dict[str, Any],
        *,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def path_hops(
        self,
        entity_id: str,
        directions: list[str],
        *,
        limit: int,
        relation_candidates: list[list[str]] | None = None,
    ) -> list[dict[str, Any]]: ...

    def execute(self, sparql: str) -> list[dict[str, str]]: ...

    def labels(self, ids: list[str]) -> list[dict[str, str]]: ...


class HttpChatClient:
    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    def generate_json(
        self,
        *,
        instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any] | None = None,
        count: int = 1,
    ) -> list[dict[str, Any]]:
        system = instruction.strip()
        payload_text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if self.config.prompt_mode == "alpaca":
            messages = [
                {
                    "role": "user",
                    "content": "\n".join(part for part in (system, payload_text) if part),
                }
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": payload_text},
            ]
        body: dict[str, Any]
        if self.config.api in {"responses", "response"}:
            response_format = (
                {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema,
                    "strict": True,
                }
                if schema is not None
                else {"type": "json_object"}
            )
            body = {
                "model": self.config.model,
                "input": [
                    {
                        "role": message["role"],
                        "content": [{"type": "input_text", "text": message["content"]}],
                    }
                    for message in messages
                ],
                "max_output_tokens": self.config.max_tokens,
                "text": {"format": response_format},
            }
        else:
            body = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "n": max(1, count),
            }
            if self.config.thinking:
                body["thinking"] = {"type": self.config.thinking}
            if self.config.reasoning_effort:
                body["reasoning_effort"] = self.config.reasoning_effort
            if schema is not None and self.config.schema_mode != "prompt":
                if self.config.schema_mode == "json_schema":
                    body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "structured_output", "schema": schema},
                    }
                elif self.config.schema_mode == "structured_outputs":
                    body["structured_outputs"] = {"json": schema}
                else:
                    body["guided_json"] = schema
            elif self.config.schema_mode != "prompt":
                body["response_format"] = {"type": "json_object"}
        if self.config.parallel_tool_calls is not None:
            body["parallel_tool_calls"] = self.config.parallel_tool_calls
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url
        response_mode = self.config.api in {"responses", "response"}
        endpoint = "/responses" if response_mode else "/chat/completions"
        if not url.endswith(endpoint) and not url.endswith("/chat/completions") and not url.endswith("/responses"):
            url = f"{url}{endpoint}"
        raw = _post_json(
            url,
            body,
            headers=headers,
            timeout=self.config.timeout,
            retries=self.config.retries,
            retry_delay=self.config.retry_delay,
        )
        outputs: list[dict[str, Any]] = []
        if response_mode:
            text = raw.get("output_text", "") if isinstance(raw, dict) else ""
            if text:
                outputs.append(_parse_json_object(str(text)))
            if not outputs:
                for item in raw.get("output", []) if isinstance(raw, dict) else []:
                    for content in item.get("content", []) if isinstance(item, dict) else []:
                        text = content.get("text", "") if isinstance(content, dict) else ""
                        if text:
                            outputs.append(_parse_json_object(str(text)))
        else:
            choices = raw.get("choices", []) if isinstance(raw, dict) else []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message", {})
                text = message.get("content", "") if isinstance(message, dict) else ""
                if text:
                    outputs.append(_parse_json_object(str(text)))
        if not outputs:
            raise RuntimeError(f"chat model returned no JSON choices: {raw!r}")
        return outputs


class HttpEmbeddingClient:
    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        cache_enabled: bool = True,
    ) -> None:
        self.config = config
        self.cache_enabled = bool(cache_enabled)
        self._cache: dict[tuple[str, str], list[float]] = {}

    def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if not texts:
            return []
        result: list[list[float] | None] = [None] * len(texts)
        missing: list[str] = []
        missing_indexes: list[int] = []
        for index, text in enumerate(texts):
            key = (input_type, text)
            if self.cache_enabled and key in self._cache:
                result[index] = self._cache[key]
            else:
                missing.append(text)
                missing_indexes.append(index)
        if missing:
            body = {
                "model": self.config.model,
                "input": missing,
                "input_type": input_type,
            }
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            raw = _post_json(self.config.url, body, headers=headers, timeout=self.config.timeout)
            data = raw.get("data", []) if isinstance(raw, dict) else []
            vectors = [item.get("embedding") for item in sorted(data, key=lambda x: int(x.get("index", 0))) if isinstance(item, dict)]
            if len(vectors) != len(missing) or not all(isinstance(v, list) for v in vectors):
                raise RuntimeError(f"embedding response has wrong shape: {raw!r}")
            for index, text, vector in zip(missing_indexes, missing, vectors):
                normalized = [float(value) for value in vector]
                if self.cache_enabled:
                    self._cache[(input_type, text)] = normalized
                result[index] = normalized
        return [vector or [] for vector in result]


class EmbeddingRanker:
    def __init__(self, client: EmbeddingClient) -> None:
        self.client = client

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        query_vector = self.client.embed([query], input_type="query")[0]
        document_vectors = self.client.embed(candidates, input_type="document")
        return [_cosine(query_vector, vector) for vector in document_vectors]


@dataclass(slots=True)
class SparqlKnowledgeGraph:
    config: FreebaseConfig
    _cvt_cache: dict[tuple[str, str], bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def search_entities(self, surface: str, *, limit: int) -> list[EntityCandidate]:
        escaped = _sparql_string(surface)
        exact_query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?label WHERE {{
  VALUES ?label {{ "{escaped}"@en "{escaped}" }}
  {{ ?entity rdfs:label ?label . }} UNION {{ ?entity ns:common.topic.alias ?label . }}
  FILTER(STRSTARTS(STR(?entity), STR(ns:)))
}}
LIMIT {max(100, limit * 20)}
""".strip()
        substring_query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?label WHERE {{
  {{ ?entity rdfs:label ?label . }} UNION {{ ?entity ns:common.topic.alias ?label . }}
  FILTER(LANG(?label) = "en" || LANG(?label) = "")
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{escaped}")))
  FILTER(STRSTARTS(STR(?entity), STR(ns:)))
}}
LIMIT {max(50, limit * 20)}
""".strip()
        candidates: list[EntityCandidate] = []
        _extend_entity_candidates(candidates, self.execute(exact_query), surface)
        if len({item.entity_id for item in candidates}) < max(1, limit):
            _extend_entity_candidates(candidates, self.execute(substring_query), surface)
        candidates.sort(
            key=lambda item: (-item.score, len(item.entity_id), item.label, item.entity_id)
        )
        unique: dict[str, EntityCandidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.entity_id, candidate)
        return list(unique.values())[: max(1, limit)]

    def first_hop(self, entity_id: str, *, limit: int) -> list[dict[str, Any]]:
        _uri(entity_id)
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation ?direction WHERE {{
  {{ ns:{entity_id} ?relation ?node . BIND("forward" AS ?direction) }}
  UNION
  {{ ?node ?relation ns:{entity_id} . BIND("backward" AS ?direction) }}
  FILTER(STRSTARTS(STR(?relation), STR(ns:)))
  FILTER(?relation != <http://www.w3.org/2000/01/rdf-schema#label>)
}}
LIMIT {max(1, limit)}
""".strip()
        return self._relation_rows(query)

    def expand_hop(
        self,
        node_id: str | dict[str, Any],
        direction: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return real neighboring nodes for one directed graph hop.

        ``first_hop``/``second_hop`` predate recursive expansion and return
        relation names only.  A stepwise search must also know the endpoint
        of each edge so that the next hop starts from the correct node.  The
        endpoint may be a literal on the final hop; intermediate callers
        filter such rows before asking for another expansion.
        """
        term = _normalize_rdf_term(node_id)
        if not term["value"]:
            return []
        clean_direction = str(direction).strip().casefold()
        if clean_direction not in {"forward", "backward"}:
            return []
        # A literal cannot be an RDF subject.  It can still be the object of
        # a backward edge, which is needed by numeric/CVT paths.
        if term["type"] == "literal" and clean_direction == "forward":
            return []
        current_term = _sparql_rdf_term(term)
        if clean_direction == "forward":
            edge_pattern = f"{current_term} ?relation ?neighbor"
        else:
            edge_pattern = f"?neighbor ?relation {current_term}"
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation ?neighbor ?neighbor_kind ?neighbor_datatype ?neighbor_lang WHERE {{
  {edge_pattern} .
  FILTER(STRSTARTS(STR(?relation), STR(ns:)))
  FILTER(?relation != <http://www.w3.org/2000/01/rdf-schema#label>)
  FILTER(?relation NOT IN (
    ns:type.object.name,
    ns:common.topic.alias,
    ns:common.topic.description
  ))
  BIND(IF(isIRI(?neighbor), "uri", IF(isBLANK(?neighbor), "bnode", "literal")) AS ?neighbor_kind)
  BIND(STR(DATATYPE(?neighbor)) AS ?neighbor_datatype)
  BIND(LANG(?neighbor) AS ?neighbor_lang)
}}
ORDER BY ?relation ?neighbor
LIMIT {max(1, int(limit))}
""".strip()
        rows = self.execute(query)
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            relation_id = _compact_uri(row.get("relation", ""))
            next_node = _compact_uri(row.get("neighbor", ""))
            if not _is_relation_id(relation_id) or not next_node:
                continue
            raw_kind = str(row.get("neighbor_kind", "")).strip()
            next_type = _normalize_term_type(raw_kind)
            if not raw_kind:
                next_type = "uri" if _is_entity_id(next_node) else "literal"
            key = (relation_id, clean_direction, next_node)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "current_node_id": term["value"],
                    "relation_id": relation_id,
                    "direction": clean_direction,
                    "next_node_id": next_node,
                    "next_node_type": next_type,
                    "next_node_datatype": str(
                        row.get("neighbor_datatype", "")
                    ),
                    "next_node_lang": str(row.get("neighbor_lang", "")),
                    "next_node_is_entity": (
                        str(row.get("neighbor_kind", "")).casefold() in {"uri", "bnode"}
                        or (
                            "neighbor_kind" not in row
                            and _is_entity_id(next_node)
                        )
                    ),
                }
            )
        return result

    def second_hop(
        self,
        entity_id: str,
        first_hop: dict[str, Any],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        _uri(entity_id)
        relation = str(first_hop.get("relation_id", ""))
        direction = str(first_hop.get("direction", "forward"))
        if not _is_relation_id(relation):
            return []
        predicate = _uri(relation)
        middle = f"?middle" if direction == "forward" else f"?middle"
        first_triple = f"ns:{entity_id} <{predicate}> {middle}" if direction == "forward" else f"{middle} <{predicate}> ns:{entity_id}"
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?second_relation ?second_direction WHERE {{
  {{ {first_triple} . ?middle ?second_relation ?value . BIND("forward" AS ?second_direction) }}
  UNION
  {{ {first_triple} . ?value ?second_relation ?middle . BIND("backward" AS ?second_direction) }}
  FILTER(STRSTARTS(STR(?second_relation), STR(ns:)))
}}
LIMIT {max(1, limit)}
""".strip()
        rows = self.execute(query)
        output: list[dict[str, Any]] = []
        for row in rows:
            second = _compact_uri(row.get("second_relation", ""))
            if _is_relation_id(second):
                output.append(
                    {
                        "first_relation_id": relation,
                        "first_direction": direction,
                        "second_relation_id": second,
                        "second_direction": str(row.get("second_direction", "forward")),
                    }
                )
        return output

    def path_hops(
        self,
        entity_id: str,
        directions: list[str],
        *,
        limit: int,
        relation_candidates: list[list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Enumerate bounded relation-id sequences for any positive path depth.

        The existing first/second-hop calls intentionally return only relation
        candidates. One bounded SPARQL query keeps intermediate nodes
        existential while returning the ordered relation sequence needed by
        the semantic grounder. When
        ``relation_candidates`` is supplied, each relation variable is
        constrained with a local ``VALUES`` clause before the endpoint performs
        the multi-hop join.
        """
        _uri(entity_id)
        clean_directions = [str(value).strip().casefold() for value in directions]
        if not clean_directions or any(
            value not in {"forward", "backward"} for value in clean_directions
        ):
            return []
        bounded_limit = max(1, min(int(limit), 10000))
        nodes = [f"?n{index}" for index in range(len(clean_directions) + 1)]
        relations = [f"?r{index}" for index in range(len(clean_directions))]
        patterns = [f"VALUES {nodes[0]} {{ ns:{entity_id} }}"]
        candidate_sets: list[list[str]] | None = None
        if relation_candidates is not None:
            if len(relation_candidates) != len(relations):
                raise ValueError("relation_candidates must match path depth")
            candidate_sets = []
            for values in relation_candidates:
                clean_values = list(
                    dict.fromkeys(
                        str(value).strip()
                        for value in values
                        if _is_relation_id(str(value).strip())
                    )
                )
                if not clean_values:
                    raise ValueError("relation_candidates cannot contain an empty hop")
                candidate_sets.append(clean_values)
        for index, direction in enumerate(clean_directions):
            if candidate_sets is not None:
                values = " ".join(f"ns:{value}" for value in candidate_sets[index])
                patterns.append(f"VALUES {relations[index]} {{ {values} }}")
            if direction == "forward":
                patterns.append(f"{nodes[index]} {relations[index]} {nodes[index + 1]} .")
            else:
                patterns.append(f"{nodes[index + 1]} {relations[index]} {nodes[index]} .")
            patterns.append(
                f"FILTER(STRSTARTS(STR({relations[index]}), STR(ns:)))"
            )
            patterns.append(
                f"FILTER({relations[index]} != <http://www.w3.org/2000/01/rdf-schema#label>)"
            )
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT {' '.join(relations)} WHERE {{
  {' '.join(patterns)}
}}
LIMIT {bounded_limit}
""".strip()
        rows = self.execute(query)
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for row in rows:
            relation_ids: list[str] = []
            for relation in relations:
                value = row.get(relation.lstrip("?"), row.get(relation, ""))
                compact = _compact_uri(value)
                if not _is_relation_id(compact):
                    relation_ids = []
                    break
                relation_ids.append(compact)
            key = tuple(relation_ids)
            if not relation_ids or key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "relation_ids": relation_ids,
                    "directions": list(clean_directions),
                }
            )
        return result

    def execute(self, sparql: str) -> list[dict[str, str]]:
        return _bindings(self._select(sparql))

    def labels(self, ids: list[str]) -> list[dict[str, str]]:
        clean = list(dict.fromkeys(value for value in ids if _is_entity_id(value)))
        if not clean:
            return []
        rows: list[dict[str, str]] = []
        for offset in range(0, len(clean), _LABEL_BATCH_SIZE):
            batch = clean[offset : offset + _LABEL_BATCH_SIZE]
            values = " ".join(f"ns:{value}" for value in batch)
            query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?label WHERE {{
  VALUES ?entity {{ {values} }}
  {{
    ?entity ns:type.object.name ?name .
    FILTER(LANGMATCHES(LANG(?name), "EN"))
    BIND(STR(?name) AS ?label)
  }}
  UNION
  {{
    ?entity rdfs:label ?rdfs_label .
    FILTER(LANGMATCHES(LANG(?rdfs_label), "EN"))
    BIND(STR(?rdfs_label) AS ?label)
  }}
}}
LIMIT {max(20, len(batch) * 3)}
""".strip()
            rows.extend(self.execute(query))
        labels_by_id = {entity_id: entity_id for entity_id in clean}
        for row in rows:
            entity_id = _compact_uri(row.get("entity", ""))
            label = str(row.get("label", "")).strip()
            if entity_id in labels_by_id and label:
                labels_by_id[entity_id] = label
        return [
            {"id": entity_id, "label": labels_by_id[entity_id]}
            for entity_id in clean
        ]

    def _relation_rows(self, query: str) -> list[dict[str, Any]]:
        rows = self.execute(query)
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            relation_id = _compact_uri(row.get("relation", ""))
            if not _is_relation_id(relation_id):
                continue
            direction = str(row.get("direction", "forward"))
            key = (relation_id, direction)
            reaches_cvt = str(row.get("reaches_cvt", "false")).casefold() == "true"
            current = merged.get(key)
            if current is None:
                merged[key] = {
                    "relation_id": relation_id,
                    "direction": direction,
                    "reaches_cvt": reaches_cvt,
                }
            else:
                current["reaches_cvt"] = bool(current["reaches_cvt"]) or reaches_cvt
        return list(merged.values())

    def annotate_cvt_metadata(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unresolved = [
            row
            for row in rows
            if (
                str(row.get("relation_id", "")),
                str(row.get("direction", "forward")),
            )
            not in self._cvt_cache
        ]
        metadata_failures: set[tuple[str, str]] = set()
        for offset in range(0, len(unresolved), 32):
            batch = unresolved[offset : offset + 32]
            batch_keys = {
                (
                    str(row.get("relation_id", "")),
                    str(row.get("direction", "forward")),
                )
                for row in batch
            }
            try:
                detected = self._query_cvt_metadata(batch)
            except RuntimeError:
                metadata_failures.update(
                    (
                        str(row.get("relation_id", "")),
                        str(row.get("direction", "forward")),
                    )
                    for row in unresolved[offset:]
                )
                break
            for key in batch_keys:
                self._cvt_cache[key] = key in detected

        output: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            key = (
                str(row.get("relation_id", "")),
                str(row.get("direction", "forward")),
            )
            value["reaches_cvt"] = self._cvt_cache.get(key, False)
            value["cvt_detection_status"] = (
                "metadata_query_failed"
                if key in metadata_failures
                else "resolved"
            )
            output.append(value)
        return output

    def _query_cvt_metadata(
        self,
        rows: list[dict[str, Any]],
    ) -> set[tuple[str, str]]:
        branches: list[str] = []
        for direction, metadata_relation in (
            ("forward", "type.property.expected_type"),
            ("backward", "type.property.schema"),
        ):
            relation_ids = sorted(
                {
                    str(row.get("relation_id", ""))
                    for row in rows
                    if str(row.get("direction", "forward")) == direction
                    and _is_relation_id(str(row.get("relation_id", "")))
                }
            )
            if not relation_ids:
                continue
            values = " ".join(f"ns:{relation_id}" for relation_id in relation_ids)
            branches.append(
                "\n".join(
                    [
                        "{",
                        f"  VALUES ?relation {{ {values} }}",
                        f"  ?relation ns:{metadata_relation} ?target_type .",
                        f'  BIND("{direction}" AS ?direction)',
                        "}",
                    ]
                )
            )
        if not branches:
            return set()
        union = "\nUNION\n".join(branches)
        query = f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation ?direction WHERE {{
  {union}
  BIND(CONCAT("/", REPLACE(STRAFTER(STR(?target_type), STR(ns:)), "\\\\.", "/")) AS ?target_key)
  ?cvt_type ns:type.object.key ?target_key .
  ?cvt_type ns:freebase.type_hints.mediator 1 .
}}
""".strip()
        return {
            (_compact_uri(row.get("relation", "")), str(row.get("direction", "")))
            for row in self.execute(query)
            if _is_relation_id(_compact_uri(row.get("relation", "")))
        }

    def _select(self, query: str) -> dict[str, Any]:
        headers = {"Accept": "application/sparql-results+json"}
        if self.config.method == "POST":
            return _post_form(
                self.config.endpoint,
                {"query": query},
                headers=headers,
                timeout=self.config.timeout,
            )
        url = f"{self.config.endpoint}?{parse.urlencode({'query': query})}"
        req = request.Request(url, headers=headers, method="GET")
        started = time.monotonic()
        try:
            with _open_url(req, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise SparqlRequestError(
                _sparql_error_message("GET", exc, detail, query, len(url)),
                endpoint=self.config.endpoint,
                method="GET",
                query=query,
                elapsed_seconds=time.monotonic() - started,
                status_code=exc.code,
                response_detail=detail,
                url_length=len(url),
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise SparqlRequestError(
                _sparql_error_message("GET", exc, "", query, len(url)),
                endpoint=self.config.endpoint,
                method="GET",
                query=query,
                elapsed_seconds=time.monotonic() - started,
                url_length=len(url),
            ) from exc


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int = 0,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    transient_statuses = {429, 500, 502, 503, 504}
    max_retries = max(0, int(retries))
    for attempt in range(max_retries + 1):
        req = request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with _open_url(req, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code in transient_statuses and attempt < max_retries:
                time.sleep(_retry_delay(exc, retry_delay, attempt))
                continue
            if len(detail) > 2000:
                detail = f"{detail[:2000]}..."
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"HTTP JSON request failed for {url}: HTTP {exc.code} {exc.reason}{suffix}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            if attempt < max_retries:
                time.sleep(max(0.0, float(retry_delay)) * (2**attempt))
                continue
            raise RuntimeError(f"HTTP JSON request failed for {url}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"HTTP response is not an object: {value!r}")
        return value
    raise RuntimeError(f"HTTP JSON request failed for {url}: retries exhausted")


def _retry_delay(exc: error.HTTPError, base_delay: float, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
    try:
        return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        return max(0.0, float(base_delay)) * (2**attempt)


def _post_form(url: str, values: dict[str, str], *, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = request.Request(url, data=parse.urlencode(values).encode("utf-8"), headers=headers, method="POST")
    query = str(values.get("query", ""))
    started = time.monotonic()
    try:
        with _open_url(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = _http_error_detail(exc)
        raise SparqlRequestError(
            _sparql_error_message("POST", exc, detail, query, None),
            endpoint=url,
            method="POST",
            query=query,
            elapsed_seconds=time.monotonic() - started,
            status_code=exc.code,
            response_detail=detail,
        ) from exc
    except (error.URLError, TimeoutError) as exc:
        raise SparqlRequestError(
            _sparql_error_message("POST", exc, "", query, None),
            endpoint=url,
            method="POST",
            query=query,
            elapsed_seconds=time.monotonic() - started,
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("SPARQL response is not an object")
    return value


def _http_error_detail(exc: error.HTTPError) -> str:
    try:
        detail = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    return f"{detail[:4000]}..." if len(detail) > 4000 else detail


def _sparql_error_message(
    method: str,
    exc: Exception,
    detail: str,
    query: str,
    url_length: int | None,
) -> str:
    size = f"query_length={len(query)}"
    if url_length is not None:
        size = f"{size}, url_length={url_length}"
    suffix = f": {detail}" if detail else ""
    return f"Freebase {method} request failed ({size}): {exc}{suffix}"


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"model response is not JSON: {text[:200]}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _bindings(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("results", {}).get("bindings", []) if isinstance(payload, dict) else []
    rows: list[dict[str, str]] = []
    for binding in raw if isinstance(raw, list) else []:
        if isinstance(binding, dict):
            rows.append({key: str(value.get("value", "")) for key, value in binding.items() if isinstance(value, dict)})
    return rows


def _extend_entity_candidates(
    candidates: list[EntityCandidate], rows: list[dict[str, str]], surface: str
) -> None:
    for row in rows:
        entity_id = _compact_uri(row.get("entity", ""))
        label = str(row.get("label", entity_id))
        if _is_entity_id(entity_id):
            candidates.append(
                EntityCandidate(entity_id, label, _entity_score(surface, label), "sparql")
            )


def _compact_uri(value: Any) -> str:
    text = str(value)
    prefix = "http://rdf.freebase.com/ns/"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _normalize_term_type(value: Any) -> str:
    text = str(value).strip().casefold()
    if text in {"uri", "iri", "entity"}:
        return "uri"
    if text in {"bnode", "blank", "blank_node"}:
        return "bnode"
    return "literal"


def _normalize_rdf_term(value: str | dict[str, Any]) -> dict[str, str]:
    """Normalize a recursive-hop endpoint into a serializable RDF term."""
    if isinstance(value, dict):
        raw_value = value.get("value", value.get("id", ""))
        raw_type = value.get("type", value.get("term_type", "uri"))
        raw_datatype = value.get("datatype", "")
        raw_lang = value.get("lang", "")
    else:
        raw_value = value
        raw_type = "uri"
        raw_datatype = ""
        raw_lang = ""
    term_type = _normalize_term_type(raw_type)
    text = str(raw_value).strip()
    return {
        "value": text,
        "type": term_type,
        "datatype": str(raw_datatype).strip(),
        "lang": str(raw_lang).strip(),
    }


def _sparql_rdf_term(term: dict[str, str]) -> str:
    """Render a validated RDF term for a recursive SPARQL pattern."""
    value = str(term.get("value", "")).strip()
    term_type = _normalize_term_type(term.get("type", "uri"))
    if term_type in {"uri", "bnode"}:
        if term_type == "bnode" and value.startswith("_:"):
            return value
        if value.startswith("http://") or value.startswith("https://"):
            return f"<{value}>"
        return f"<{_uri(value)}>"
    escaped = _sparql_string(value)
    lang = str(term.get("lang", "")).strip()
    if lang:
        return f'"{escaped}"@{lang}'
    datatype = str(term.get("datatype", "")).strip()
    if datatype:
        if not datatype.startswith("http://") and not datatype.startswith("https://"):
            datatype = f"http://www.w3.org/2001/XMLSchema#{datatype}"
        return f'"{escaped}"^^<{datatype}>'
    return f'"{escaped}"'


def _uri(value: str) -> str:
    if not _is_freebase_token(value):
        raise ValueError(f"unsafe Freebase token: {value}")
    return f"http://rdf.freebase.com/ns/{value}"


def _is_freebase_token(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_./-]+", value))


def _is_entity_id(value: str) -> bool:
    return bool(re.fullmatch(r"m\.[A-Za-z0-9_]+|g\.[A-Za-z0-9_]+", value))


def _is_relation_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", value)) and not _is_entity_id(value)


def _sparql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _entity_score(surface: str, label: str) -> float:
    left = " ".join(surface.casefold().split())
    right = " ".join(label.casefold().split())
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.8
    return 0.4


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0
