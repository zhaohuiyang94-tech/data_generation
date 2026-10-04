"""Small, local Freebase schema index used to constrain path queries."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
from pathlib import Path
import re
from typing import Any


_SEGMENT_RE = re.compile(r"[^A-Za-z0-9]+")
_RELATION_ID_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+")


def _search_token(value: Any) -> str:
    token = _SEGMENT_RE.sub("", str(value).casefold())
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _search_tokens(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value).replace(".", " ").replace("_", " ")
    return tuple(
        token
        for raw in re.findall(r"[A-Za-z0-9]+", text)
        if (token := _search_token(raw))
    )


def _weighted_token_f1(
    query: tuple[str, ...],
    candidate: tuple[str, ...],
    document_frequency: dict[str, int],
    document_count: int,
) -> float:
    query_set, candidate_set = set(query), set(candidate)
    if not query_set or not candidate_set:
        return 0.0

    def weight(token: str) -> float:
        frequency = max(0, int(document_frequency.get(token, 0)))
        return math.log((max(1, document_count) + 1) / (frequency + 1)) + 1.0

    overlap = query_set & candidate_set
    overlap_weight = sum(weight(token) for token in overlap)
    precision = overlap_weight / sum(weight(token) for token in candidate_set)
    recall = overlap_weight / sum(weight(token) for token in query_set)
    return (
        0.0
        if precision + recall == 0.0
        else (2.0 * precision * recall) / (precision + recall)
    )


def _trigram_jaccard(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        compact = _SEGMENT_RE.sub("", value.casefold())
        if len(compact) < 3:
            return {compact} if compact else set()
        return {compact[index : index + 3] for index in range(len(compact) - 2)}

    left_grams, right_grams = grams(left), grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _segment_token(value: Any) -> str:
    """Convert a canonical human-readable schema segment to an ID segment."""
    return _SEGMENT_RE.sub("_", str(value).strip()).strip("_")


def relation_id_from_label(label: Any) -> str:
    """Return the conventional Freebase predicate ID for a label array."""
    if not isinstance(label, (list, tuple)):
        return ""
    segments = [_segment_token(value) for value in label]
    if not segments or any(not value for value in segments):
        return ""
    return ".".join(segments)


def relation_label_key(label: Any) -> str:
    """Normalize a label array for exact schema-index lookup."""
    if isinstance(label, (list, tuple)):
        values = label
    else:
        values = [label]
    return " ".join(" ".join(str(value).split()).casefold() for value in values).strip()


def relation_label_from_id(relation_id: str) -> list[str]:
    """Humanize an ID using the same reversible convention as Grounding."""
    return [segment.replace("_", " ") for segment in str(relation_id).split(".")]


@dataclass(frozen=True, slots=True)
class FreebaseOntology:
    """Indexed Freebase relation schema loaded once per pipeline process.

    ``fb_roles`` is a schema file, not an entity graph.  The index is therefore
    used only to turn canonical Semantic relation labels into a small set of
    predicate IDs.  Canonical labels are also reversibly converted when the
    local snapshot lacks a custom predicate, while the SPARQL endpoint still
    verifies entity-level paths.
    """

    directory: str
    relation_ids: frozenset[str]
    label_to_relation_ids: dict[str, tuple[str, ...]]
    relation_domains: dict[str, str]
    relation_ranges: dict[str, str]
    domain_to_relation_ids: dict[str, tuple[str, ...]]
    type_parents: dict[str, tuple[str, ...]]
    reverse_relation_ids: dict[str, tuple[str, ...]]
    relation_search_tokens: dict[str, tuple[str, ...]]
    relation_leaf_tokens: dict[str, tuple[str, ...]]
    relation_token_document_frequency: dict[str, int]
    relation_token_index: dict[str, tuple[str, ...]]
    role_rows: int
    type_rows: int
    reverse_rows: int
    overlay_rows: int = 0
    inverse_inferred_rows: int = 0

    @classmethod
    def load(cls, directory: str | Path) -> "FreebaseOntology":
        root = Path(directory).expanduser().resolve()
        roles_path = root / "fb_roles"
        if not roles_path.is_file():
            raise FileNotFoundError(f"Freebase ontology fb_roles does not exist: {roles_path}")

        relation_ids: set[str] = set()
        aliases: dict[str, set[str]] = {}
        relation_domains: dict[str, str] = {}
        relation_ranges: dict[str, str] = {}
        domain_relations: dict[str, set[str]] = {}
        role_rows = 0
        overlay_rows = 0

        def add_role(parts: list[str]) -> bool:
            if len(parts) != 3 or not _RELATION_ID_RE.fullmatch(parts[1]):
                return False
            relation_id = parts[1]
            relation_ids.add(relation_id)
            relation_domains[relation_id] = parts[0]
            relation_ranges[relation_id] = parts[2]
            domain_relations.setdefault(parts[0], set()).add(relation_id)
            key = relation_label_key(relation_label_from_id(relation_id))
            if key:
                aliases.setdefault(key, set()).add(relation_id)
            return True

        for raw_line in roles_path.read_text(encoding="utf-8").splitlines():
            role_rows += add_role(raw_line.strip().split())

        # The overlay is generated from endpoint-verified property metadata.
        # It is data, not a question/entity routing list, and remains optional.
        for overlay_name in ("schema_overlay.tsv", "schema_inferred.tsv"):
            overlay_path = root / overlay_name
            if overlay_path.is_file():
                for raw_line in overlay_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    overlay_rows += add_role(line.split())

        reverse_relations: dict[str, set[str]] = {}
        reverse_path = root / "reverse_properties"
        reverse_rows = 0
        if reverse_path.is_file():
            for raw_line in reverse_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().split()
                if len(parts) != 2 or not all(
                    _RELATION_ID_RE.fullmatch(value) for value in parts
                ):
                    continue
                left, right = parts
                reverse_relations.setdefault(left, set()).add(right)
                reverse_relations.setdefault(right, set()).add(left)
                reverse_rows += 1

        # If exactly one side of an inverse pair has schema metadata, the
        # missing side has the swapped domain/range.  Keep it available for
        # type propagation without adding it to domain discovery beams; this
        # prevents a large reverse file from expanding normal candidate sets.
        inverse_inferred_rows = 0
        for relation_id, inverses in sorted(reverse_relations.items()):
            if relation_id in relation_domains and relation_id in relation_ranges:
                continue
            source = next(
                (
                    inverse
                    for inverse in sorted(inverses)
                    if inverse in relation_domains and inverse in relation_ranges
                ),
                "",
            )
            if not source:
                continue
            relation_ids.add(relation_id)
            relation_domains[relation_id] = relation_ranges[source]
            relation_ranges[relation_id] = relation_domains[source]
            key = relation_label_key(relation_label_from_id(relation_id))
            if key:
                aliases.setdefault(key, set()).add(relation_id)
            inverse_inferred_rows += 1

        type_parents: dict[str, set[str]] = {}
        types_path = root / "fb_types"
        if types_path.is_file():
            for raw_line in types_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.strip().rstrip(".").split()
                if len(parts) >= 3 and parts[1] == "meta.subclassOf":
                    type_parents.setdefault(parts[0], set()).add(parts[2])

        def line_count(name: str) -> int:
            path = root / name
            if not path.is_file():
                return 0
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

        relation_search_tokens = {
            relation_id: _search_tokens(relation_id)
            for relation_id in relation_ids
        }
        relation_leaf_tokens = {
            relation_id: _search_tokens(str(relation_id).split(".")[-1])
            for relation_id in relation_ids
        }
        relation_token_document_frequency: dict[str, int] = {}
        relation_token_index_sets: dict[str, set[str]] = {}
        for tokens in relation_search_tokens.values():
            for token in set(tokens):
                relation_token_document_frequency[token] = (
                    relation_token_document_frequency.get(token, 0) + 1
                )
        for relation_id, tokens in relation_search_tokens.items():
            for token in set(tokens):
                relation_token_index_sets.setdefault(token, set()).add(relation_id)

        return cls(
            directory=str(root),
            relation_ids=frozenset(relation_ids),
            label_to_relation_ids={
                key: tuple(sorted(values)) for key, values in aliases.items()
            },
            relation_domains=relation_domains,
            relation_ranges=relation_ranges,
            domain_to_relation_ids={
                key: tuple(sorted(values))
                for key, values in domain_relations.items()
            },
            type_parents={
                key: tuple(sorted(values))
                for key, values in type_parents.items()
            },
            reverse_relation_ids={
                key: tuple(sorted(values))
                for key, values in reverse_relations.items()
            },
            relation_search_tokens=relation_search_tokens,
            relation_leaf_tokens=relation_leaf_tokens,
            relation_token_document_frequency=relation_token_document_frequency,
            relation_token_index={
                token: tuple(sorted(values))
                for token, values in relation_token_index_sets.items()
            },
            role_rows=role_rows,
            type_rows=line_count("fb_types"),
            reverse_rows=reverse_rows,
            overlay_rows=overlay_rows,
            inverse_inferred_rows=inverse_inferred_rows,
        )

    def candidates_for_label(
        self,
        label: Any,
        *,
        limit: int = 8,
    ) -> list[str]:
        """Return local predicate candidates for one Semantic label array."""
        bounded_limit = max(1, int(limit))
        direct = relation_id_from_label(label)
        if direct in self.relation_ids:
            return [direct]
        # Canonical Semantic labels are deliberately reversible to Freebase
        # predicate IDs.  Keep that deterministic candidate even when a local
        # fb_roles snapshot is missing a custom/user predicate; the endpoint
        # will still verify whether the predicate exists in an entity path.
        aliases = list(self.label_to_relation_ids.get(relation_label_key(label), ()))
        candidates = list(dict.fromkeys([direct, *aliases])) if direct else aliases
        return candidates[:bounded_limit]

    def candidates_for_steps(
        self,
        steps: list[dict[str, Any]],
        *,
        limit: int = 8,
    ) -> list[list[str]]:
        return [
            self.candidates_for_label(step.get("relation_label", []), limit=limit)
            for step in steps
        ]

    def similar_candidates_for_label(
        self,
        label: Any,
        *,
        limit: int = 8,
    ) -> list[str]:
        """Return schema-near predicates for a failed exact retrieval lane.

        This is deliberately not used by ``candidates_for_label``: exact
        ontology grounding remains byte-for-byte stable on its success path.
        The caller may use these candidates only after the exact endpoint
        query returned no path.  Ranking is generic and schema-derived; it
        contains no question, entity, or relation whitelist.
        """
        bounded_limit = max(1, int(limit))
        direct = relation_id_from_label(label)
        query_tokens = _search_tokens(label)
        label_parts = list(label) if isinstance(label, (list, tuple)) else [label]
        query_leaf_text = str(label_parts[-1]) if label_parts else ""
        query_leaf_tokens = _search_tokens(query_leaf_text)
        query_namespace = _search_token(label_parts[0]) if label_parts else ""
        document_count = max(1, len(self.relation_search_tokens))
        candidate_pool = {
            relation_id
            for token in {*query_tokens, *query_leaf_tokens}
            for relation_id in self.relation_token_index.get(token, ())
        }
        scored: list[tuple[float, str]] = []
        for relation_id in candidate_pool:
            candidate_tokens = self.relation_search_tokens.get(relation_id, ())
            candidate_leaf_tokens = self.relation_leaf_tokens.get(relation_id, ())
            # At least one full or leaf token must agree. Character similarity
            # is a tie-breaker, not permission to search the entire ontology.
            if not (
                set(query_tokens) & set(candidate_tokens)
                or set(query_leaf_tokens) & set(candidate_leaf_tokens)
            ):
                continue
            full_f1 = _weighted_token_f1(
                query_tokens,
                candidate_tokens,
                self.relation_token_document_frequency,
                document_count,
            )
            leaf_f1 = _weighted_token_f1(
                query_leaf_tokens,
                candidate_leaf_tokens,
                self.relation_token_document_frequency,
                document_count,
            )
            relation_leaf = str(relation_id).split(".")[-1].replace("_", " ")
            trigram = _trigram_jaccard(query_leaf_text, relation_leaf)
            namespace = (
                1.0
                if query_namespace
                and query_namespace == _search_token(str(relation_id).split(".")[0])
                else 0.0
            )
            score = (
                (0.35 * full_f1)
                + (0.45 * leaf_f1)
                + (0.15 * trigram)
                + (0.05 * namespace)
            )
            scored.append((score, relation_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ordered = [
            relation_id
            for _, relation_id in scored
            if relation_id != direct
        ]
        if direct in self.relation_ids:
            ordered.insert(0, direct)
        return ordered[:bounded_limit]

    def range_for_relation(self, relation_id: str) -> str:
        return str(self.relation_ranges.get(str(relation_id), ""))

    def domain_for_relation(self, relation_id: str) -> str:
        return str(self.relation_domains.get(str(relation_id), ""))

    def relations_for_domain(self, domain_id: str) -> tuple[str, ...]:
        return self.domain_to_relation_ids.get(str(domain_id), ())

    def reverse_for_relation(self, relation_id: str) -> tuple[str, ...]:
        return self.reverse_relation_ids.get(str(relation_id), ())

    def supertypes(self, type_id: str, *, max_depth: int = 6) -> tuple[str, ...]:
        """Return a bounded transitive superclass closure, including itself."""
        start = str(type_id)
        ordered: list[str] = []
        seen: set[str] = set()
        frontier = [start]
        depth = 0
        while frontier and depth <= max(0, int(max_depth)):
            next_frontier: list[str] = []
            for current in frontier:
                if not current or current in seen:
                    continue
                seen.add(current)
                ordered.append(current)
                next_frontier.extend(self.type_parents.get(current, ()))
            frontier = next_frontier
            depth += 1
        return tuple(ordered)

    def subtypes(self, type_id: str, *, max_depth: int = 2) -> tuple[str, ...]:
        """Return a bounded subclass closure, including the requested type."""
        start = str(type_id)
        children: dict[str, list[str]] = {}
        for child, parents in self.type_parents.items():
            for parent in parents:
                children.setdefault(parent, []).append(child)
        ordered: list[str] = []
        seen: set[str] = set()
        frontier = [start]
        depth = 0
        while frontier and depth <= max(0, int(max_depth)):
            next_frontier: list[str] = []
            for current in frontier:
                if not current or current in seen:
                    continue
                seen.add(current)
                ordered.append(current)
                next_frontier.extend(sorted(children.get(current, ())))
            frontier = next_frontier
            depth += 1
        return tuple(ordered)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "directory": self.directory,
            "relation_ids": len(self.relation_ids),
            "label_keys": len(self.label_to_relation_ids),
            "role_rows": self.role_rows,
            "type_rows": self.type_rows,
            "reverse_rows": self.reverse_rows,
            "overlay_rows": self.overlay_rows,
            "inverse_inferred_rows": self.inverse_inferred_rows,
        }
