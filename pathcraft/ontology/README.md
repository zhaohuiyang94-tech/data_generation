# Freebase ontology assets

This directory contains the schema-layer assets copied from
`/home/yangzhaohui/grailqa_mas/ontology` for local CWQ relation candidate
generation and pruning.

## Files

- `fb_roles`: whitespace-separated `domain relation range` records;
  the middle column is the canonical Freebase relation ID.
- `fb_types`: Freebase type/subclass records.
- `reverse_properties`: relation-to-inverse-relation mappings.
- `domain_dict` and `domain_info`: copied source files retained for provenance.
  At the time of migration they are HTML documents rather than parsed plain
  dictionaries, so the runtime should not treat them as structured indexes
  without a separate conversion step.

These files describe Freebase schema, not entity-level facts. The current CWQ
runtime loads `fb_roles` once when `ontology.enabled` is true, maps Semantic
canonical relation labels to predicate IDs, and injects those IDs into
per-hop SPARQL `VALUES` constraints. The endpoint is still needed to verify
that a particular entity path exists and to execute the final query.
