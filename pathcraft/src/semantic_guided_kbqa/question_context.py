"""Read-only KG context for already-linked entities; never reads Gold answers."""
import re
from .evaluation import canonicalize_answer_id


def linked_entity_context(kg, linked_entities):
    ids = [x for x in linked_entities if re.fullmatch(r'[mg]\.[A-Za-z0-9_]+', x)]
    if not ids:
        return []
    values = ' '.join('ns:' + x for x in ids)
    query = '''PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?entity ?type ?label ?start ?end WHERE {
VALUES ?entity { ''' + values + ''' }
OPTIONAL { ?entity ns:type.object.type ?type .
FILTER(!STRSTARTS(STR(?type), "http://rdf.freebase.com/ns/base.") && !STRSTARTS(STR(?type), "http://rdf.freebase.com/ns/user.") && ?type != ns:common.topic)
OPTIONAL { ?type rdfs:label ?label . FILTER(LANG(?label) = "en") } }
OPTIONAL { ?entity ns:time.event.start_date ?start . }
OPTIONAL { ?entity ns:time.event.end_date ?end . }
} LIMIT 256'''
    records = {x: {'id': x, 'name': linked_entities[x], 'types': []} for x in ids}
    for row in kg.execute(query):
        entity = canonicalize_answer_id(row.get('entity', ''))
        if entity not in records:
            continue
        record = records[entity]
        label = row.get('label') or canonicalize_answer_id(row.get('type', '')).replace('.', ' ')
        if label and label not in record['types']:
            record['types'].append(label)
        for key in ('start', 'end'):
            if row.get(key):
                record[key + '_date'] = canonicalize_answer_id(row[key])
    # The Gold ID/name itself is useful anchor evidence even when the optional
    # type/date lookup returns no row. Never silently drop a configured entity.
    return list(records.values())
