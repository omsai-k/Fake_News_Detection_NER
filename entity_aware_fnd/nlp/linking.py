from typing import List, Dict, Optional
import requests
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("entity_aware_fnd/data/cache/linking")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

HEADERS = {
    'Accept': 'application/sparql-results+json'
}


def _cache_key(text: str) -> Path:
    return CACHE_DIR / (hashlib.md5(text.encode('utf-8')).hexdigest() + ".json")


def _get_cached(text: str) -> Optional[List[Dict]]:
    p = _cache_key(text)
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return None


def _set_cached(text: str, value: List[Dict]):
    p = _cache_key(text)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def link_entities(entities: List[Dict]) -> List[Dict]:
    """
    Very lightweight entity linking by querying Wikidata for entity labels.
    For production-grade EL, integrate BLINK or spaCy-EL.
    """
    results = []
    for ent in entities:
        label = ent['text']
        cached = _get_cached(label)
        if cached is not None:
            qids = cached
        else:
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            SELECT ?item ?itemLabel WHERE {{
              ?item rdfs:label "{label}"@en .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }} LIMIT 3
            """
            r = requests.get(WIKIDATA_SPARQL, params={'query': query, 'format': 'json'}, headers=HEADERS, timeout=30)
            data = r.json()
            qids = []
            for b in data.get('results', {}).get('bindings', []):
                item = b['item']['value']
                if item.startswith('http://www.wikidata.org/entity/'):
                    qids.append({'qid': item.split('/')[-1], 'label': b.get('itemLabel', {}).get('value', label)})
            _set_cached(label, qids)
        results.append({**ent, 'candidates': qids})
    return results
