from typing import List, Dict
from SPARQLWrapper import SPARQLWrapper, JSON
from pathlib import Path
import json
import hashlib

CACHE_DIR = Path("entity_aware_fnd/data/cache/kg")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class KGClient:
    def __init__(self, endpoint: str = "https://query.wikidata.org/sparql"):
        self.endpoint = endpoint
        self.client = SPARQLWrapper(endpoint)
        self.client.setReturnFormat(JSON)

    def _cache_key(self, qid: str) -> Path:
        return CACHE_DIR / f"{qid}.json"

    def neighbors(self, qid: str, limit: int = 10) -> List[Dict]:
        cache = self._cache_key(qid)
        if cache.exists():
            return json.loads(cache.read_text(encoding='utf-8'))
        query = f"""
        SELECT ?p ?pLabel ?o ?oLabel WHERE {{
          wd:{qid} ?p ?o .
          ?p a wikibase:Property .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT {limit}
        """
        self.client.setQuery(query)
        results = self.client.query().convert()
        triples = []
        for b in results['results']['bindings']:
            p = b['p']['value']
            o = b['o']['value']
            triples.append({
                'p': p, 'pLabel': b.get('pLabel', {}).get('value', ''),
                'o': o, 'oLabel': b.get('oLabel', {}).get('value', '')
            })
        cache.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding='utf-8')
        return triples
