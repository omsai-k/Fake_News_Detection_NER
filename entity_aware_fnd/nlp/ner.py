import spacy
from typing import List, Dict
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model(name: str = "en_core_web_sm"):
    return spacy.load(name)


def extract_entities(text: str) -> List[Dict]:
    nlp = load_model()
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return ents
