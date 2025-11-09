from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def get_zero_shot(model_name: str = 'typeform/distilroberta-base-mnli'):
    return pipeline('zero-shot-classification', model=model_name)


def classify(text: str, labels=('real','fake'), hypothesis_template: str = 'This news is {}.'):
    z = get_zero_shot()
    return z(text, candidate_labels=list(labels), hypothesis_template=hypothesis_template)
