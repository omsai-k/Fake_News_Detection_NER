import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key: str, default=None):
        parts = key.split('.')
        cur = self._data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    @property
    def raw(self):
        return self._data

    def __getitem__(self, item):
        return self._data[item]


def load_config(path: str = 'configs/default.yaml') -> Config:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(data)

CONFIG = load_config()  # simple global convenience
