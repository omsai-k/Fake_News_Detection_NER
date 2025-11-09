from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
from pathlib import Path

@dataclass
class NewsExample:
    id: str
    text: str
    label: Optional[int] = None  # 0 real, 1 fake

class NewsDataset:
    def __init__(self, examples: List[NewsExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @classmethod
    def from_csv(cls, path: str):
        df = pd.read_csv(path)
        examples = [NewsExample(str(r['id']), r['text'], int(r['label'])) for _, r in df.iterrows()]
        return cls(examples)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self.examples])
