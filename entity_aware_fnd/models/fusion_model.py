from typing import List, Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512, freeze: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        outputs = self.model(**batch)
        # CLS pooling
        cls = outputs.last_hidden_state[:, 0, :]
        return cls


class FusionClassifier(nn.Module):
    def __init__(self, text_dim: int = 768, kg_dim: int = 100, hidden_dims: List[int] = [512, 256], dropout: float = 0.2, fusion: str = 'concat'):
        super().__init__()
        self.fusion = fusion
        input_dim = text_dim + kg_dim if fusion == 'concat' else text_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 2)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, text_repr: torch.Tensor, kg_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.fusion == 'concat' and kg_repr is not None:
            x = torch.cat([text_repr, kg_repr], dim=-1)
        else:
            x = text_repr
        return self.mlp(x)


def aggregate_entity_embeddings(entity_qids: List[List[str]], get_vec, dim: int = 100) -> torch.Tensor:
    """Average embeddings per sample; returns tensor [B, dim]."""
    reps = []
    for qids in entity_qids:
        if len(qids) == 0:
            reps.append(np.zeros((dim,), dtype=np.float32))
        else:
            arr = np.vstack([get_vec(q) for q in qids])
            reps.append(arr.mean(axis=0))
    return torch.from_numpy(np.vstack(reps))
