from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

@dataclass
class Batch:
    texts: List[str]
    kg_qids: List[List[str]]
    labels: torch.Tensor

class SimpleDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def train_loop(model_text, model_cls, embed_getter, train_data, val_data, epochs=3, lr=2e-5, weight_decay=0.01, warmup_steps=100, batch_size=8, device='cpu', eval_steps=200, mixed_precision=True, save_path: str = None):
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device.startswith('cuda'))

    optimizer = torch.optim.AdamW(list(model_text.parameters()) + list(model_cls.parameters()), lr=lr, weight_decay=weight_decay)

    t_total = max(1, (len(train_data) // batch_size) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    ce = nn.CrossEntropyLoss()
    model_text.to(device)
    model_cls.to(device)

    step = 0
    best_f1 = -1
    best_state = None

    def collate(batch):
        # Preserve variable-length qid lists without attempting tensor conversion
        return {
            'texts': [b['text'] for b in batch],
            'labels': torch.tensor([b['label'] for b in batch], dtype=torch.long, device=device),
            'kg_qids': [b.get('qids', []) for b in batch]
        }

    def make_loader(data):
        return DataLoader(SimpleDataset(data), batch_size=batch_size, shuffle=True, collate_fn=collate)

    for epoch in range(epochs):
        model_text.train(); model_cls.train()
        for batch in make_loader(train_data):
            texts = batch['texts']
            labels = batch['labels']
            kg_qids = batch['kg_qids']

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                text_repr = model_text(texts).to(device)
                kg_repr = embed_getter(kg_qids).to(device)
                logits = model_cls(text_repr, kg_repr)
                loss = ce(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            if step % eval_steps == 0:
                metrics = evaluate(model_text, model_cls, embed_getter, val_data, device=device)
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_state = {
                        'classifier': model_cls.state_dict(),
                        'text_encoder_frozen': all(not p.requires_grad for p in model_text.parameters())
                    }
        
    if save_path and best_state:
        torch.save(best_state, save_path)
    return best_f1


def evaluate(model_text, model_cls, embed_getter, data, device='cpu'):
    model_text.eval(); model_cls.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for item in data:
            text = item['text']
            label = item.get('label', 0)
            qids = item.get('qids', [])
            text_repr = model_text([text]).to(device)
            kg_repr = embed_getter([qids]).to(device)
            logits = model_cls(text_repr, kg_repr)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(probs.argmax())
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(probs[1])
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    }
    return metrics
