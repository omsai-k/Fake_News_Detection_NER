import argparse
import torch
from entity_aware_fnd.config import load_config
from entity_aware_fnd.utils.logging import setup_logging, banner
from entity_aware_fnd.data.dataset import NewsDataset
from entity_aware_fnd.nlp.ner import extract_entities
from entity_aware_fnd.nlp.linking import link_entities
from entity_aware_fnd.kg.embeddings import SimpleTransEStore
from entity_aware_fnd.models.fusion_model import TextEncoder, FusionClassifier, aggregate_entity_embeddings
from entity_aware_fnd.training.trainer import train_loop, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get('logging.level', 'INFO'))
    banner('Entity-Aware Fake News Detection - Train')

    # Load data
    dataset = NewsDataset.from_csv(cfg.get('paths.raw_data'))
    df = dataset.to_dataframe()
    if cfg.get('data.shuffle', True):
        df = df.sample(frac=1.0, random_state=cfg.get('seed', 42)).reset_index(drop=True)
    n = len(df)
    n_train = int(n * cfg.get('data.train_split', 0.8))
    n_val = int(n * cfg.get('data.val_split', 0.1))

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]

    # NER + Linking (demo: do minimal preprocessing)
    def process_row(row):
        ents = extract_entities(row['text'])
        linked = link_entities(ents)
        qids = [c['qid'] for ent in linked for c in ent.get('candidates', [])[:1]]  # take top-1 candidate per ent
        return {'id': row['id'], 'text': row['text'], 'label': int(row['label']), 'qids': qids}

    train_data = [process_row(r) for _, r in train_df.iterrows()]
    val_data = [process_row(r) for _, r in val_df.iterrows()]
    test_data = [process_row(r) for _, r in test_df.iterrows()]

    # Models
    text_encoder = TextEncoder(cfg.get('model.text_encoder'), cfg.get('model.max_length', 512), cfg.get('model.freeze_text_encoder', False))
    cls = FusionClassifier(fusion=cfg.get('model.fusion', 'concat'))

    # KG embeddings store (placeholder)
    store = SimpleTransEStore(dim=cfg.get('kg.embedding_dim', 100))
    embed_getter = lambda batch_qids: aggregate_entity_embeddings(batch_qids, store.get, dim=store.dim)

    # Ensure checkpoints dir exists
    from pathlib import Path
    ckpt_dir = Path(cfg.get('paths.checkpoints_dir', 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / 'best.pt')

    best = train_loop(text_encoder, cls, embed_getter, train_data, val_data,
                      epochs=cfg.get('training.num_epochs', 3),
                      lr=float(cfg.get('training.lr', 2e-5)),
                      weight_decay=float(cfg.get('training.weight_decay', 0.01)),
                      warmup_steps=int(cfg.get('training.warmup_steps', 100)),
                      batch_size=int(cfg.get('training.batch_size', 8)),
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      eval_steps=int(cfg.get('training.eval_steps', 200)),
                      mixed_precision=bool(cfg.get('training.mixed_precision', True)),
                      save_path=ckpt_path)

    print({'best_val_f1': best})

    # Final test evaluation
    metrics = evaluate(text_encoder, cls, embed_getter, test_data, device='cuda' if torch.cuda.is_available() else 'cpu')
    print({'test_metrics': metrics})

if __name__ == '__main__':
    main()
