import argparse
from entity_aware_fnd.config import load_config
from entity_aware_fnd.utils.logging import setup_logging, banner
from entity_aware_fnd.nlp.ner import extract_entities
from entity_aware_fnd.nlp.linking import link_entities
from entity_aware_fnd.kg.embeddings import SimpleTransEStore
from entity_aware_fnd.models.fusion_model import TextEncoder, FusionClassifier, aggregate_entity_embeddings
import torch
from pathlib import Path
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--text', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get('logging.level', 'INFO'))
    banner('Entity-Aware Fake News Detection - Inference')

    text = args.text
    ents = extract_entities(text)
    linked = link_entities(ents)
    qids = [c['qid'] for ent in linked for c in ent.get('candidates', [])[:1]]

    text_encoder = TextEncoder(cfg.get('model.text_encoder'), cfg.get('model.max_length', 512), True)
    cls = FusionClassifier(fusion=cfg.get('model.fusion', 'concat'))

    ckpt_path = Path(cfg.get('paths.checkpoints_dir', 'checkpoints')) / 'best.pt'
    used_zero_shot = False
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location='cpu')
        cls.load_state_dict(state['classifier'])
    elif cfg.get('inference.use_zero_shot_fallback', True):
        used_zero_shot = True
        z_model = cfg.get('inference.zero_shot_model', 'typeform/distilroberta-base-mnli')
        z = pipeline('zero-shot-classification', model=z_model)
        z_out = z(text, candidate_labels=['real', 'fake'])
        print({'zero_shot': z_out})
        # continue for entity display; return early for classification result
        return
    store = SimpleTransEStore(dim=cfg.get('kg.embedding_dim', 100))
    embed = aggregate_entity_embeddings([qids], store.get, dim=store.dim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_encoder.to(device); cls.to(device)
    with torch.no_grad():
        text_repr = text_encoder([text]).to(device)
        logits = cls(text_repr, embed.to(device))
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    print({'probs': probs.tolist(), 'entities': linked, 'used_zero_shot': used_zero_shot})

if __name__ == '__main__':
    main()
