import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'entity_aware_fnd' is importable when running via Streamlit
def add_project_root():
    here = Path(__file__).resolve()
    # climb up to find a directory that contains 'entity_aware_fnd'
    cur = here.parent
    for _ in range(5):
        if (cur / 'entity_aware_fnd').exists():
            if str(cur) not in sys.path:
                sys.path.insert(0, str(cur))
            break
        cur = cur.parent

add_project_root()

from entity_aware_fnd.nlp.ner import extract_entities
from entity_aware_fnd.nlp.linking import link_entities
from entity_aware_fnd.models.fusion_model import TextEncoder, FusionClassifier, aggregate_entity_embeddings
from entity_aware_fnd.kg.embeddings import SimpleTransEStore
from entity_aware_fnd.config import load_config
from transformers import pipeline
import torch

st.set_page_config(page_title="Entity-Aware Fake News Detection")
cfg = load_config('configs/default.yaml')

st.title("Entity-Aware Fake News Detection Using Knowledge Graphs")

text = st.text_area("Paste news article text", height=200)

if 'init' not in st.session_state:
    # Try load checkpoint for trained classifier
    from pathlib import Path
    ckpt = Path(cfg.get('paths.checkpoints_dir', 'checkpoints')) / 'best.pt'
    st.session_state.use_zero_shot = False
    if ckpt.exists():
        st.session_state.text_encoder = TextEncoder(cfg.get('model.text_encoder'), cfg.get('model.max_length', 512), True)
        st.session_state.cls = FusionClassifier()
        state = torch.load(str(ckpt), map_location='cpu')
        st.session_state.cls.load_state_dict(state['classifier'])
    else:
        # zero-shot fallback
        st.session_state.use_zero_shot = True
        z_model = cfg.get('inference.zero_shot_model', 'typeform/distilroberta-base-mnli')
        st.session_state.zero_shot = pipeline('zero-shot-classification', model=z_model)
    st.session_state.store = SimpleTransEStore()
    st.session_state.init = True

if st.button("Analyze") and text.strip():
    ents = extract_entities(text)
    linked = link_entities(ents)
    qids = [c['qid'] for ent in linked for c in ent.get('candidates', [])[:1]]

    if st.session_state.use_zero_shot:
        z_out = st.session_state.zero_shot(text, candidate_labels=['real','fake'])
        scores = dict(zip(z_out['labels'], z_out['scores']))
        probs = [scores.get('real', 0.0), scores.get('fake', 0.0)]
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.session_state.text_encoder.to(device)
        st.session_state.cls.to(device)
        with torch.no_grad():
            text_repr = st.session_state.text_encoder([text]).to(device)
            kg_repr = aggregate_entity_embeddings([qids], st.session_state.store.get, dim=st.session_state.store.dim).to(device)
            logits = st.session_state.cls(text_repr, kg_repr)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    st.subheader("Prediction")
    st.write({"real": float(probs[0]), "fake": float(probs[1])})

    st.subheader("Entities & Links")
    st.json(linked)
