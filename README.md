# Entity-Aware Fake News Detection Using Knowledge Graphs

An end-to-end NLP system that detects fake news by enriching text representations with entity-centric knowledge graph (KG) signals. It performs:

1. Text preprocessing (cleaning, sentence segmentation)
2. Named Entity Recognition (NER)
3. Entity Linking (Wikidata / DBpedia)
4. Subgraph extraction & KG feature engineering (neighbors, relation types, centrality)
5. KG Embeddings (simple placeholder embeddings for demo)
6. Text encoding (Transformers: BERT/RoBERTa)
7. Fusion of text + KG embeddings (attention / concatenation)
8. Classification (fake vs real) + rich evaluation (F1, ROC-AUC, explainability)
9. Interactive Streamlit app for inference and entity context visualization.

## Quick Start

```powershell
# Create environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt

# (Optional) Download SpaCy model
python -m spacy download en_core_web_sm

# Train
python -m entity_aware_fnd.cli.train --config configs/default.yaml

# Infer single text
python -m entity_aware_fnd.cli.infer --text "Sample news article text" --config configs/default.yaml

# Launch app
streamlit run entity_aware_fnd/app/streamlit_app.py
```

## Project Structure
```
entity_aware_fnd/
  data/               # Data loaders, preprocessing, caching
  nlp/                # NER and entity linking
  kg/                 # Graph building and embeddings
  models/             # Fusion model
  training/           # Trainer, schedulers
  evaluation/         # Metrics and analysis
  cli/                # Command line interfaces
  app/                # Streamlit UI
  utils/              # Logging, config helpers
configs/              # YAML config files
checkpoints/          # Saved model weights
logs/                 # Training & evaluation logs
```

## Configuration
Edit `configs/default.yaml` to control model architecture, data paths, and hyperparameters.

## Data Expectations
Provide a CSV with columns:
- id
- text
- label (0 = real, 1 = fake)

Place raw dataset at `entity_aware_fnd/data/raw/news.csv` or override via config.

## Fusion Model Overview
- Text encoder: HuggingFace transformer (CLS embedding)
- KG entity embeddings: averaged or attention-weighted
- Fusion: concatenation + MLP or attention block

## Metrics
- Accuracy, Precision, Recall, F1
- ROC-AUC
- Calibration (optional)
- Entity coverage & linking success rate

## Explainability
For each prediction, we surface:
- Contributing entities and attention weights
- Top related KG triples

## Extensibility
- Swap text encoder via config
- Replace placeholder KG embeddings with trained embeddings (e.g., PyKEEN) if needed later

## License
MIT (adjust as needed).

## Disclaimer
Entity linking and KG queries may require network access and can be rate-limited. Caching is recommended for reproducibility.
