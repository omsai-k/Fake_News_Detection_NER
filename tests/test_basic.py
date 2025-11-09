import unittest
import torch
from entity_aware_fnd.models.fusion_model import FusionClassifier, aggregate_entity_embeddings
from entity_aware_fnd.kg.embeddings import SimpleTransEStore

class TestFusionModel(unittest.TestCase):
    def test_forward_shapes(self):
        batch_size = 2
        text_dim = 768
        cls = FusionClassifier(text_dim=text_dim)
        store = SimpleTransEStore()
        # Fake text representations
        text_repr = torch.zeros((batch_size, text_dim))
        kg_repr = aggregate_entity_embeddings([["Q42", "Q1"], []], store.get, dim=store.dim)
        logits = cls(text_repr, kg_repr)
        self.assertEqual(list(logits.shape), [batch_size, 2])

if __name__ == '__main__':
    unittest.main()
