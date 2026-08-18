import pytest
import torch
from unittest.mock import MagicMock

from Steering.extractors.nonlinear import IDSExtractor
from Steering.steer_models.nonlinear import IDSSteerModel


class MockModel:
    def __init__(self, d_model=16):
        self.cfg = MagicMock()
        self.cfg.device = "cpu"
        self.tokenizer = MagicMock()
        self.tokenizer.chat_template = None
        self.d_model = d_model

    def to_tokens(self, text):
        return torch.tensor([[1, 2, 3]])

    def run_with_cache(self, inputs, names_filter=None, return_type=None, attention_mask=None):
        if isinstance(inputs, list):
            batch_size = len(inputs)
            seq_len = 5
        else:
            batch_size = inputs.shape[0]
            seq_len = inputs.shape[1]

        hook_names = names_filter or []
        # Return deterministic but distinct activations to ensure covariance is positive definite
        cache = {}
        for name in hook_names:
            acts = torch.randn(batch_size, seq_len, self.d_model)
            # Add a strong variance along first few dimensions to avoid singularity
            for i in range(min(5, self.d_model)):
                acts[..., i] += 5.0 * torch.randn(batch_size, seq_len)
            cache[name] = acts

        return None, cache


def test_ids_extractor():
    d_model = 16
    model = MockModel(d_model)
    
    extractor = IDSExtractor(
        model=model,
        layer=[10],
        batch_size=2,
        ids_var_explained=0.40,
        ids_epsilon_pct=0.95,
        ids_f1_threshold=0.50,
        ids_ot_eps=1e-5,
        device=torch.device("cpu")
    )
    
    target_data = ["test positive prompt 1", "test positive prompt 2", "test positive prompt 3"]
    contrast_data = ["test negative prompt 1", "test negative prompt 2", "test negative prompt 3"]
    
    vec = extractor.extract(target_data=target_data, contrast_data=contrast_data)
    
    assert isinstance(vec, dict)
    assert 10 in vec
    assert vec[10].shape == (d_model,)
    assert extractor.metadata["method"] == "IDS"
    assert "layer_stats" in extractor.metadata
    
    stats = extractor.metadata["layer_stats"][10]
    assert "pca_components" in stats
    assert "pca_mean" in stats
    assert "mu_tgt_pca" in stats
    assert "L_inv" in stats
    assert "epsilon_sq" in stats
    assert "f1_score" in stats
    
    # Check that dimensions are consistent
    r = stats["pca_components"].shape[1]
    assert stats["pca_components"].shape[0] == d_model
    assert stats["pca_mean"].shape == (d_model,)
    assert stats["mu_tgt_pca"].shape == (r,)
    assert stats["L_inv"].shape == (r, r)
    assert isinstance(stats["epsilon_sq"], float)
    assert isinstance(stats["f1_score"], float)


def test_ids_steer_model():
    torch.manual_seed(42)
    d_model = 16
    r = 4
    model = MockModel(d_model)
    
    # Setup mock metadata stats
    pca_components = torch.randn(d_model, r)
    # Ensure columns of pca_components are orthogonal
    pca_components, _ = torch.linalg.qr(pca_components)
    
    pca_mean = torch.randn(d_model)
    mu_tgt_pca = torch.randn(r)
    L_inv = torch.eye(r)
    epsilon_sq = 9.5
    f1_score = 0.85
    
    layer_stats = {
        10: {
            "pca_components": pca_components,
            "pca_mean": pca_mean,
            "mu_tgt_pca": mu_tgt_pca,
            "L_inv": L_inv,
            "epsilon_sq": epsilon_sq,
            "f1_score": f1_score,
        }
    }
    
    steering_vector = {10: torch.randn(d_model)}
    
    steer_model = IDSSteerModel(
        model=model,
        layer=[10],
        steering_vector=steering_vector,
        layer_stats=layer_stats,
        ids_f1_threshold=0.70,
        position="last",
    )
    
    # Mock Hook class
    mock_hook = MagicMock()
    mock_hook.name = "layers.10.hook_resid_pre"
    
    batch_size = 2
    seq_len = 5
    resid = torch.randn(batch_size, seq_len, d_model)
    coeff = 1.5
    
    original_resid = resid.clone()
    
    # Test hook application
    mod_resid = steer_model.hook_fn(
        resid=resid.clone(),
        coeff=coeff,
        position="last",
        steering_vector=steering_vector[10],
        hook=mock_hook,
    )
    
    # Assert modification happened
    assert not torch.allclose(original_resid, mod_resid)
    # Check that modification affected last token
    assert not torch.allclose(original_resid[:, -1, :], mod_resid[:, -1, :])
    # Check that it did NOT affect other tokens (since position is "last")
    assert torch.allclose(original_resid[:, :-1, :], mod_resid[:, :-1, :])
    
    # Test layer selection (F1-score below threshold)
    low_f1_stats = {
        10: {
            "pca_components": pca_components,
            "pca_mean": pca_mean,
            "mu_tgt_pca": mu_tgt_pca,
            "L_inv": L_inv,
            "epsilon_sq": epsilon_sq,
            "f1_score": 0.45,  # 0.45 < 0.70 threshold
        }
    }
    
    steer_model_low_f1 = IDSSteerModel(
        model=model,
        layer=[10],
        steering_vector=steering_vector,
        layer_stats=low_f1_stats,
        ids_f1_threshold=0.70,
        position="last",
    )
    
    mod_resid_low_f1 = steer_model_low_f1.hook_fn(
        resid=resid.clone(),
        coeff=coeff,
        position="last",
        steering_vector=steering_vector[10],
        hook=mock_hook,
    )
    
    # Assert no modification happened because of layer selection threshold
    assert torch.allclose(original_resid, mod_resid_low_f1)


if __name__ == "__main__":
    test_ids_extractor()
    test_ids_steer_model()
    print("All IDS tests passed!")
