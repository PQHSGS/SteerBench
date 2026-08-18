import pytest
import torch
from unittest.mock import MagicMock

from Steering.extractors.nonlinear import BIPOExtractor
from Steering.steer_models.nonlinear import BIPOSteerModel

class MockModel(torch.nn.Module):
    def __init__(self, d_model=16, d_vocab=100):
        super().__init__()
        self.cfg = MagicMock()
        self.cfg.device = "cpu"
        self.cfg.d_model = d_model
        self.cfg.dtype = torch.float32
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.linear = torch.nn.Linear(d_model, d_vocab)

    def to_tokens(self, text, prepend_bos=True):
        if isinstance(text, str):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)
        return torch.tensor([[1, 2, 3] for _ in range(len(text))], dtype=torch.long)

    def forward(self, tokens):
        # returns logits
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        
        logits = torch.randn(batch_size, seq_len, self.d_vocab, requires_grad=True)
        return logits

    def hooks(self, win_hooks):
        # Context manager that does nothing in mock
        class DummyContext:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return DummyContext()


def test_bipo_extractor():
    d_model = 16
    d_vocab = 50
    model = MockModel(d_model, d_vocab)
    
    extractor = BIPOExtractor(
        model=model,
        layer=[10],
        batch_size=2,
        device=torch.device("cpu"),
        bipo_lr=1e-3,
        bipo_beta=0.1,
        bipo_epochs=2,
    )
    
    target_data = ["test win 1", "test win 2"]
    contrast_data = ["test lose 1", "test lose 2"]
    
    vec = extractor.extract(target_data, contrast_data)
    
    assert isinstance(vec, dict)
    assert 10 in vec
    assert vec[10].shape == (d_model,)
    assert extractor.metadata["method"] == "BIPO"
    assert extractor.metadata["bipo_epochs"] == 2


def test_bipo_steer_model():
    d_model = 16
    model = MockModel(d_model)
    
    steering_vector = torch.randn(d_model)
    
    steer_model = BIPOSteerModel(
        model=model,
        layer=[10],
        steering_vector={10: steering_vector},
    )
    
    # Test hook
    batch_size = 2
    seq_len = 5
    resid = torch.randn(batch_size, seq_len, d_model)
    coeff = 2.0
    
    original_resid = resid.clone()
    
    mod_resid = steer_model.hook_fn(
        resid=resid.clone(),
        coeff=coeff,
        position="all",
        steering_vector=steering_vector,
        hook=None,
    )
    
    # Check that modification happened to all tokens
    assert not torch.allclose(original_resid, mod_resid)
    assert torch.allclose(mod_resid, original_resid + coeff * steering_vector)


if __name__ == "__main__":
    test_bipo_extractor()
    test_bipo_steer_model()
    print("BIPO Tests passed successfully!")
