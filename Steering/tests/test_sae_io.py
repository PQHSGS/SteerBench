
import pytest
import torch
from unittest.mock import MagicMock

from Steering.extractors.sae import SAEIOExtractor
from Steering.steer_models.sae import SAEIOSteerModel

# Mock classes
class MockSAE:
    def __init__(self, d_model=16, n_latents=32):
        self.W_dec = torch.randn(n_latents, d_model)
        self.cfg = MagicMock()
        self.cfg.d_sae = n_latents
        self.d_sae = n_latents
        
    def encode(self, x):
        # simple linear + relu
        return torch.relu(x @ self.W_dec.T)
        
    def decode(self, z):
        return z @ self.W_dec

class MockModel:
    def __init__(self, d_model=16, d_vocab=100):
        self.cfg = MagicMock()
        self.cfg.device = "cpu"
        self.tokenizer = MagicMock()
        self.tokenizer.chat_template = None
        self.d_model = d_model
        
        # Components for logit lens
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.unembed = torch.nn.Linear(d_model, d_vocab)
        
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
        cache = {
            hook_name: torch.randn(batch_size, seq_len, self.d_model)
            for hook_name in hook_names
        }
        return None, cache
        
    def run_with_hooks_with_saes(self, inputs, saes, fwd_hooks):
        # Simulate HookedSAETransformer behavior
        # inputs is list of strings or tokens
        # We just call the hook with dummy data
        batch_size = len(inputs) if isinstance(inputs, list) else 1
        seq_len = 5
        n_latents = saes.d_sae
        
        # Fake activations
        sae_acts = torch.rand(batch_size, seq_len, n_latents)
        
        for name, hook_fn in fwd_hooks:
            # hook_fn(sae_acts, hook)
            hook_fn(sae_acts, None)
            
    def run_with_hooks(self, tokens, fwd_hooks):
        # Simulate HookedTransformer behavior
        # returns logits
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]
        
        # Fake residual stream
        resid = torch.zeros(batch_size, seq_len, self.d_model)
        
        for name, hook_fn in fwd_hooks:
            # hook_fn(resid, hook)
            # Modifies resid in place usually
            ret = hook_fn(resid, None)
            if ret is not None:
                resid = ret
                
        # Fake logits
        return torch.randn(batch_size, seq_len, 100)


def test_sae_io_extractor():
    d_model = 16
    n_latents = 32
    d_vocab = 50
    
    sae = MockSAE(d_model, n_latents)
    model = MockModel(d_model, d_vocab)
    
    extractor = SAEIOExtractor(
        model=model,
        sae={10: sae},
        layer=[10],
        batch_size=2,
        top_k=5,
        act_threshold=0.5,
        amp_factor=10.0,
        device=torch.device("cpu")
    )
    
    # Test extraction
    target_data = ["test prompt 1", "test prompt 2"]
    
    # We need to mock cache_logit_lens and get_output_score in Steering.utils
    # but they are imported inside the method.
    # We can patch `Steering.utils.cache_logit_lens` if imported at top level
    # but here imported inside.
    
    # Actually, we can just let them run with our MockModel if it supports them.
    # cache_logit_lens uses sae.W_dec, model.ln_final, model.unembed.
    # Our MockModel has them.
    
    # get_output_score uses model.run_with_hooks. Our MockModel has it.
    
    vec = extractor.extract(target_data)
    
    # extract() returns Dict[int, Tensor]
    assert isinstance(vec, dict)
    assert 10 in vec
    assert vec[10].shape == (d_model,)
    assert extractor.metadata["method"] == "SAE_IO"
    assert 10 in extractor.metadata["top_idx"]
    assert len(extractor.selected_features[10]) <= 5
    
    # Verify sparse_latent metadata stores the binary SAE feature mask
    sparse_latent = extractor.metadata["sparse_latent"][10]
    if getattr(sparse_latent, "is_sparse", False):
        sparse_latent = sparse_latent.to_dense()
    assert sparse_latent.shape == (n_latents,)
    assert torch.all(torch.logical_or(sparse_latent == 0, sparse_latent == 1))


def test_sae_io_steer_model():
    d_model = 16
    n_latents = 32
    
    sae = MockSAE(d_model, n_latents)
    model = MockModel(d_model)
    
    # sparse_latent carries feature mask; steering_vector is dense decoded direction.
    sparse_latent = torch.zeros(n_latents)
    sparse_latent[0] = 1.0
    steering_vector = sae.decode(sparse_latent)
    
    steer_model = SAEIOSteerModel(
        model=model,
        layer=[10],
        sae={10: sae},
        steering_vector={10: steering_vector},
        sparse_latent={10: sparse_latent},
    )
    
    # Test hook
    batch_size = 2
    seq_len = 5
    resid = torch.randn(batch_size, seq_len, d_model)
    coeff = 2.0
    
    # We expect resid to change
    original_resid = resid.clone()
    
    mod_resid = steer_model.hook_fn(
        resid=resid.clone(),
        coeff=coeff,
        position="last",
        steering_vector=steering_vector,
        sae=sae,
        top_idx=[0],
        sparse_latent=sparse_latent,
        hook=None,
    )
    
    # Check that modification happened
    assert not torch.allclose(original_resid, mod_resid)
    
    # Check that modification affected last token
    assert not torch.allclose(original_resid[:, -1, :], mod_resid[:, -1, :])
    
    # Check adaptive scaling logic?
    # Hard to verify exact math without re-implementing logic, 
    # but we can check direction if we control SAE.W_dec.
    
    # If we set W_dec[0] to unit vector [1, 0, ...]
    # And input resid such that feature 0 activates.

if __name__ == "__main__":
    test_sae_io_extractor()
    test_sae_io_steer_model()
    print("Tests passed!")
