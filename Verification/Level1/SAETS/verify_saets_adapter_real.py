import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
import logging
from pathlib import Path
import numpy as np
import scipy.linalg as linalg

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Benchmark root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import Reference Logic
sae_ts_path = str(Path("/home/aiotlab/mnt/hoplt/Benchmark/Code/SAE-TS/src"))
if sae_ts_path not in sys.path:
    sys.path.append(sae_ts_path)

try:
    from sae_ts.ft_effects.utils import LinearAdapter
    logger.info("Successfully imported LinearAdapter from Code/SAE-TS")
except ImportError as e:
    logger.error(f"Failed to import LinearAdapter: {e}")
    sys.exit(1)

device = "cuda:2" if torch.cuda.is_available() else "cpu"

from huggingface_hub import hf_hub_download

def download_effects_data(filename="effects_2b.pt"):
    """Download pre-computed effects data from HuggingFace."""
    repo_id = "schalnev/sae-ts-effects"
    try:
        logger.info(f"Downloading {filename} from {repo_id}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(Path("/home/aiotlab/mnt/hoplt/Benchmark")) # Download to Benchmark root
        )
        return Path(path)
    except Exception as e:
        logger.error(f"Error downloading effects data: {e}")
        raise

def load_data(path):
    if not path.exists():
        logger.info(f"{path} not found. Attempting download...")
        path = download_effects_data(path.name)
        
    logger.info(f"Loading effects data from {path}...")
    data = torch.load(path, map_location=device)
    features = data['features']
    effects = data['effects']
    
    # Normalize features (Reference logic: train.py line 50)
    # train.py: features = features / torch.norm(features, dim=-1, keepdim=True)
    # My sae.py: features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    # Use exact reference logic here for the reference training
    features = features / torch.norm(features, dim=-1, keepdim=True)
    
    return features, effects

def train_reference_adapter(features, effects, d_model, d_sae, epochs=2):
    """
    Train reference adapter using logic copied from Code/SAE-TS/src/sae_ts/ft_effects/train.py.
    """
    n_val = 100
    val_features = features[-n_val:]
    val_effects = effects[-n_val:]
    train_features = features[:-n_val]
    train_effects = effects[:-n_val]
    
    dataset = TensorDataset(train_features, train_effects)
    val_dataset = TensorDataset(val_features, val_effects)
    
    # Reference: shuffle=True, batch_size=64
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    adapter = LinearAdapter(d_model, d_sae).to(device)
    # Reference: lr=1e-4 (default in train arg) -> But main calls with lr=2e-4?
    # train.py main: train(15, lr=2e-4).
    # SAETSExtractor default: 1e-4.
    # verify_saets_real.py didn't specify, so it used default 1e-4.
    # So I should use 1e-4 here.
    lr = 1e-4 
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    
    logger.info(f"Training Reference Adapter for {epochs} epochs...")
    
    metrics = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        adapter.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_effects in dataloader:
            opt.zero_grad()
            batch_features = batch_features.to(device)
            batch_effects = batch_effects.to(device)
            pred = adapter(batch_features)
            loss = F.mse_loss(pred, batch_effects)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            num_batches += 1
            
        avg_train_loss = total_loss / num_batches
        
        # Val
        adapter.eval()
        val_total_loss = 0
        val_num_batches = 0
        with torch.no_grad():
            for batch_features, batch_effects in val_dataloader:
                batch_features = batch_features.to(device)
                batch_effects = batch_effects.to(device)
                pred = adapter(batch_features)
                loss = F.mse_loss(pred, batch_effects)
                val_total_loss += loss.item()
                val_num_batches += 1
                
        avg_val_loss = val_total_loss / val_num_batches
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        
    return adapter, metrics

def evaluate_adapter(adapter, features, effects):
    """Evaluate an adapter on given data."""
    adapter.eval()
    with torch.no_grad():
        pred = adapter(features.to(device))
        loss = F.mse_loss(pred, effects.to(device))
    return loss.item()

def main():
    torch.manual_seed(42) # Consistent seed
    
    base_dir = Path("/home/aiotlab/mnt/hoplt/Benchmark")
    effects_path = base_dir / "effects_2b.pt"
    
    # Let load_data handle download if missing
    features, effects = load_data(effects_path)
    logger.info(f"Data shape: Features {features.shape}, Effects {effects.shape}")
    
    d_model = features.shape[1]
    d_sae = effects.shape[1]
    
    # 1. Train Reference Adapter
    # Matches verify_saets_real.py config (2 epochs)
    # verify_saets_real.py used: ExtractorConfig(..., saets_epochs=2)
    epochs = 2
    adapter_ref, metrics_ref = train_reference_adapter(features, effects, d_model, d_sae, epochs=epochs)
    
    # 2. Load My Adapter (from verify_saets_real.py)
    my_adapter_path = base_dir / "Verification_Results" / "saets_adapter_real.pt"
    if not my_adapter_path.exists():
        logger.error(f"My Adapter not found at {my_adapter_path}")
        sys.exit(1)
        
    logger.info(f"Loading My Adapter from {my_adapter_path}...")
    my_adapter = LinearAdapter(d_model, d_sae).to(device)
    my_adapter.load_state_dict(torch.load(my_adapter_path, map_location=device))
    
    # 3. Compare Performance
    # Validation data (last 100)
    n_val = 100
    val_features = features[-n_val:].to(device)
    val_effects = effects[-n_val:].to(device)
    
    loss_ref = evaluate_adapter(adapter_ref, val_features, val_effects)
    loss_mine = evaluate_adapter(my_adapter, val_features, val_effects)
    
    logger.info("==========================================")
    logger.info("Validation Results (MSE Loss on last 100 samples)")
    logger.info(f"Reference Adapter: {loss_ref:.4f}")
    logger.info(f"My Adapter:        {loss_mine:.4f}")
    logger.info("==========================================")
    
    # Check if comparable (e.g. within 2x or absolute difference)
    # My adapter trained on ALL data, Ref trained on Train Split.
    # My adapter should likely be BETTER or EQUAL on Val data (as it saw it during training).
    # If My Adapter loss is HUGE, then something is wrong.
    
    if loss_mine > (loss_ref * 1.5): # Liberal threshold
        logger.warning("FAILED: My Adapter loss is significantly higher than Reference.")
    else:
        logger.info("PASSED: My Adapter performance is comparable to Reference.")

    # 4. Compare Matrices (Optional)
    # Cosine similarity of flattened weights
    w_ref = adapter_ref.W.detach().flatten()
    w_mine = my_adapter.W.detach().flatten()
    cos_sim = F.cosine_similarity(w_ref.unsqueeze(0), w_mine.unsqueeze(0)).item()
    logger.info(f"Matrix Cosine Similarity: {cos_sim:.4f}")

if __name__ == "__main__":
    main()
