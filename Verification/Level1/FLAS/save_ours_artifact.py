"""Run FLASExtractor deterministically on the smoke dataset and save artifact.

Usage: python save_ours_artifact.py
"""
import json
import os
from pathlib import Path
import random
import torch


def set_deterministic(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    repo_root = Path(__file__).resolve().parents[3]
    artifacts = repo_root / 'Verification' / 'Level1' / 'FLAS' / 'artifacts'
    report_path = artifacts / 'comparison_report.json'
    report = json.loads(report_path.read_text())

    smoke_parquet = Path(report['smoke_data_dir']) / 'train_data.parquet'
    if not smoke_parquet.exists():
        raise FileNotFoundError(smoke_parquet)

    set_deterministic(int(report['hyperparameters'].get('seed', 42)))

    # import local extractor
    import sys
    sys.path.insert(0, str(repo_root))
    from Steering.extractors.nonlinear import FLASExtractor
    import pandas as pd

    df = pd.read_parquet(smoke_parquet)
    hyper = report['hyperparameters']

    class DummyModel:
        def __init__(self, device_str: str):
            self.cfg = type('C', (), {'device': device_str})
            self.tokenizer = None

    dummy = DummyModel(hyper['device'])
    extractor = FLASExtractor(
        model=dummy,
        layer=[int(hyper['layer'])],
        model_name=str(hyper['model_id']),
        batch_size=int(hyper['batch_size']),
        device=torch.device(str(hyper['device'])),
        hook_point='pre',
        flas_checkpoint_path=None,
        flas_num_blocks=int(hyper['num_blocks']),
        flas_time_conditioned=True,
        flas_disable_cross_attn=bool(hyper['disable_cross_attn']),
        flas_disable_self_attn=bool(hyper['disable_self_attn']),
        flas_disable_mlp=bool(hyper['disable_mlp']),
        flas_strict_load=True,
        flas_concept_encoder_layers=int(hyper['concept_encoder_layers']),
        flas_train_concept_text=None,
        flas_train_lr=float(hyper['lr']),
        flas_train_enc_lr=float(hyper['enc_lr']),
        flas_train_div_weight=float(hyper['div_weight']),
        flas_train_epochs=int(hyper['ours_epochs']),
        flas_train_batch_size=int(hyper['batch_size']),
        flas_train_grad_accum=int(hyper['grad_accum']),
        flas_train_max_len=int(hyper['max_len']),
        flas_train_concept_max_len=int(hyper['concept_max_len']),
        flas_train_T_min=float(hyper['T_min']),
        flas_train_T_max=float(hyper['T_max']),
        flas_train_n_steps=int(hyper['n_steps']),
        flas_train_seed=int(hyper['seed']),
        flas_train_max_steps=int(hyper['ours_max_steps']),
        flas_unfreeze_concept_enc=bool(hyper['unfreeze_concept_enc']),
        flas_no_gemma_init=bool(hyper['no_gemma_init']),
    )

    extractor.extract(
        target_data=df['output'].astype(str).tolist(),
        contrast_data=df['input'].astype(str).tolist(),
        flas_train_concepts=df['output_concept'].astype(str).tolist(),
        flas_train_concept_ids=df['concept_id'].astype(int).tolist(),
    )

    out_dir = artifacts / 'ours_vector'
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        'metadata': extractor.metadata,
        'vector': extractor.vector,
    }
    torch.save(artifact, out_dir / 'ours_flas_artifact.pt')
    print('Saved ours_flas_artifact.pt to', out_dir)


if __name__ == '__main__':
    main()
