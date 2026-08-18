"""
Method Configurations.

Contains ExtractorConfig, SteerConfig with all defaults as single source of truth.
Also defines SAE_METHODS set.

Parameter Organization:
- Required params first (method, layer)
- Common params shared by all/most methods
- Method-specific params grouped alphabetically by method name
"""

import warnings
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Optional, Set, Union, List, Tuple
import torch

# =============================================================================
# METHOD CLASSIFICATION
# =============================================================================

# Dense methods: work directly on residual stream activations
DENSE_METHODS: Set[str] = {
    "CAA", "COLD", "CAST", "SPHERICAL", "MANIFOLD", "SAE-FREE", "LQR", "JSPACE", "RIEMANNIAN"
}

# Nonlinear methods: work on activations nonlinearly
NONLINEAR: Set[str] = {
    "ANGULAR", "ACT", "CURVEBALL", "FLOW", "PID", "LOREFT", "REPS", "ODE", "FLAS", "BIPO", "CHARS", "COBRA", "LinNEAS",
    "INNSTEER", "IDS", "FISHBACK", "GINN",
}

# Weight methods: work on parameter weights
WEIGHT: Set[str] = {"WEIGHTSTEER"}

# SAE methods: require Sparse Autoencoder
SAE_METHODS: Set[str] = {"SAS", "SPARE", "SRE", "SRPS", "SSV", "CORRSTEER", "SAE-RSV", "SAE-TS", "SAEIO", "SAE-COT", "FEAT", "FGAA"}


EXTRACTOR_COMMON_FIELDS: Set[str] = {
    "batch_size",
    "position",
    "apply_chat_template",
    "hook_point",
    "change_pad_token",
    "inverse"
}

EXTRACTOR_SAE_COMMON_FIELDS: Set[str] = {
    "top_k",
    "act_threshold",
    "act_frac",
}

EXTRACTOR_METHOD_FIELDS: Dict[str, Set[str]] = {
    "ANGULAR": {"strategy"},
    "CAST": {
        "use_pca",
        "conditional_dataset",
        "conditional_layer",
        "apply_conditional_chat_template",
        "save_conditional_vector",
    },
    "CAST_HF": {
        "use_pca",
        "conditional_dataset",
        "conditional_layer",
        "apply_conditional_chat_template",
        "save_conditional_vector",
    },
    "ACT": {"act_mode", "act_std_eps", "pca_components"},
    "IDS": {
        "ids_var_explained",
        "ids_epsilon_pct",
        "ids_f1_threshold",
        "ids_ot_eps",
    },

    "BIPO": {
        "bipo_lr",
        "bipo_beta",
        "bipo_epochs",
        "bipo_vector_path",
    },
    "CURVEBALL": {
        "curveball_kernel",
        "curveball_dim",
        "curveball_degree",
        "curveball_gamma",
        "curveball_coef0",
        "curveball_inverse_alpha",
    },
    "COLD": {"cold_variant", "cold_eta", "cold_epsilon", "cold_pair_margin"},
    "CORRSTEER": {
        "corrsteer_max_new_tokens",
        "corrsteer_pool",
        "corrsteer_steer_pool",
        "corrsteer_pos_only",
        "corrsteer_neg_only",
        "corrsteer_real",
        "corrsteer_decode",
        "corrsteer_raw",
        "corrsteer_reverse",
        "corrsteer_selection",
        "corrsteer_caacoeff",
        "corrsteer_layer_mode",
        "corrsteer_reward_evaluator",
        "corrsteer_prompt_suffix",
    },
    "MANIFOLD": {"manifold_dim"},
    "FLOW": {
        "flow_hidden_dim",
        "flow_layers",
        "flow_lr",
        "flow_epochs",
        "flow_batch_size",
        "flow_seed",
        "flow_subspace_dim",
        "flow_train_space",
        "flow_norm_mode",
        "flow_loss_mode",
        "flow_max_weight",
        "flow_weighted",
        "flow_target_type",
        "flow_ot",
        "flow_lm_loss",
        "flow_lm_lambda",
        "flow_lm_lr",
        "flow_lm_epochs",
        "flow_lm_grad_accum",
        "flow_lm_batch_size",
        "flow_max_new_tokens",
    },
    "LOREFT": {
        "reft_type",
        "reft_low_rank_dimension",
        "dropout",
        "act_fn",
        "add_bias",
        "preference_pairs",
        "substraction_type",
        "steering_factors",
        "grad_accum",
        "reft_seed",
        "lr",
        "weight_decay",
        "epochs",
        "reft_steer_once"
    },
    "REPS": {
        "reft_type",
        "reft_low_rank_dimension",
        "dropout",
        "act_fn",
        "add_bias",
        "preference_pairs",
        "substraction_type",
        "steering_factors",
        "grad_accum",
        "reft_seed",
        "lr",
        "weight_decay",
        "epochs",
        "reft_steer_once"
    },
    "ODE": {
        "classifier_type",
        "degree",
        "n_components",
        "gamma",
        "coef0",
        "sigma",
        "lin_clf_type",
        "solver",
        "steps",
    },
    "PID": {"pid_kp", "pid_ki", "pid_kd", "pid_normalize_error"},
    "CHARS": {
        "chars_k",
        "chars_eps",
        "chars_lambda",
        "chars_tau",
        "chars_max_iter",
        "chars_pct",
        "chars_pct_l",
        "chars_diag",
        "chars_whiten",
        "chars_pca_k",
        "chars_tail_transform",
    },
    "COBRA": {
        "cobra_k",
        "cobra_lambda",
        "cobra_tau",
        "cobra_max_iter",
        "cobra_dim",
    },
    "LinNEAS": {
        "linneas_lr",
        "linneas_steps",
        "linneas_reg_l1",
        "linneas_reg_l2",
        "linneas_optimizer",
        "linneas_proximal",
        "linneas_init_identity",
    },
    "INNSTEER": {
        "inn_n_coupling",
        "inn_hidden_dim",
        "inn_lr",
        "inn_weight_decay",
        "inn_epochs",
        "inn_batch_size",
        "inn_lambda_dir",
        "inn_lambda_logdet",
        "inn_grad_clip",
        "inn_warmup_epochs",
        "inn_checkpoint_dir",
    },
    "LQR": {
        "lqr_Q",
        "lqr_R",
        "lqr_Qf",
        "lqr_jac_chunk_size",
        "lqr_store_jacobians",
    },
    "FLAS": {
        "flas_checkpoint_path",
        "flas_num_blocks",
        "flas_time_conditioned",
        "flas_disable_cross_attn",
        "flas_disable_self_attn",
        "flas_disable_mlp",
        "flas_strict_load",
        "flas_concept_encoder_layers",
        "flas_train_concept_text",
        "flas_train_lr",
        "flas_train_enc_lr",
        "flas_train_div_weight",
        "flas_train_epochs",
        "flas_train_batch_size",
        "flas_train_grad_accum",
        "flas_train_max_len",
        "flas_train_concept_max_len",
        "flas_train_T_min",
        "flas_train_T_max",
        "flas_train_n_steps",
        "flas_train_seed",
        "flas_train_max_steps",
        "flas_unfreeze_concept_enc",
        "flas_no_gemma_init",
        "flas_steer_once",
        "flas_binary_class",
    },
    "FGAA": {
        "fgaa_density_threshold",
        "fgaa_n1",
        "fgaa_n2",
        "fgaa_remove_bos",
        "fgaa_bos_feature_ids",
        "fgaa_effect_matrix_path",
        "fgaa_l1_normalize_target",
    },
    "WEIGHTSTEER": {
        "weight_steer_lr",
        "weight_steer_epochs",
        "weight_steer_target_modules",
        "weight_steer_lora_r",
        "weight_steer_lora_alpha",
        "weight_steer_lora_dropout",
        "weight_steer_lambda_sparse",
        "weight_steer_grad_accum",
        "steering_factors",
    },
    "SPARE": {"loss_weight", "n_neighbors", "top_k_proportion"},
    "SAE-FREE": {
        "saefree_component_idx",
        "saefree_center_diffs",
        "saefree_cov_normalize",
        "saefree_align_sign",
    },
    "SAE-COT": {"saecot_score_mode", "saecot_value_mode", "saecot_max_act"},
    "SAEIO": {"neutral_prompt", "amp_factor"},
    "SAE-RSV": {"alpha1", "alpha2", "alpha3"},
    "SAE-TS": {
        "saets_lr",
        "saets_epochs",
        "saets_bias_scale",
        "saets_adapter_path",
        "saets_seed",
        "saets_effects_n_samples",
        "saets_effects_loss_batch_size",
        "saets_effects_feature_batch_size",
        "saets_effects_baseline_batches",
        "saets_effects_steer_batches",
        "target_features",
    },
    "SRPS": {"beta"},
    "SSV": {
        "ssv_feature_dim",
        "ssv_lambda_dist",
        "ssv_lambda_lm",
        "ssv_lambda_l1",
        "ssv_opt_lr",
        "ssv_opt_steps",
        "ssv_feature_refinement_k",
    },
    "JSPACE": {
        "jspace_mode",
        "jspace_k",
        "jspace_lens_path",
        "jspace_target_token",
        "jspace_dim_batch",
        "jspace_max_seq_len",
    },
    "FISHBACK": {
        "fb_hidden_dim",
        "fb_n_layers",
        "fb_lr",
        "fb_weight_decay",
        "fb_epochs",
        "fb_grad_clip",
        "fb_n_steps",
        "fb_max_grad_len",
    },
    "GINN": {
        "ginn_n_coupling",
        "ginn_hidden_dim",
        "ginn_lr",
        "ginn_weight_decay",
        "ginn_epochs",
        "ginn_batch_size",
        "ginn_lambda_sep",
        "ginn_grad_clip",
    },
    "RIEMANNIAN": {
        "riemannian_steps",
        "riemannian_calpha",
        "riemannian_seed",
        "riemannian_norm_output",
    },
}

STEER_COMMON_FIELDS: Set[str] = {
    "apply_chat_template",
    "hook_point",
    "position",
    "batch_size",
    "norm",
    "apply_all_layers",
    "top_k",
    "steer_once",
}

STEER_METHOD_FIELDS: Dict[str, Set[str]] = {
    "ANGULAR": {"target_angle", "adaptive_mode", "selected_layer"},
    "SPHERICAL": {
        "spherical_kappa",
        "spherical_alpha",
        "spherical_beta",
        "spherical_use_vmf_gate",
    },
    "CHARS": {
        "chars_mode",
        "chars_pct",
        "chars_pct_l",
        "chars_diag",
        "chars_whiten",
        "chars_clip_tail",
        "chars_clip_z",
        "chars_pca_k",
        "chars_tail_transform",
    },
    "COBRA": {
        "cobra_mode",
    },
    "FLOW": {
        "flow_steps",
        "flow_guidance_strength",
        "flow_guidance_mode",
        "flow_denoise_mode",
    },
    "LOREFT": {
        "reft_type",
        "dropout",
        "act_fn",
        "add_bias",
        "substraction_type",
    },
    "REPS": {
        "reft_type",
        "dropout",
        "act_fn",
        "add_bias",
        "substraction_type",
    },
    "ODE": {
        "solver",
        "steps",
        "one_step",
    },
    "ACT": {"act_mode"},
    "IDS": {"ids_f1_threshold"},

    "BIPO": set(),
    "LinNEAS": set(),
    "CAST": {
        "use_conditional",
        "conditional_threshold",
        "conditional_threshold_is",
        "conditional_pos",
        "conditional_vector",
        "load_conditional_vector",
        "apply_to_all_tokens",
        "use_ooi_preventive_normalization",
    },
    "CAST_HF": {
        "use_conditional",
        "conditional_threshold",
        "conditional_threshold_is",
        "conditional_pos",
        "conditional_vector",
        "load_conditional_vector",
        "apply_to_all_tokens",
        "use_ooi_preventive_normalization",
    },
    "CORRSTEER": {"corrsteer_lastk", "corrsteer_subtract"},
    "SAE-FREE": {"saefree_norm_eps"},
    "SAE-COT": {"saecot_overwrite", "saecot_norm_eps"},
    "FEAT": {"feature_list", "max_act", "overwrite_with_max_act", "use_reconstruction_error"},
    "SAE-TS": {"auto_scale", "target_loss_increase"},
    "SPARE": {"target_behavior"},
    "SSV": {"use_important_dims"},
    "FLAS": {
        "flas_concept_text",
        "flas_concept_max_len",
        "flas_n_steps",
        "flas_max_prompt_len",
    },
    "INNSTEER": set(),
    "FISHBACK": set(),  # All steer params come from metadata via to_steer_params()
    "GINN": set(),  # No steer-specific params — uses inn_state_dicts from metadata
    "RIEMANNIAN": {"riemannian_calpha"},
}


# =========================================================================
# EXTRACTOR CONFIG
# =============================================================================

@dataclass
class ExtractorConfig:
    """
    Configuration for steering vector extraction.
    
    This is the SINGLE SOURCE OF TRUTH for all extractor defaults.
    Only params used by the specific method will be passed to the extractor.
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED
    # -------------------------------------------------------------------------
    method: str
    layer: Union[int, List[int], None] = None  # Can be inferred from dict-valued train_dataset
    

    # -------------------------------------------------------------------------
    # COMMON (used by most methods)
    # -------------------------------------------------------------------------
    batch_size: int = 8
    position: Union[str, int] = "last"           # Token position: "last" or "mean" or "all" or int
    apply_chat_template: bool = True
    hook_point: Union[str,List[str]] = "pre"        # Hook point: "pre" or "post"
    change_pad_token: bool = False
    inverse: bool = False
    # -------------------------------------------------------------------------
    # ANGULAR - 2D rotation steering
    # -------------------------------------------------------------------------
    # use_normalized: bool = True
    # exclude_last_layer: bool = True
    # auto_select_layer: bool = True
    strategy: str = "max_sim"            # "max_sim" or "max_norm"
    # -------------------------------------------------------------------------
    # CAST - Contrastive Activation Addition
    # -------------------------------------------------------------------------
    use_pca: bool = False

    # -------------------------------------------------------------------------
    # ACT - Activation Transport / Mean-AcT
    # -------------------------------------------------------------------------
    act_mode: str = "linear"          # "mean", "linear", "gaussian", or "pca_ot"
    act_std_eps: float = 1e-4
    pca_components: int = 16

    # -------------------------------------------------------------------------
    # IDS - In-Distribution Steering
    # -------------------------------------------------------------------------
    ids_var_explained: float = 0.40
    ids_epsilon_pct: float = 0.95
    ids_f1_threshold: float = 0.70
    ids_ot_eps: float = 1e-6


    # -------------------------------------------------------------------------
    # CURVEBALL - Polynomial Kernel PCA steering
    # -------------------------------------------------------------------------
    curveball_kernel: str = "rbf"
    curveball_dim: int = 8
    curveball_degree: int = 2
    curveball_gamma: Optional[float] = None
    curveball_coef0: float = 1.0
    curveball_inverse_alpha: float = 1e-3

    # -------------------------------------------------------------------------
    # COLD - In-context One-step Learning Dynamics
    # -------------------------------------------------------------------------
    cold_variant: str = "fd"       # "fd" or "kernel"
    cold_eta: float = 1.0           # Paper default steering multiplier
    cold_epsilon: float = 1e-6      # Paper default finite-difference epsilon
    cold_pair_margin: float = 0.0
    
    # CAST-specific (conditional steering)
    conditional_dataset: Optional[str] = None
    conditional_layer: Optional[int] = None
    apply_conditional_chat_template: bool = True
    save_conditional_vector: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # CORRSTEER - Generation-time correlation steering
    # Reference: Code/CorrSteer/train.py CorrConfig
    #
    # Removed (use shared pipeline params instead):
    #   - corrsteer_scale          → steer.coeff (SteerConfig) at steer time; extraction always uses 1.0
    # -------------------------------------------------------------------------
    corrsteer_max_new_tokens: int = 50
    corrsteer_pool: str = "max"       # Pooling for correlation computation (default: max)
    corrsteer_steer_pool: str = "max" # Pooling for coefficient computation (default: max)
    corrsteer_pos_only: bool = True   # Only positive correlations (default)
    corrsteer_neg_only: bool = False  # Only negative correlations
    corrsteer_real: bool = False      # Mean over ACTIVE features only
    corrsteer_decode: bool = False    # Use sae.decode() vs @ W_dec
    corrsteer_raw: bool = False       # SAE-free residual mode (no SAE encoding)
    corrsteer_reverse: bool = False   # Invert rewards (1→0, 0→1)
    corrsteer_selection: str = "correlation"  # Feature selection: correlation/mi/fisher/caa
    corrsteer_caacoeff: bool = False  # Contrastive coefficient (success-failure mean)
    corrsteer_layer_mode: str = "foreach" # Enable global layer mode ("foreach" or "global")
    corrsteer_reward_evaluator: Optional[str] = None  # Reward evaluator for coefficient scaling (e.g. "cast_refusal")
    corrsteer_prompt_suffix: Optional[str] = None  # Optional prompt suffix for correlation evaluation (e.g. "Between A and B, the answer is: (")
    # -------------------------------------------------------------------------
    # MANIFOLD - Manifold-based steering
    # -------------------------------------------------------------------------
    manifold_dim: int = 10

    # -------------------------------------------------------------------------
    # FLOW / FLOW / TRUTHFLOW - Flow Matching steering
    # -------------------------------------------------------------------------
    flow_hidden_dim: int = 512
    flow_layers: int = 6
    flow_lr: float = 1e-3
    flow_epochs: int = 200
    flow_batch_size: int = 64
    flow_seed: int = 42
    flow_subspace_dim: Optional[int] = None
    flow_train_space: str = "full"   # full | pca_diff | pca_stack | lda
    flow_norm_mode: str = "iqr"
    flow_loss_mode: str = "huber"
    flow_max_weight: Optional[float] = None
    flow_weighted: bool = False
    flow_target_type: str = "concept"  # concept | correction
    flow_ot: Optional[str] = None      # None or "sinkhorn"
    flow_lm_loss: bool = False
    flow_lm_lambda: float = 0.1
    flow_lm_lr: float = 5e-5
    flow_lm_epochs: int = 3
    flow_lm_grad_accum: int = 4
    flow_lm_batch_size: int = 4
    flow_max_new_tokens: int = 30

    # -------------------------------------------------------------------------
    # ODE - Kernel classifier steering
    # -------------------------------------------------------------------------
    classifier_type: str = "normed_poly"
    degree: int = 2
    n_components: int = 100
    gamma: float = 1.0
    coef0: float = 0.1
    sigma: Union[float, str] = "median"
    lin_clf_type: str = "lr"
    solver: str = "euler"
    steps: int = 10

    # -------------------------------------------------------------------------
    # BIPO - Bilateral Preference Optimization
    # -------------------------------------------------------------------------
    bipo_lr: float = 5e-4
    bipo_beta: float = 0.1
    bipo_epochs: int = 5
    bipo_vector_path: Optional[str] = None

    # -------------------------------------------------------------------------
    # INNSTEER - Invertible Neural Network Steering
    # -------------------------------------------------------------------------
    inn_n_coupling: int = 4
    inn_hidden_dim: int = 512
    inn_lr: float = 1e-3
    inn_weight_decay: float = 1e-3
    inn_epochs: int = 300
    inn_batch_size: int = 64
    inn_lambda_dir: float = 1.0
    inn_lambda_logdet: float = 0.5
    inn_grad_clip: float = 1.0
    inn_warmup_epochs: int = 60
    inn_checkpoint_dir: Optional[str] = None

    # -------------------------------------------------------------------------
    # FISHBACK - Pullback Fisher Geometry Steering
    # -------------------------------------------------------------------------
    # LQR - Activation Linear Quadratic Regulator
    # -------------------------------------------------------------------------
    lqr_Q: float = 1.0
    lqr_R: float = 1.0
    lqr_Qf: float = 1.0
    lqr_jac_chunk_size: int = 64
    lqr_store_jacobians: bool = False

    # -------------------------------------------------------------------------
    # JSPACE - Jacobian-lens / J-space steering
    # -------------------------------------------------------------------------
    jspace_mode: str = "project"             # "project" or "direct"
    jspace_k: int = 16                       # Sparsity level (number of J-lens vectors)
    jspace_lens_path: Optional[str] = None    # Path to pre-computed J_ℓ matrices (default Vector/JSPACE/lens.pt)
    jspace_target_token: Optional[str] = None # Token name for "direct" mode
    jspace_dim_batch: int = 8                # Output dims per backward pass
    jspace_max_seq_len: int = 128            # Max prompt length for fitting

    # -------------------------------------------------------------------------
    # LinNEAS - Linearized Non-linear End-to-end Activation Steering
    # -------------------------------------------------------------------------
    linneas_lr: float = 0.1
    linneas_steps: int = 1000
    linneas_reg_l1: float = 0.0
    linneas_reg_l2: float = 0.0
    linneas_optimizer: str = "SGD"
    linneas_proximal: Optional[str] = None
    linneas_init_identity: bool = True

    # -------------------------------------------------------------------------
    # REFT / LOREFT / REPS - Low-rank residual steering
    # -------------------------------------------------------------------------
    reft_type: str = "Loreft"
    reft_low_rank_dimension: int = 8
    dropout: float = 0.0
    act_fn: str = "linear"
    add_bias: bool = True
    preference_pairs: Optional[List[str]] = None
    substraction_type: str = "zero"
    steering_factors: Optional[List[float]] = None
    grad_accum: int = 8
    lr: float = 5e-3
    weight_decay: float = 0.0
    epochs: int = 3
    reft_seed: int = 42  # Global seed for REPS/LoReFT training (controls init + shuffling)
    reft_steer_once: bool = True

    # -------------------------------------------------------------------------
    # OT-REPS - Gaussian OT in REPS subspace
    # -------------------------------------------------------------------------
    ot_eps: float = 1e-6


    # -------------------------------------------------------------------------
    # CHARS - Concept Heterogeneity-aware Representation Steering
    # -------------------------------------------------------------------------
    chars_k: Union[int, str] = 10
    chars_eps: Optional[float] = None
    chars_lambda: float = 0.1
    chars_tau: float = 1e-4
    chars_max_iter: int = 1000
    chars_pct: bool = False
    chars_pct_l: int = 4
    chars_diag: bool = False
    chars_whiten: bool = False
    chars_pca_k: int = 0
    chars_tail_transform: str = "none"

    # -------------------------------------------------------------------------
    # COBRA - Cluster-Optimized Barycentric Representation Alignment
    # -------------------------------------------------------------------------
    cobra_k: int = 10
    cobra_lambda: float = 0.1
    cobra_tau: float = 1e-4
    cobra_max_iter: int = 1000
    cobra_dim: int = 8

    # -------------------------------------------------------------------------
    # PID - PID steering-vector construction
    # -------------------------------------------------------------------------
    pid_kp: float = 1.0
    pid_ki: float = 0.005
    pid_kd: float = 0.0
    pid_normalize_error: bool = False

    # -------------------------------------------------------------------------
    # FLAS - Flow-based Activation Steering
    # -------------------------------------------------------------------------
    flas_checkpoint_path: Optional[str] = None
    flas_num_blocks: Optional[int] = 1
    flas_time_conditioned: bool = True
    flas_disable_cross_attn: bool = False
    flas_disable_self_attn: bool = False
    flas_disable_mlp: bool = False
    flas_strict_load: bool = True
    flas_concept_encoder_layers: int = 2
    flas_train_concept_text: Union[str, List[str], None] = None
    flas_train_lr: float = 5e-5
    flas_train_enc_lr: float = 1e-5
    flas_train_div_weight: float = 0.1
    flas_train_epochs: int = 1
    flas_train_batch_size: int = 4
    flas_train_grad_accum: int = 8
    flas_train_max_len: int = 256
    flas_train_concept_max_len: int = 64
    flas_train_T_min: float = 0.5
    flas_train_T_max: float = 2.0
    flas_train_n_steps: int = 3
    flas_train_seed: int = 42
    flas_train_max_steps: Optional[int] = 80000
    flas_unfreeze_concept_enc: bool = False
    flas_no_gemma_init: bool = False
    flas_steer_once: bool = True
    flas_binary_class: bool = False

    # -------------------------------------------------------------------------
    # FGAA - Feature Guided Activation Additions
    # -------------------------------------------------------------------------
    fgaa_density_threshold: float = 0.01
    fgaa_n1: int = 8
    fgaa_n2: int = 0
    fgaa_remove_bos: bool = True
    fgaa_bos_feature_ids: Optional[List[int]] = None
    fgaa_effect_matrix_path: Optional[str] = None
    fgaa_l1_normalize_target: bool = True
    
    # -------------------------------------------------------------------------
    # WEIGHTSTEER - Contrastive Weight Steering
    # -------------------------------------------------------------------------
    weight_steer_lr: float = 1e-4
    weight_steer_epochs: int = 3
    weight_steer_target_modules: Optional[List[str]] = None
    weight_steer_lora_r: int = 32
    weight_steer_lora_alpha: int = 64
    weight_steer_lora_dropout: float = 0.05
    weight_steer_lambda_sparse: float = 0.0
    weight_steer_grad_accum: int = 8
    
    # -------------------------------------------------------------------------
    # SAE COMMON - Shared by SAE-based methods
    # -------------------------------------------------------------------------
    top_k: int = 15                  # Number of top features
    act_threshold: float = 1e-5     # Activation threshold
    act_frac: float = 0.01
    feature_dim: int = 128

    # -------------------------------------------------------------------------
    # SPARE - Loss-derived weighting switch
    # -------------------------------------------------------------------------
    loss_weight: bool = False
    n_neighbors: int = 5            # k-NN neighbors for mutual_info_classif
    top_k_proportion: float = 0.01
    # -------------------------------------------------------------------------
    # SAE-FREE - Eigendecomposition of activation differences
    # -------------------------------------------------------------------------
    saefree_component_idx: int = -1  # Eigenvector index (-1 = largest eigenvalue)
    saefree_center_diffs: bool = False   # Center pairwise diffs before covariance
    saefree_cov_normalize: bool = False  # Divide covariance by n_pairs
    saefree_align_sign: bool = True      # Resolve eigenvector sign ambiguity via mean-diff alignment

    # -------------------------------------------------------------------------
    # SAE-COT - GT SAE latent feature overwrite steering
    # -------------------------------------------------------------------------
    saecot_score_mode: str = "target_minus_contrast"  # Reserved for future scoring variants.
    saecot_value_mode: str = "target_mean"            # Current supported modes: target_mean | selected
    saecot_max_act: float = 1.0                        # Used when value_mode=selected
    
    # -------------------------------------------------------------------------
    # SAE-IO - Input/Output filtering
    # -------------------------------------------------------------------------
    neutral_prompt: str = "From my experience,"
    amp_factor: float = 10.0
    
    # -------------------------------------------------------------------------
    # SAE-RSV - Residual Steering Vector
    # -------------------------------------------------------------------------
    alpha1: float = 1.0
    alpha2: float = 0.5
    alpha3: float = 0.5
    
    # -------------------------------------------------------------------------
    # SAE-TS - Task-Specific LinearAdapter
    # -------------------------------------------------------------------------
    saets_lr: float = 2e-4           # Training learning rate (GT train.py default)
    saets_epochs: int = 15           # Training epochs
    saets_bias_scale: float = 1.0    # Correction-bias scale in SAE-TS steering formula
    saets_adapter_path: Optional[str] = None  # Adapter checkpoint path (load/save)
    saets_seed: Optional[int] = 42   # Training seed for reproducibility
    saets_effects_n_samples: int = 512
    saets_effects_loss_batch_size: int = 64
    saets_effects_feature_batch_size: int = 32
    saets_effects_baseline_batches: int = 10
    saets_effects_steer_batches: int = 1
    target_features: Optional[Dict[int, List[Tuple[int, float]]]] = None  # Full weighted target vector
    
    # -------------------------------------------------------------------------
    # SRPS - Sparse Representation Projection Steering
    # -------------------------------------------------------------------------
    beta: float = 1.0
    
    # -------------------------------------------------------------------------
    # SSV - Supervised Steering Vector (two-stage optimization)
    # -------------------------------------------------------------------------
    ssv_feature_dim: int = 128
    ssv_lambda_dist: float = 1.0     # Distance loss weight
    ssv_lambda_lm: float = 0.5       # Language model loss weight
    ssv_lambda_l1: float = 0.01      # L1 sparsity regularization
    ssv_opt_lr: float = 0.05         # Learning rate
    ssv_opt_steps: int = 100         # Optimization steps
    ssv_feature_refinement_k: int = 30   # Classifier-based feature refinement top-k

    # -------------------------------------------------------------------------
    # FISHBACK - Gradient-Guided Flow Steering (VGG-Flow style)
    # -------------------------------------------------------------------------
    fb_hidden_dim: int = 128  # Hidden dimension for flow MLP
    fb_n_layers: int = 2  # Number of MLP layers (matches FlowMLP)
    fb_lr: float = 1e-3  # Learning rate
    fb_weight_decay: float = 1e-4  # Weight decay
    fb_epochs: int = 100  # Training epochs
    fb_grad_clip: float = 1.0  # Gradient clipping norm
    fb_n_steps: int = 10  # Euler integration steps at inference
    fb_max_grad_len: int = 32  # Maximum target response tokens for target gradient computation

    # -------------------------------------------------------------------------
    # GINN - Contrastive Invertible Mapping (extractor params)
    # -------------------------------------------------------------------------
    ginn_n_coupling: int = 4  # Number of affine coupling layers
    ginn_hidden_dim: int = 512  # Hidden dimension in coupling MLPs
    ginn_lr: float = 1e-3  # Learning rate
    ginn_weight_decay: float = 1e-4  # Weight decay
    ginn_epochs: int = 100  # Training epochs
    ginn_batch_size: int = 64  # Batch size for training
    ginn_lambda_sep: float = 2.0  # Contrastive separation loss weight
    ginn_grad_clip: float = 1.0  # Gradient clipping norm

    # -------------------------------------------------------------------------
    # RIEMANNIAN - Riemannian Activation Steering (SPREAD, AAAI 2026)
    # -------------------------------------------------------------------------
    riemannian_steps: int = 50        # Block-coordinate Riemannian descent iterations
    riemannian_calpha: float = 1.0    # Scale constant for per-sample sphere radius
    riemannian_seed: int = 0          # Random seed for reproducibility
    riemannian_norm_output: bool = True  # L2-normalize the final steering vector

    def to_dict(self, include_none: bool = False, method_scoped: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not method_scoped:
            return data if include_none else {key: value for key, value in data.items() if value is not None}

        method = data.get("method", "")
        keep_fields: Set[str] = {"method", "layer"}
        keep_fields.update(EXTRACTOR_COMMON_FIELDS)
        keep_fields.update(EXTRACTOR_METHOD_FIELDS.get(method, set()))
        if method in SAE_METHODS:
            keep_fields.update(EXTRACTOR_SAE_COMMON_FIELDS)

        return {k: v for k, v in data.items() if k in keep_fields and (include_none or v is not None)}
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractorConfig":
        """Create from dict, warning on unknown keys."""
        data = dict(data)

        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            warnings.warn(
                f"ExtractorConfig: ignoring unknown keys {sorted(unknown)}. "
                f"Check for typos or stale config fields.",
                stacklevel=2,
            )
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# STEER CONFIG
# =============================================================================

@dataclass
class SteerConfig:
    """
    Configuration for steering application during generation.
    
    This is the SINGLE SOURCE OF TRUTH for all steer defaults.
    Only params used by the specific method will be passed to the steer model.
    """
    
    # -------------------------------------------------------------------------
    # REQUIRED
    # -------------------------------------------------------------------------
    method: str
    layer: Union[int, List[int], None] = None  # Can be inferred from dict-valued coeff
    coeff: Union[float, Dict, List] = 1.0      # float, {layer: coeff}, or [coeff_per_layer]
    
    # -------------------------------------------------------------------------
    # COMMON
    # -------------------------------------------------------------------------
    apply_chat_template: bool = True
    hook_point: Union[str,List[str]] = "pre"    
    position: Union[int, str] = "last"  # token position for methods that support token selection
    batch_size: int = 8              # Batch size for evaluation
    norm: bool = False               # Whether to L2-normalize the steering vector
    apply_all_layers: bool = False    # Used by methods that install hooks globally (e.g. ANGULAR, MANIFOLD)
    top_k: Union[int, List[int], None] = None   # SAE feature selection: int->first k ranks, list->explicit ranks, None->pass through extractor selection
    steer_once: bool = False                   # Whether to apply steering only once at the last prompt token

    
    # -------------------------------------------------------------------------
    # ANGULAR
    # -------------------------------------------------------------------------
    target_angle: float = 90.0
    adaptive_mode: int = 1            # 1=standard adaptive (mask>0), 0=non-adaptive, etc.
    selected_layer: int = 15

    # -------------------------------------------------------------------------
    # SPHERICAL
    # -------------------------------------------------------------------------
    spherical_kappa: float = 20.0
    spherical_alpha: float = 0.7
    spherical_beta: float = -0.15
    spherical_use_vmf_gate: bool = True

    # -------------------------------------------------------------------------
    # FLOW / TRUTHFLOW
    # -------------------------------------------------------------------------
    flow_steps: int = 16
    flow_guidance_strength: float = 0.0
    flow_guidance_mode: str = "fixed"
    flow_denoise_mode: str = "none"

    # -------------------------------------------------------------------------
    # ODE - Runtime steering controls
    # -------------------------------------------------------------------------
    solver: str = "euler"
    steps: int = 10
    one_step: bool = False

    # -------------------------------------------------------------------------
    # SUBSPACE GLP / FLOWGLP - Inference-time steering
    # -------------------------------------------------------------------------
    subspace_source: Optional[str] = None
    subspace_checkpoint: str = "final"
    subspace_local_files_only: Optional[bool] = None
    timesteps: int = 20

    # -------------------------------------------------------------------------
    # CAST (conditional steering)
    # -------------------------------------------------------------------------
    use_conditional: bool = True
    conditional_threshold: float = 0.0
    conditional_threshold_is: str = "larger"   # CAST default: "larger" = sim < thresh triggers steering
    conditional_pos: str = "mean"             # CAST default: mean over sequence for condition checking
    conditional_vector: Optional[torch.Tensor] = None
    load_conditional_vector: Optional[str] = None
    apply_to_all_tokens: bool = True        # CAST: apply steering to all sequence positions
    use_ooi_preventive_normalization: bool = False # CAST: prevent norm growth
    
    # -------------------------------------------------------------------------
    # CORRSTEER
    # Reference: Code/CorrSteer/corrsteer/steer.py SteeringHook
    # -------------------------------------------------------------------------
    corrsteer_lastk: int = 1          # Number of last tokens to steer (default: 1)
    corrsteer_subtract: bool = False  # Subtract steering vector (for negative corr)

    # -------------------------------------------------------------------------
    # SAE-FREE
    # -------------------------------------------------------------------------
    saefree_norm_eps: float = 1e-8             # Numerical stability for renormalization

    # -------------------------------------------------------------------------
    # SAE-COT
    # -------------------------------------------------------------------------
    saecot_overwrite: bool = True
    saecot_norm_eps: float = 1e-8

    # -------------------------------------------------------------------------
    # FEAT - Feature testing in SAE latent space
    # -------------------------------------------------------------------------
    feature_list: Optional[Union[List[int], Dict[int, List[int]]]] = None
    max_act: Optional[float] = None
    overwrite_with_max_act: bool = False
    use_reconstruction_error: bool = True
    
    # -------------------------------------------------------------------------
    # SAE-TS
    # -------------------------------------------------------------------------
    auto_scale: bool = True
    target_loss_increase: float = 0.5
    
    # -------------------------------------------------------------------------
    # SPARE
    # -------------------------------------------------------------------------
    target_behavior: str = "contextual"
    
    # -------------------------------------------------------------------------
    # SSV
    # -------------------------------------------------------------------------
    use_important_dims: bool = True

    # -------------------------------------------------------------------------
    # FLAS
    # -------------------------------------------------------------------------
    flas_concept_text: Optional[str] = None
    flas_concept_max_len: int = 64
    flas_n_steps: int = 2
    flas_max_prompt_len: int = 512

    # -------------------------------------------------------------------------
    # REFT / LOREFT / REPS - Low-rank residual steering
    # -------------------------------------------------------------------------
    reft_type: str = "Loreft"
    dropout: float = 0.0
    act_fn: str = "linear"
    add_bias: bool = True
    substraction_type: str = "zero"

    # -------------------------------------------------------------------------
    # CHARS - Concept Heterogeneity-aware Representation Steering
    # -------------------------------------------------------------------------
    chars_mode: str = "addition"
    chars_pct: bool = False
    chars_pct_l: int = 4
    chars_diag: bool = False
    chars_whiten: bool = False
    chars_clip_tail: bool = False
    chars_clip_z: float = 3.0
    chars_pca_k: int = 0
    chars_tail_transform: str = "none"

    # -------------------------------------------------------------------------
    # COBRA - Cluster-Optimized Barycentric Representation Alignment
    # -------------------------------------------------------------------------
    cobra_mode: str = "manifold"

    # -------------------------------------------------------------------------
    # RIEMANNIAN - Exponential map on sphere manifold
    # -------------------------------------------------------------------------
    riemannian_calpha: float = 1.0  # Sphere radius scale: r_k = sqrt(calpha * ||h_k|| / D)

    # -------------------------------------------------------------------------
    # ACT
    # -------------------------------------------------------------------------

    act_support: str = "q_all"        # "q_all" or support-bounded transport

    # -------------------------------------------------------------------------
    # IDS - In-Distribution Steering
    # -------------------------------------------------------------------------
    ids_f1_threshold: float = 0.70

    
    def to_dict(self, include_none: bool = False, method_scoped: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not method_scoped:
            return data if include_none else {key: value for key, value in data.items() if value is not None}

        method = data.get("method", "")
        keep_fields: Set[str] = {"method", "layer", "coeff"}
        keep_fields.update(STEER_COMMON_FIELDS)
        keep_fields.update(STEER_METHOD_FIELDS.get(method, set()))

        return {k: v for k, v in data.items() if k in keep_fields and (include_none or v is not None)}
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SteerConfig":
        """Create from dict, warning on unknown keys."""
        data = dict(data)

        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            warnings.warn(
                f"SteerConfig: ignoring unknown keys {sorted(unknown)}. "
                f"Check for typos or stale config fields.",
                stacklevel=2,
            )
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
