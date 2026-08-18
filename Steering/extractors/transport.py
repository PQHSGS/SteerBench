"""Transport-based extractors: ACT (with PCA-OT), CHARS, LinNEAS."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

from ..base import BaseExtractor
from ..logger import setup_logger
from ..utils import collect_dense_activations, get_hook_name, get_resid_acts, set_resid_acts

logger = setup_logger(__name__)


# =============================================================================
# Helper functions
# =============================================================================

def _match_pairs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[0], b.shape[0])
    if n <= 0:
        raise ValueError("Need at least one paired activation sample")
    return a[:n].to(torch.float32), b[:n].to(torch.float32)


def _sorted_affine_transport(
    src: torch.Tensor,
    dst: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_sorted = torch.sort(src, dim=0).values
    dst_sorted = torch.sort(dst, dim=0).values
    src_mean = src_sorted.mean(dim=0)
    dst_mean = dst_sorted.mean(dim=0)
    src_centered = src_sorted - src_mean
    dst_centered = dst_sorted - dst_mean
    denom = (src_centered.square()).sum(dim=0).clamp(min=eps)
    omega = (src_centered * dst_centered).sum(dim=0) / denom
    beta = dst_mean - omega * src_mean
    return omega, beta


def _matrix_sqrt(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


def _matrix_sqrt_inv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T


def _gaussian_ot_map(
    src_mean: torch.Tensor,
    src_cov: torch.Tensor,
    dst_mean: torch.Tensor,
    dst_cov: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    cov_src_inv_sqrt = _matrix_sqrt_inv(src_cov, eps)
    inner = cov_src_inv_sqrt @ dst_cov @ cov_src_inv_sqrt
    inner_sqrt = _matrix_sqrt(inner, eps)
    A = cov_src_inv_sqrt @ inner_sqrt @ cov_src_inv_sqrt
    b = dst_mean - A @ src_mean
    return A, b


# =============================================================================
# Activation Transport (ACT) — includes PCA-OT as mode
# =============================================================================

class ActivationTransportExtractor(BaseExtractor):
    """Extractor for Activation Transport / Mean-AcT statistics.

    Supports modes:
      - ``"mean"``: simple mean shift
      - ``"gaussian"``: per-dimension Gaussian (std ratio + mean shift)
      - ``"linear"``: per-dimension sorted affine transport
      - ``"pca_ot"``: PCA subspace + full Gaussian OT (PCA-OT)
    """

    METHOD_NAME = "ACT"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "mean",
        hook_point: List[str] | str = "post",
        act_mode: str = "linear",
        act_std_eps: float = 1e-4,
        pca_components: int = 2,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.act_mode = act_mode
        self.act_std_eps = float(act_std_eps)
        self.pca_components = pca_components

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=kwargs.get("layers", self.layer),
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def _compute_pca_ot(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, d = src.shape
        pooled = torch.cat([src, dst], dim=0)
        mu_pool = pooled.mean(dim=0)
        centered = pooled - mu_pool

        K = min(self.pca_components, n + n - 1, d)
        centered_np = centered.cpu().numpy()
        pca = PCA(n_components=K)
        pca.fit(centered_np)

        P = torch.from_numpy(pca.components_.T).to(device=src.device, dtype=torch.float32)

        src_proj = (src - mu_pool) @ P
        dst_proj = (dst - mu_pool) @ P

        mu_src_k = src_proj.mean(dim=0)
        mu_dst_k = dst_proj.mean(dim=0)
        cov_src_k = (src_proj - mu_src_k).T @ (src_proj - mu_src_k) / (n - 1)
        cov_dst_k = (dst_proj - mu_dst_k).T @ (dst_proj - mu_dst_k) / (n - 1)

        A, b = _gaussian_ot_map(mu_src_k, cov_src_k, mu_dst_k, cov_dst_k, eps=self.act_std_eps)
        return P.cpu(), A.cpu(), b.cpu(), mu_pool.cpu()

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("ActivationTransportExtractor requires contrast_data as source distribution")

        self.layer = sorted(self.layer)
        vectors: Dict[int, torch.Tensor] = {}
        stats: Dict[int, Dict[str, torch.Tensor]] = {}
        act_mode = self.act_mode.strip().lower()
        is_pcaot = (act_mode == "pca_ot")

        def make_incremental_hook(layer_stats, mode):
            def hook_fn(resid, hook):
                mu_src = layer_stats["mu_src"].to(resid.device)
                mu_dst = layer_stats["mu_dst"].to(resid.device)
                omega = layer_stats.get("omega")
                beta = layer_stats.get("beta")
                acts = get_resid_acts(resid, "last")
                dtype = acts.dtype
                x = acts.to(torch.float32)
                if omega is not None and beta is not None:
                    omega = omega.to(resid.device)
                    beta = beta.to(resid.device)
                    transported = omega * x + beta
                else:
                    if mode in {"mean", "mean_ot", "mean-act"}:
                        transported = x - mu_src + mu_dst
                    else:
                        std_src = layer_stats["std_src"].to(resid.device).clamp(min=self.act_std_eps)
                        std_dst = layer_stats["std_dst"].to(resid.device).clamp(min=self.act_std_eps)
                        transported = (std_dst / std_src) * (x - mu_src) + mu_dst
                updated = transported
                return set_resid_acts(resid, "last", updated.to(dtype))
            return hook_fn

        hp_list = [self.hook_point] if isinstance(self.hook_point, (str, bytes)) else list(self.hook_point)

        try:
            for layer in self.layer:
                target_acts = self._get_activations(target_data, layers=[layer])
                source_acts = self._get_activations(contrast_data, layers=[layer])
                src, dst = _match_pairs(source_acts[layer], target_acts[layer])
                mu_src = src.mean(dim=0)
                mu_dst = dst.mean(dim=0)
                src_std = src.std(dim=0, unbiased=False).clamp(min=self.act_std_eps)
                dst_std = dst.std(dim=0, unbiased=False).clamp(min=self.act_std_eps)
                src_min = src.min(dim=0).values
                src_max = src.max(dim=0).values

                if is_pcaot:
                    P, A, b, mu_pool = self._compute_pca_ot(src, dst)
                    vectors[layer] = torch.zeros(src.shape[1], device=self.device)
                    stats[layer] = {
                        "pca_components": P,
                        "transport_matrix": A,
                        "transport_bias": b,
                        "pooled_mean": mu_pool,
                    }
                elif act_mode == "mean":
                    omega = torch.ones_like(mu_src)
                    beta = mu_dst - mu_src
                    vectors[layer] = (omega * mu_src + beta - mu_src).to(self.device)
                    stats[layer] = {
                        "mu_src": mu_src.cpu(), "mu_dst": mu_dst.cpu(),
                        "std_src": src_std.cpu(), "std_dst": dst_std.cpu(),
                        "omega": omega.cpu(), "beta": beta.cpu(),
                        "support_min": src_min.cpu(), "support_max": src_max.cpu(),
                    }
                elif act_mode == "gaussian":
                    omega = dst_std / src_std
                    beta = mu_dst - omega * mu_src
                    vectors[layer] = (omega * mu_src + beta - mu_src).to(self.device)
                    stats[layer] = {
                        "mu_src": mu_src.cpu(), "mu_dst": mu_dst.cpu(),
                        "std_src": src_std.cpu(), "std_dst": dst_std.cpu(),
                        "omega": omega.cpu(), "beta": beta.cpu(),
                        "support_min": src_min.cpu(), "support_max": src_max.cpu(),
                    }
                elif act_mode == "linear":
                    omega, beta = _sorted_affine_transport(src, dst, eps=self.act_std_eps)
                    vectors[layer] = (omega * mu_src + beta - mu_src).to(self.device)
                    stats[layer] = {
                        "mu_src": mu_src.cpu(), "mu_dst": mu_dst.cpu(),
                        "std_src": src_std.cpu(), "std_dst": dst_std.cpu(),
                        "omega": omega.cpu(), "beta": beta.cpu(),
                        "support_min": src_min.cpu(), "support_max": src_max.cpu(),
                    }
                else:
                    raise ValueError(f"Unsupported act_mode '{self.act_mode}'")

                if not is_pcaot:
                    for hp in hp_list:
                        hook_name = get_hook_name(layer, hp)
                        self.model.add_hook(
                            hook_name,
                            make_incremental_hook(stats[layer], act_mode)
                        )
        finally:
            self.model.reset_hooks()

        self.vector = vectors
        self.metadata = {
            "method": "ACT",
            "act_stats": stats,
        }
        return self.vector


# =============================================================================
# CHARS - Concept Heterogeneity-aware Representation Steering
# =============================================================================

class CHARSExtractor(BaseExtractor):
    """Concept Heterogeneity-aware Representation Steering (CHARS) Extractor."""

    METHOD_NAME = "CHARS"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        position: str = "last",
        device: Optional[torch.device] = None,
        hook_point: List[str] = ["pre"],
        chars_k: Union[int, str] = 10,
        chars_eps: Optional[float] = None,
        chars_lambda: float = 0.1,
        chars_tau: float = 1e-4,
        chars_max_iter: int = 1000,
        chars_pct: bool = False,
        chars_pct_l: int = 4,
        chars_diag: bool = False,
        chars_whiten: bool = False,
        chars_pca_k: int = 0,
        chars_tail_transform: str = "none",
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.chars_k = chars_k
        self.chars_eps = chars_eps
        self.chars_lambda = chars_lambda
        self.chars_tau = chars_tau
        self.chars_max_iter = chars_max_iter
        self.chars_pct = chars_pct
        self.chars_pct_l = chars_pct_l
        self.chars_diag = chars_diag
        self.chars_whiten = chars_whiten
        self.chars_pca_k = chars_pca_k
        self.chars_tail_transform = chars_tail_transform

    def _get_activations(self, inputs: List[str]) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
            pretokenize_all=False,
            change_pad_token=False,
        )

    @staticmethod
    def _dp_means(
        X: torch.Tensor,
        eps: Optional[float] = None,
        max_iter: int = 100,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        DP-Means: non-parametric clustering (Dirichlet Process limit of K-means).

        Derives from the Beta-Bernoulli foundation of the Dirichlet Process:
        the DP stick-breaking uses Beta(1, alpha) priors (Gundersen, 2020).
        When cluster variance sigma^2 -> 0 in a DP mixture model, the Gibbs
        sampler converges to DP-Means (Kulis & Jordan, 2012). The `eps`
        parameter = alpha * sigma^2 controls cluster creation.

        Args:
            X: (N, D) data tensor.
            eps: distance threshold. If None, uses median pairwise distance
                 scaled by 0.15 (fallback, only used directly, not via auto).
            max_iter: maximum iterations.
            seed: random seed for subset sampling.

        Returns:
            centroids: (K, D) tensor of cluster centers.
            labels: (N,) integer cluster assignments.
            counts: (K,) number of points per cluster.
        """
        X_np = X.cpu().numpy()
        N, D = X_np.shape

        # Auto-compute eps from median pairwise distance
        if eps is None:
            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(N, min(500, N), replace=False)
            sub = X_np[sub_idx]
            pw = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
            triu = np.triu(pw, k=1)
            nonzero = triu[triu > 0]
            eps = float(np.median(nonzero) * 0.15) if len(nonzero) > 0 else 1.0

        centroids = [X_np[0].copy()]
        labels = np.full(N, -1, dtype=np.int64)

        for iteration in range(max_iter):
            old_labels = labels.copy()
            K = len(centroids)
            centers_arr = np.stack(centroids)

            # E-step: assign each point, potentially creating new clusters
            for i in range(N):
                dists = np.linalg.norm(centers_arr - X_np[i], axis=1)
                min_dist = float(dists.min())
                if min_dist > eps:
                    centroids.append(X_np[i].copy())
                    labels[i] = len(centroids) - 1
                else:
                    labels[i] = int(dists.argmin())

            K_new = len(centroids)
            new_centroids = []
            for k in range(K_new):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids.append(X_np[mask].mean(axis=0))
                else:
                    new_centroids.append(np.zeros(D))
            centroids = new_centroids

            if np.array_equal(labels, old_labels):
                break

        centroids_arr = np.stack(centroids)
        counts = np.bincount(labels, minlength=len(centroids_arr))
        alive = counts > 0
        centroids_arr = centroids_arr[alive]
        label_map = {old: new for new, old in enumerate(np.where(alive)[0])}
        labels = np.array([label_map[l] for l in labels])
        counts = counts[alive]

        return (
            torch.from_numpy(centroids_arr).float(),
            labels,
            counts,
        )

    @staticmethod
    def _elbow_search(
        X: torch.Tensor,
        grid_size: int = 20,
        seed: int = 42,
        max_iter: int = 100,
    ) -> float:
        """
        Find optimal eps via elbow search on K(eps) curve.

        Grids eps from 5th to 95th percentile of pairwise distances,
        runs DP-Means at each, picks eps at max curvature of K(eps).

        Returns:
            Optimal eps value.
        """
        X_np = X.cpu().numpy()
        N = X_np.shape[0]
        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(N, min(500, N), replace=False)
        sub = X_np[sub_idx]
        pw = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        triu = np.triu(pw, k=1)
        nonzero = triu[triu > 0]
        if len(nonzero) == 0:
            return 1.0

        eps_grid = np.linspace(
            float(np.percentile(nonzero, 5)),
            float(np.percentile(nonzero, 95)),
            grid_size,
        )

        Ks = []
        for eps in eps_grid:
            centroids, _, _ = CHARSExtractor._dp_means(X, eps=float(eps), max_iter=max_iter)
            Ks.append(len(centroids))

        Ks = np.array(Ks, dtype=float)

        x1, y1 = float(eps_grid[0]), float(Ks[0])
        x2, y2 = float(eps_grid[-1]), float(Ks[-1])
        chord_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if chord_len < 1e-12:
            return float(eps_grid[grid_size // 2])

        distances = np.abs(
            (x2 - x1) * (y1 - Ks) - (x1 - eps_grid) * (y2 - y1)
        ) / chord_len
        elbow_idx = int(np.argmax(distances))
        return float(eps_grid[elbow_idx])

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(x.abs())

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("CHARS requires contrast_data (source distribution)")

        target_acts_dict = self._get_activations(target_data)
        contrast_acts_dict = self._get_activations(contrast_data)

        # Apply tail transform to all activations before any computation
        if self.chars_tail_transform == "symlog":
            target_acts_dict = {
                k: self._symlog(v.to(torch.float32)).to(v.dtype)
                for k, v in target_acts_dict.items()
            }
            contrast_acts_dict = {
                k: self._symlog(v.to(torch.float32)).to(v.dtype)
                for k, v in contrast_acts_dict.items()
            }

        self.vector = {}
        chars_centroids_A = {}
        chars_centroids_B = {}
        chars_coupling = {}
        chars_k_dict = {}
        chars_components = {}
        chars_std_A = {}
        chars_std_B = {}
        chars_P_concept_dict = {}
        chars_X_mean_dict = {}

        for layer in self.layer:
            X_A = contrast_acts_dict[layer].to(torch.float32)
            X_B = target_acts_dict[layer].to(torch.float32)

            self.vector[layer] = X_B.mean(dim=0) - X_A.mean(dim=0)

            # -----------------------------------------------------------------
            # DP-Means non-parametric mode (chars_k = "auto")
            # -----------------------------------------------------------------
            if self.chars_k == "auto":
                if self.chars_eps is None:
                    eps_A = self._elbow_search(X_A)
                    eps_B = self._elbow_search(X_B)
                    logger.info(f"Elbow eps: eps_A={eps_A:.2f}, eps_B={eps_B:.2f} (layer={layer})")
                else:
                    eps_A = self.chars_eps
                    eps_B = self.chars_eps
                centroids_A_t, labels_A, counts_A = self._dp_means(X_A, eps=eps_A)
                centroids_B_t, labels_B, counts_B = self._dp_means(X_B, eps=eps_B)
                K_A, K_B = len(centroids_A_t), len(centroids_B_t)
                K_actual = max(K_A, K_B)
                logger.info(f"DP-Means: K_A={K_A}, K_B={K_B} (layer={layer})")

                if K_A == 0 or K_B == 0:
                    centroids_A = X_A.mean(dim=0, keepdim=True)
                    centroids_B = X_B.mean(dim=0, keepdim=True)
                    w_A = torch.ones(1, device=self.device)
                    w_B = torch.ones(1, device=self.device)
                    P_star = torch.ones((1, 1), device=self.device)
                    std_A = X_A.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8)
                    std_B = X_B.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8)
                else:
                    centroids_A = centroids_A_t.to(device=self.device)
                    centroids_B = centroids_B_t.to(device=self.device)
                    w_A = torch.tensor(counts_A / counts_A.sum(), dtype=torch.float32, device=self.device)
                    w_B = torch.tensor(counts_B / counts_B.sum(), dtype=torch.float32, device=self.device)

                    std_A = torch.zeros_like(centroids_A)
                    for i in range(K_A):
                        mask_i = (labels_A == i)
                        std_A[i] = X_A[mask_i].std(dim=0, unbiased=False).clamp(min=1e-8) if mask_i.sum() > 1 else torch.ones_like(centroids_A[0])
                    std_B = torch.zeros_like(centroids_B)
                    for j in range(K_B):
                        mask_j = (labels_B == j)
                        std_B[j] = X_B[mask_j].std(dim=0, unbiased=False).clamp(min=1e-8) if mask_j.sum() > 1 else torch.ones_like(centroids_B[0])

                    # Pad to same K so coupling stays K×K
                    if K_A < K_actual:
                        pad = K_actual - K_A
                        mean_A = X_A.mean(dim=0, keepdim=True)
                        centroids_A = torch.cat([centroids_A, mean_A.expand(pad, -1)])
                        w_A = torch.cat([w_A, torch.zeros(pad, device=self.device)])
                        std_A = torch.cat([std_A, X_A.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8).expand(pad, -1)])
                    if K_B < K_actual:
                        pad = K_actual - K_B
                        mean_B = X_B.mean(dim=0, keepdim=True)
                        centroids_B = torch.cat([centroids_B, mean_B.expand(pad, -1)])
                        w_B = torch.cat([w_B, torch.zeros(pad, device=self.device)])
                        std_B = torch.cat([std_B, X_B.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8).expand(pad, -1)])

                    C = torch.sum((centroids_A.unsqueeze(1) - centroids_B.unsqueeze(0)) ** 2, dim=-1)
                    if self.chars_diag:
                        bures = torch.sum((std_B.unsqueeze(0) - std_A.unsqueeze(1)) ** 2, dim=-1)
                        C = C + bures
                    C = C / (C.max() + 1e-12)
                    K_matrix = torch.exp(-C / self.chars_lambda)

                    u = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                    v = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                    for t in range(self.chars_max_iter):
                        v_prev = v.clone()
                        u = w_A / (torch.matmul(K_matrix, v) + 1e-12)
                        v = w_B / (torch.matmul(K_matrix.T, u) + 1e-12)
                        if torch.norm(v - v_prev, p=1) < self.chars_tau:
                            break
                    P_star = torch.diag(u) @ K_matrix @ torch.diag(v)

                # Store results and skip standard K-means path
                chars_centroids_A[layer] = centroids_A.cpu()
                chars_centroids_B[layer] = centroids_B.cpu()
                chars_coupling[layer] = P_star.cpu()
                chars_k_dict[layer] = K_actual
                chars_std_A[layer] = std_A.cpu()
                chars_std_B[layer] = std_B.cpu()

                diffs = centroids_B.unsqueeze(0) - centroids_A.unsqueeze(1)
                v_mean = self.vector[layer]
                v_centered = diffs - v_mean.to(device=diffs.device, dtype=diffs.dtype)
                weighted_diffs = torch.sqrt(P_star).unsqueeze(-1).to(device=diffs.device, dtype=diffs.dtype) * v_centered
                Y = weighted_diffs.reshape(-1, X_A.shape[-1])
                try:
                    _, _, Vh = torch.linalg.svd(Y.to(torch.float32), full_matrices=False)
                    chars_components[layer] = Vh.cpu()
                except Exception:
                    chars_components[layer] = torch.zeros((min(K_actual * K_actual, X_A.shape[-1]), X_A.shape[-1]))
                continue

            # -----------------------------------------------------------------
            # Standard fixed-K K-means path
            # -----------------------------------------------------------------
            N_A = X_A.shape[0]
            N_B = X_B.shape[0]
            K_actual = min(self.chars_k, N_A, N_B)

            if K_actual < 1:
                K_actual = 1
                centroids_A = torch.zeros((1, X_A.shape[-1]), device=self.device)
                centroids_B = torch.zeros((1, X_B.shape[-1]), device=self.device)
                w_A = torch.ones(1, device=self.device)
                w_B = torch.ones(1, device=self.device)
                P_star = torch.ones((1, 1), device=self.device)
                std_A = torch.ones((1, X_A.shape[-1]), device=self.device)
                std_B = torch.ones((1, X_B.shape[-1]), device=self.device)
            elif K_actual == 1:
                centroids_A = X_A.mean(dim=0, keepdim=True)
                centroids_B = X_B.mean(dim=0, keepdim=True)
                w_A = torch.ones(1, device=self.device)
                w_B = torch.ones(1, device=self.device)
                P_star = torch.ones((1, 1), device=self.device)
                std_A = X_A.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8)
                std_B = X_B.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8)
            else:
                # Optional subspace projection before K-means
                if self.chars_pca_k > 0:
                    X_combined = torch.cat([X_A, X_B], dim=0)
                    X_combined_mean = X_combined.mean(dim=0)
                    X_centered = X_combined - X_combined_mean
                    U, S, Vh = torch.linalg.svd(X_centered.to(torch.float32), full_matrices=False)
                    k_sub = min(self.chars_pca_k, Vh.shape[0], Vh.shape[1])
                    P_concept = Vh[:k_sub].T.contiguous().to(torch.float32)

                    Z_A = (X_A - X_combined_mean) @ P_concept
                    Z_B = (X_B - X_combined_mean) @ P_concept

                    kmeans_A = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                    labels_A = kmeans_A.fit_predict(Z_A.cpu().numpy())
                    centroids_A = torch.tensor(kmeans_A.cluster_centers_, dtype=torch.float32, device=self.device)
                    counts_A = np.bincount(labels_A, minlength=K_actual)
                    w_A = torch.tensor(counts_A / len(labels_A), dtype=torch.float32, device=self.device)

                    kmeans_B = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                    labels_B = kmeans_B.fit_predict(Z_B.cpu().numpy())
                    centroids_B = torch.tensor(kmeans_B.cluster_centers_, dtype=torch.float32, device=self.device)
                    counts_B = np.bincount(labels_B, minlength=K_actual)
                    w_B = torch.tensor(counts_B / len(labels_B), dtype=torch.float32, device=self.device)

                    std_A = torch.zeros_like(centroids_A)
                    std_B = torch.zeros_like(centroids_B)
                    for i in range(K_actual):
                        mask_i = (labels_A == i)
                        if mask_i.sum() > 1:
                            std_A[i] = Z_A[mask_i].std(dim=0, unbiased=False).clamp(min=1e-8)
                        else:
                            std_A[i] = 1.0
                    for j in range(K_actual):
                        mask_j = (labels_B == j)
                        if mask_j.sum() > 1:
                            std_B[j] = Z_B[mask_j].std(dim=0, unbiased=False).clamp(min=1e-8)
                        else:
                            std_B[j] = 1.0

                    C = torch.sum((centroids_A.unsqueeze(1) - centroids_B.unsqueeze(0)) ** 2, dim=-1)
                    C = C / (C.max() + 1e-12)
                    K_matrix = torch.exp(-C / self.chars_lambda)

                    u = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                    v = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                    for t in range(self.chars_max_iter):
                        v_prev = v.clone()
                        u = w_A / (torch.matmul(K_matrix, v) + 1e-12)
                        v = w_B / (torch.matmul(K_matrix.T, u) + 1e-12)
                        if torch.norm(v - v_prev, p=1) < self.chars_tau:
                            break
                    P_star = torch.diag(u) @ K_matrix @ torch.diag(v)

                    chars_P_concept_dict[layer] = P_concept.cpu()
                    chars_X_mean_dict[layer] = X_combined_mean.cpu()
                elif self.chars_whiten:
                    from sklearn.decomposition import PCA
                    X_combined = torch.cat([X_A, X_B], dim=0).cpu().numpy()
                    pca = PCA(n_components=min(X_combined.shape[0], X_combined.shape[1]), whiten=True)
                    Z_combined = pca.fit_transform(X_combined)
                    Z_A = torch.from_numpy(Z_combined[:len(X_A)]).to(device=self.device, dtype=torch.float32)
                    Z_B = torch.from_numpy(Z_combined[len(X_A):]).to(device=self.device, dtype=torch.float32)
                    kmeans_A = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                    labels_A = kmeans_A.fit_predict(Z_A.cpu().numpy())
                    centroids_Z_A = torch.tensor(kmeans_A.cluster_centers_, dtype=torch.float32, device=self.device)
                    centroids_A = torch.tensor(pca.inverse_transform(centroids_Z_A.cpu().numpy()), dtype=torch.float32, device=self.device)
                    counts_A = np.bincount(labels_A, minlength=K_actual)
                    w_A = torch.tensor(counts_A / len(labels_A), dtype=torch.float32, device=self.device)
                    kmeans_B = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                    labels_B = kmeans_B.fit_predict(Z_B.cpu().numpy())
                    centroids_Z_B = torch.tensor(kmeans_B.cluster_centers_, dtype=torch.float32, device=self.device)
                    centroids_B = torch.tensor(pca.inverse_transform(centroids_Z_B.cpu().numpy()), dtype=torch.float32, device=self.device)
                    counts_B = np.bincount(labels_B, minlength=K_actual)
                    w_B = torch.tensor(counts_B / len(labels_B), dtype=torch.float32, device=self.device)
                else:
                    kmeans_A = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                    labels_A = kmeans_A.fit_predict(X_A.cpu().numpy())
                    centroids_A = torch.tensor(kmeans_A.cluster_centers_, dtype=torch.float32, device=self.device)
                    counts_A = np.bincount(labels_A, minlength=K_actual)
                    w_A = torch.tensor(counts_A / len(labels_A), dtype=torch.float32, device=self.device)

                    kmeans_B = KMeans(n_clusters=K_actual, random_state=42, n_init="auto")
                    labels_B = kmeans_B.fit_predict(X_B.cpu().numpy())
                    centroids_B = torch.tensor(kmeans_B.cluster_centers_, dtype=torch.float32, device=self.device)
                    counts_B = np.bincount(labels_B, minlength=K_actual)
                    w_B = torch.tensor(counts_B / len(labels_B), dtype=torch.float32, device=self.device)

                if not self.chars_pca_k > 0:
                    std_A = torch.zeros_like(centroids_A)
                    std_B = torch.zeros_like(centroids_B)
                    for i in range(K_actual):
                        mask_i = (labels_A == i)
                        if mask_i.sum() > 1:
                            std_A[i] = X_A[mask_i].std(dim=0, unbiased=False).clamp(min=1e-8)
                        else:
                            std_A[i] = 1.0
                    for j in range(K_actual):
                        mask_j = (labels_B == j)
                        if mask_j.sum() > 1:
                            std_B[j] = X_B[mask_j].std(dim=0, unbiased=False).clamp(min=1e-8)
                        else:
                            std_B[j] = 1.0

                    C = torch.sum((centroids_A.unsqueeze(1) - centroids_B.unsqueeze(0)) ** 2, dim=-1)
                    if self.chars_diag:
                        bures = torch.sum((std_B.unsqueeze(0) - std_A.unsqueeze(1)) ** 2, dim=-1)
                        C = C + bures
                    C = C / (C.max() + 1e-12)
                    K_matrix = torch.exp(-C / self.chars_lambda)

                    u = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                    v = torch.ones(K_actual, dtype=torch.float32, device=self.device)
                    for t in range(self.chars_max_iter):
                        v_prev = v.clone()
                        u = w_A / (torch.matmul(K_matrix, v) + 1e-12)
                        v = w_B / (torch.matmul(K_matrix.T, u) + 1e-12)
                        if torch.norm(v - v_prev, p=1) < self.chars_tau:
                            break
                    P_star = torch.diag(u) @ K_matrix @ torch.diag(v)

            chars_centroids_A[layer] = centroids_A.cpu()
            chars_centroids_B[layer] = centroids_B.cpu()
            chars_coupling[layer] = P_star.cpu()
            chars_k_dict[layer] = K_actual
            chars_std_A[layer] = std_A.cpu()
            chars_std_B[layer] = std_B.cpu()

            if self.chars_pca_k > 0:
                chars_components[layer] = torch.zeros((1, X_A.shape[-1]))
            else:
                diffs = centroids_B.unsqueeze(0) - centroids_A.unsqueeze(1)
                v_mean = self.vector[layer]
                v_centered = diffs - v_mean.to(device=diffs.device, dtype=diffs.dtype)
                weighted_diffs = torch.sqrt(P_star).unsqueeze(-1).to(device=diffs.device, dtype=diffs.dtype) * v_centered
                Y = weighted_diffs.reshape(-1, X_A.shape[-1])
                try:
                    _, _, Vh = torch.linalg.svd(Y.to(torch.float32), full_matrices=False)
                    chars_components[layer] = Vh.cpu()
                except Exception:
                    chars_components[layer] = torch.zeros((min(K_actual * K_actual, X_A.shape[-1]), X_A.shape[-1]))

        self.metadata = {
            "method": "CHARS",
            "chars_centroids_A": chars_centroids_A,
            "chars_centroids_B": chars_centroids_B,
            "chars_coupling": chars_coupling,
            "chars_k": chars_k_dict,
            "chars_components": chars_components,
            "chars_std_A": chars_std_A,
            "chars_std_B": chars_std_B,
            "chars_pca_k": self.chars_pca_k if self.chars_pca_k > 0 else 0,
            "chars_eps": self.chars_eps,
            "n_target": len(target_data),
            "n_contrast": len(contrast_data),
        }
        if self.chars_pca_k > 0:
            self.metadata["chars_P_concept"] = chars_P_concept_dict
            self.metadata["chars_X_mean"] = chars_X_mean_dict
        return self.vector


# =============================================================================
# LinNEAS - Linearized Non-linear End-to-end Activation Steering
# =============================================================================

class TargetModuleReached(Exception):
    pass


def _wasserstein_trim_sort(A, B, p=2):
    """Wasserstein distance between sorted marginal distributions."""
    A_sorted = torch.sort(A, dim=0).values
    B_sorted = torch.sort(B, dim=0).values
    return torch.pow(torch.abs(A_sorted - B_sorted), p).mean(dim=0)


class LinNEASExtractor(BaseExtractor):
    """Train or load a LinNEAS steering model payload."""

    METHOD_NAME = "LinNEAS"

    def __init__(
        self,
        model,
        layer: List[int],
        batch_size: int = 8,
        device: Optional[torch.device] = None,
        hook_point: str = "pre",
        position: str = "mean",
        linneas_lr: float = 0.1,
        linneas_steps: int = 1000,
        linneas_reg_l1: float = 0.0,
        linneas_reg_l2: float = 0.0,
        linneas_optimizer: str = "SGD",
        linneas_proximal: Optional[str] = None,
        linneas_init_identity: bool = True,
        **kwargs,
    ):
        super().__init__(model, layer, batch_size, device, hook_point=hook_point, position=position)
        self.linneas_lr = linneas_lr
        self.linneas_steps = linneas_steps
        self.linneas_reg_l1 = linneas_reg_l1
        self.linneas_reg_l2 = linneas_reg_l2
        self.linneas_optimizer = linneas_optimizer
        self.linneas_proximal = linneas_proximal
        self.linneas_init_identity = linneas_init_identity

    def _get_activations(self, inputs: List[str], **kwargs) -> Dict[int, torch.Tensor]:
        return collect_dense_activations(
            model=self.model,
            texts=inputs,
            layers=self.layer,
            hook_point=self.hook_point,
            batch_size=self.batch_size,
            pooling=self.position,
            device=self.device,
            tokenizer=self.model.tokenizer,
            reduce="none",
            return_key_format="layer",
        )

    def extract(
        self,
        target_data: List[str],
        contrast_data: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        if contrast_data is None:
            raise ValueError("LinNEAS requires contrast_data (source distribution)")

        d_model = self.model.cfg.d_model
        dtype = self.model.cfg.dtype

        self.w1 = torch.nn.ParameterDict({
            str(l): torch.nn.Parameter(
                torch.ones(d_model, device=self.device, dtype=dtype) if self.linneas_init_identity
                else torch.randn(d_model, device=self.device, dtype=dtype)
            ) for l in self.layer
        })
        self.b1 = torch.nn.ParameterDict({
            str(l): torch.nn.Parameter(torch.zeros(d_model, device=self.device, dtype=dtype))
            for l in self.layer
        })

        logger.info("Pre-computing target activations...")
        target_acts_dict = self._get_activations(target_data)
        contrast_tokens = self.model.to_tokens(contrast_data, prepend_bos=True)
        N_contrast = len(contrast_data)
        N_target = len(target_data)

        parameters_to_optimize = list(self.w1.parameters()) + list(self.b1.parameters())
        if self.linneas_optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=self.linneas_lr)
        else:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.linneas_lr)

        original_requires_grad = {}
        for name, p in self.model.named_parameters():
            original_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        hp = self.hook_point[0] if isinstance(self.hook_point, list) else self.hook_point
        max_layer = max(self.layer)
        max_layer_name = get_hook_name(max_layer, hp)

        try:
            logger.info("Optimizing LinNEAS transport maps...")
            pbar = tqdm(range(self.linneas_steps), desc="LinNEAS training")
            for step in pbar:
                contrast_idx = torch.randint(0, N_contrast, (self.batch_size,))
                batch_tokens = contrast_tokens[contrast_idx].to(self.device)
                target_idx = torch.randint(0, N_target, (self.batch_size,))

                cache = {}
                def make_hook(layer_str, h_name):
                    def hook_fn(resid, hook):
                        w = self.w1[layer_str]
                        b = self.b1[layer_str]
                        steered = resid * w + b
                        cache[h_name] = steered
                        if h_name == max_layer_name:
                            raise TargetModuleReached()
                        return steered
                    return hook_fn

                hooks = [
                    (get_hook_name(l, hp), make_hook(str(l), get_hook_name(l, hp)))
                    for l in self.layer
                ]

                try:
                    with self.model.hooks(fwd_hooks=hooks):
                        self.model(batch_tokens)
                except TargetModuleReached:
                    pass

                total_loss = torch.tensor(0.0, device=self.device)
                for l in self.layer:
                    h_name = get_hook_name(l, hp)
                    c_acts_pooled = get_resid_acts(cache[h_name], self.position)
                    t_acts_pooled = target_acts_dict[l][target_idx].to(self.device)
                    loss_layer = _wasserstein_trim_sort(c_acts_pooled.T, t_acts_pooled.T, p=2).mean()
                    total_loss = total_loss + loss_layer

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_loss_val = total_loss.item()
                del total_loss, loss_layer, c_acts_pooled, t_acts_pooled
                cache.clear()
                del cache, hooks, batch_tokens, contrast_idx, target_idx

                lr_min = self.linneas_lr * 0.1
                lr = lr_min + 0.5 * (self.linneas_lr - lr_min) * (1 + math.cos(math.pi * step / self.linneas_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                if self.linneas_proximal == "l1":
                    with torch.no_grad():
                        tau = lr * self.linneas_reg_l1
                        for l in self.layer:
                            l_str = str(l)
                            w = self.w1[l_str]
                            b = self.b1[l_str]
                            w.data = torch.sign(w.data - 1.0) * torch.relu(torch.abs(w.data - 1.0) - tau) + 1.0
                            b.data = torch.sign(b.data) * torch.relu(torch.abs(b.data) - tau)
                elif self.linneas_proximal == "sparse":
                    with torch.no_grad():
                        tau = lr
                        lam_l1_w = self.linneas_reg_l1
                        lam_l2_w = self.linneas_reg_l2
                        for l in self.layer:
                            l_str = str(l)
                            w = self.w1[l_str]
                            b = self.b1[l_str]
                            dim_w = w.shape[-1]
                            sqrt_dim = math.sqrt(dim_w)
                            w_centered = w.data - 1.0
                            w_l1 = torch.sign(w_centered) * torch.relu(torch.abs(w_centered) - tau * lam_l1_w)
                            norm_w = torch.linalg.norm(w_l1, ord=2, dim=-1, keepdim=True)
                            w.data = torch.relu(1.0 - (tau * lam_l2_w * sqrt_dim / norm_w)) * w_l1 + 1.0
                            w.data = torch.where(torch.isnan(w.data), 1.0, w.data)
                            b_l1 = torch.sign(b.data) * torch.relu(torch.abs(b.data) - tau * lam_l1_w)
                            norm_b = torch.linalg.norm(b_l1, ord=2, dim=-1, keepdim=True)
                            b.data = torch.relu(1.0 - (tau * lam_l2_w * sqrt_dim / norm_b)) * b_l1
                            b.data = torch.where(torch.isnan(b.data), 0.0, b.data)

                if step % 10 == 0:
                    pbar.set_postfix({"loss": f"{total_loss_val:.4f}"})
                    torch.cuda.empty_cache()
        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad = original_requires_grad.get(name, True)
            self.model.reset_hooks()

        self.vector = {l: self.b1[str(l)].detach().cpu() for l in self.layer}
        self.metadata = {
            "linneas_w1": {l: self.w1[str(l)].detach().cpu() for l in self.layer},
            "linneas_b1": {l: self.b1[str(l)].detach().cpu() for l in self.layer},
        }
        return self.vector
