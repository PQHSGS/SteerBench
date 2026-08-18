# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import torch
import numpy as np


# Adapted from scipy.stats
def wasserstein_sort_trim(u_values, v_values, p=1):
    """
    Computes a naive, non-deterministic approximation of the p-Wasserstein
    distance. This version first sorts both distributions, then randomly
    subsamples an equal number of points from each before computing the distance.

    *** WARNING: This is not the correct Wasserstein distance. ***
    The result is non-deterministic (changes on every run) and biased.
    It should not be used for reliable or reproducible measurements.

    The subsampling process is as follows:
    1. Both input distributions are sorted independently.
    2. If their sizes differ, the smaller size (`min_size`) is determined.
    3. `min_size` random indices are generated for each distribution.
    4. These random indices are sorted.
    5. The sorted distributions are subsampled using these sorted indices.

    Args:
        u_values (Tensor): Samples from the first distribution.
        v_values (Tensor): Samples from the second distribution.
        p (int): The order of the distance.

    Returns:
        Tensor: A random, biased approximation of the p-Wasserstein distance.
    """
    # u_values = torch.as_tensor(u_values, dtype=torch.float32)
    # v_values = torch.as_tensor(v_values, dtype=torch.float32)

    # 1. Sort the full input distributions first.
    u_sorted = torch.sort(u_values, dim=-1).values
    v_sorted = torch.sort(v_values, dim=-1).values

    src_size = u_sorted.shape[-1]
    dst_size = v_sorted.shape[-1]

    # 2. If sizes are different, subsample BOTH distributions.
    if src_size != dst_size:
        # Determine the target size for subsampling (the smaller of the two)
        min_size = min(src_size, dst_size)

        # Generate sorted random indices for the source distribution
        # This is equivalent to your snippet:
        # torch.sort(torch.randperm(src_size)[:min_size]).values
        rnd_idx_src = torch.sort(torch.randperm(src_size)[:min_size]).values
        u_sub = u_sorted[..., rnd_idx_src]

        # Generate sorted random indices for the destination distribution
        rnd_idx_dst = torch.sort(torch.randperm(dst_size)[:min_size]).values
        v_sub = v_sorted[..., rnd_idx_dst]
    else:
        # Sizes are the same, no subsampling needed.
        # The distributions are already sorted from step 1.
        u_sub = u_sorted
        v_sub = v_sorted

    # 3. Compute the distance on the subsampled and sorted distributions.
    # Note: u_sub and v_sub are already sorted.
    diff = u_sub - v_sub

    if p == 1:
        return torch.sum(torch.abs(diff), dim=-1)

    return torch.sum(torch.pow(diff, p), dim=-1)


def wasserstein_trim_sort(u_values, v_values, p=1):
    """
    Computes a naive, non-deterministic approximation of the p-Wasserstein
    distance by randomly subsampling the larger distribution.

    *** WARNING: This is not the correct Wasserstein distance. ***
    The result is non-deterministic (changes on every run) and biased.
    It should not be used for reliable or reproducible measurements.

    Args:
        u_values (Tensor): Samples from the first distribution.
        v_values (Tensor): Samples from the second distribution.
        p (int): The order of the distance.

    Returns:
        Tensor: A random, biased approximation of the p-Wasserstein distance.
    """
    u_values = torch.as_tensor(u_values, dtype=torch.float32)
    v_values = torch.as_tensor(v_values, dtype=torch.float32)

    src_size = u_values.shape[-1]
    dst_size = v_values.shape[-1]

    # If sizes are different, subsample the larger distribution
    if src_size != dst_size:
        min_size = min(src_size, dst_size)
        if src_size > dst_size:
            # u_values is larger, sample it down
            # NOTE: This uses the same random indices for all items in a batch
            rand_indices = torch.randperm(src_size)[:min_size]
            u_sub = u_values[..., rand_indices]
            v_sub = v_values
        else:
            # v_values is larger, sample it down
            rand_indices = torch.randperm(dst_size)[:min_size]
            u_sub = u_values
            v_sub = v_values[..., rand_indices]
    else:
        # Sizes are the same, no subsampling needed
        u_sub = u_values
        v_sub = v_values

    # Sort the samples
    u_sorted = torch.sort(u_sub, dim=-1).values
    v_sorted = torch.sort(v_sub, dim=-1).values

    # Compute the distance on the (potentially subsampled) distributions
    diff = u_sorted - v_sorted

    if p == 1:
        return torch.sum(torch.abs(diff), dim=-1)

    return torch.sum(torch.pow(diff, p), dim=-1)


def wasserstein_quantile_transform(
    u_values, v_values, u_weights=None, v_weights=None, p=1
):
    """
    Computes 1D Wasserstein-p distance for batches, optimized for a very small `v`.

    This 'hybrid' method avoids `searchsorted` by sorting all values together
    with corresponding IDs and weights, then calculating CDFs with `cumsum`.
    """
    # --- 1. Initial Setup and Type Conversion ---
    u_values = torch.as_tensor(u_values, dtype=torch.float64)
    v_values = torch.as_tensor(v_values, dtype=torch.float64)
    if u_values.dim() == 1:
        u_values = u_values.unsqueeze(0)
    if v_values.dim() == 1:
        v_values = v_values.unsqueeze(0)

    batch_size, num_u_points = u_values.shape
    _, num_v_points = v_values.shape
    device, dtype = u_values.device, u_values.dtype

    # Handle weights
    if u_weights is None:
        u_weights = torch.full(
            (batch_size, num_u_points), 1.0 / num_u_points, dtype=dtype, device=device
        )
    else:
        u_weights = torch.as_tensor(u_weights, dtype=dtype, device=device)
        if u_weights.dim() == 1:
            u_weights = u_weights.unsqueeze(0)

    if v_weights is None:
        v_weights = torch.full(
            (batch_size, num_v_points), 1.0 / num_v_points, dtype=dtype, device=device
        )
    else:
        v_weights = torch.as_tensor(v_weights, dtype=dtype, device=device)
        if v_weights.dim() == 1:
            v_weights = v_weights.unsqueeze(0)

    # --- 2. Create combined tensors with IDs ---
    # Tag `u` values with 0 and `v` values with 1
    u_ids = torch.zeros_like(u_values)
    v_ids = torch.ones_like(v_values)

    # Concatenate all data
    all_values = torch.cat([u_values, v_values], dim=-1)
    all_weights = torch.cat([u_weights, v_weights], dim=-1)
    all_ids = torch.cat([u_ids, v_ids], dim=-1)

    # --- 3. Single Sort Operation ---
    sorter = torch.argsort(all_values, dim=-1, stable=True)

    # Apply the same sorting to all three tensors
    all_values_sorted = torch.gather(all_values, -1, sorter)
    all_weights_sorted = torch.gather(all_weights, -1, sorter)
    all_ids_sorted = torch.gather(all_ids, -1, sorter)

    # --- 4. Direct CDF Calculation via Cumsum ---
    # Separate weights by their origin (u or v)
    u_weights_mask = 1 - all_ids_sorted
    v_weights_mask = all_ids_sorted

    u_weights_sorted = all_weights_sorted * u_weights_mask
    v_weights_sorted = all_weights_sorted * v_weights_mask

    # Get total weights for normalization (handle unweighted case correctly)
    total_u_weight = u_weights.sum(dim=-1, keepdim=True)
    total_v_weight = v_weights.sum(dim=-1, keepdim=True)

    # Compute cumulative sums to get the CDF values
    u_cdf = torch.cumsum(u_weights_sorted, dim=-1) / total_u_weight
    v_cdf = torch.cumsum(v_weights_sorted, dim=-1) / total_v_weight

    # --- 5. Final Distance Calculation ---
    # Compute differences between adjacent sorted values
    deltas = torch.diff(all_values_sorted, dim=-1)

    # The CDF difference is evaluated at the left point of each interval
    diff_cdf = u_cdf[:, :-1] - v_cdf[:, :-1]

    if p == 1:
        print((torch.abs(diff_cdf) * deltas).sum(dim=-1))
        return (torch.abs(diff_cdf) * deltas).sum(dim=-1)
    if p == 2:
        return (torch.square(diff_cdf * deltas)).sum(dim=-1)

    return (torch.pow(diff_cdf, p) * deltas).sum(dim=-1)


def wasserstein_distance_pytorch(
    u_values, v_values, u_weights=None, v_weights=None, p=1
):
    """
    Computes the 1D Wasserstein-p distance for batches of discrete distributions.

    Parameters
    ----------
    u_values, v_values : torch.Tensor
        2-D tensors of shape (batch_size, num_u_values) and (batch_size, num_v_values).
    u_weights, v_weights : torch.Tensor, optional
        2-D tensors of weights. If None, weights are assumed to be uniform.
        Shapes must match u_values and v_values respectively.
    p : int, optional
        The order of the Wasserstein distance. Default is 1.

    Returns
    -------
    distance : torch.Tensor
        A 1-D tensor of shape (batch_size,) containing the Wasserstein distance
        for each pair of distributions in the batch.
    """
    # Ensure inputs are 2D tensors.
    # We'll keep float64 for precision.
    u_values = torch.as_tensor(u_values, dtype=torch.float64)
    v_values = torch.as_tensor(v_values, dtype=torch.float64)
    if u_values.dim() == 1:
        u_values = u_values.unsqueeze(0)
    if v_values.dim() == 1:
        v_values = v_values.unsqueeze(0)

    batch_size = u_values.shape[0]
    device, dtype = u_values.device, u_values.dtype

    if u_weights is not None:
        u_weights = torch.as_tensor(u_weights, dtype=dtype, device=device)
        if u_weights.dim() == 1:
            u_weights = u_weights.unsqueeze(0)
    if v_weights is not None:
        v_weights = torch.as_tensor(v_weights, dtype=dtype, device=device)
        if v_weights.dim() == 1:
            v_weights = v_weights.unsqueeze(0)

    # Sort the values along the last dimension
    u_sorter = torch.argsort(u_values, dim=-1)
    v_sorter = torch.argsort(v_values, dim=-1)

    # Gather the sorted values
    u_values_sorted = torch.gather(u_values, -1, u_sorter)
    v_values_sorted = torch.gather(v_values, -1, v_sorter)

    # Concatenate and sort all values
    all_values = torch.cat([u_values_sorted, v_values_sorted], dim=-1)
    all_values_sorted, _ = torch.sort(all_values, dim=-1, stable=True)

    # Compute the differences between adjacent sorted values
    deltas = torch.diff(all_values_sorted, dim=-1)

    # Get the respective positions of the values of u and v among the values of
    # both distributions. `torch.searchsorted` is batched-aware.
    u_cdf_indices = torch.searchsorted(
        u_values_sorted, all_values_sorted[:, :-1].contiguous(), right=True
    )
    v_cdf_indices = torch.searchsorted(
        v_values_sorted, all_values_sorted[:, :-1].contiguous(), right=True
    )

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        # Uniform weights
        u_cdf = u_cdf_indices.to(dtype) / u_values.shape[-1]
    else:
        # Use provided weights
        u_sorted_weights = torch.gather(u_weights, -1, u_sorter)
        u_cumweights = torch.cumsum(u_sorted_weights, dim=-1)
        # Pad with a zero at the beginning for each batch item
        zeros_pad = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        u_sorted_cumweights = torch.cat([zeros_pad, u_cumweights], dim=-1)

        # Gather the cumulative weights and normalize
        u_cdf = torch.gather(u_sorted_cumweights, -1, u_cdf_indices)
        # Denominator needs to be broadcastable: (batch_size, 1)
        total_weight = u_sorted_cumweights[:, -1].unsqueeze(-1)
        u_cdf = u_cdf / total_weight

    if v_weights is None:
        # Uniform weights
        v_cdf = v_cdf_indices.to(dtype) / v_values.shape[-1]
    else:
        # Use provided weights
        v_sorted_weights = torch.gather(v_weights, -1, v_sorter)
        v_cumweights = torch.cumsum(v_sorted_weights, dim=-1)
        zeros_pad = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        v_sorted_cumweights = torch.cat([zeros_pad, v_cumweights], dim=-1)

        v_cdf = torch.gather(v_sorted_cumweights, -1, v_cdf_indices)
        total_weight = v_sorted_cumweights[:, -1].unsqueeze(-1)
        v_cdf = v_cdf / total_weight

    # Compute the value of the integral based on the CDFs.
    # This is a batched dot product: (a * b).sum(dim=-1)
    diff_cdf = u_cdf - v_cdf

    if p == 1:
        # Batched dot product
        return (torch.abs(diff_cdf) * deltas).sum(dim=-1)
    if p == 2:
        return (torch.square(diff_cdf) * deltas).sum(dim=-1)

    return (torch.pow(diff_cdf, p) * deltas).sum(dim=-1)


if __name__ == "__main__":
    from scipy.stats._stats_py import _cdf_distance as wasserstein_distance

    # --- Step 2: Generate a batch of test data ---
    BATCH_SIZE = 100
    NUM_U_POINTS = 4
    NUM_V_POINTS = 4

    # Use a fixed seed for reproducible results
    np.random.seed(42)

    # Generate random values for the distributions
    u_values_batch = np.random.rand(BATCH_SIZE, NUM_U_POINTS) * 10
    v_values_batch = np.random.rand(BATCH_SIZE, NUM_V_POINTS) * 10

    # Generate random weights and normalize them to sum to 1 for each distribution
    u_weights_batch = np.random.rand(BATCH_SIZE, NUM_U_POINTS)
    u_weights_batch_T = u_weights_batch / u_weights_batch.sum(axis=1, keepdims=True)
    u_weights_batch /= u_weights_batch.sum(axis=1, keepdims=True)

    v_weights_batch = np.random.rand(BATCH_SIZE, NUM_V_POINTS)
    v_weights_batch_T = v_weights_batch / v_weights_batch.sum(axis=1, keepdims=True)
    v_weights_batch /= v_weights_batch.sum(axis=1, keepdims=True)

    import time

    # --- Test 1: Unweighted Distributions ---
    print("--- Testing Unweighted Distributions ---")

    # Method 1: SciPy with a for loop
    start_time = time.time()
    scipy_results_unweighted = []
    for i in range(BATCH_SIZE):
        dist = wasserstein_distance(1, u_values_batch[i], v_values_batch[i])
        scipy_results_unweighted.append(dist)
    scipy_results_unweighted = np.array(scipy_results_unweighted)
    scipy_loop_time = time.time() - start_time

    # Method 2: Batched PyTorch function
    start_time = time.time()
    pytorch_results_unweighted = wasserstein_distance_fast(
        u_values_batch, v_values_batch
    ).numpy()
    pytorch_batched_time = time.time() - start_time
    # Method 3: Hybrid PyTorch function
    start_time = time.time()
    hybrid_results_unweighted = wasserstein_distance_fast_v2(
        u_values_batch.T, v_values_batch.T
    ).numpy()
    hybrid_batched_time = time.time() - start_time
    # Method 4: Naive Wasserstein
    start_time = time.time()
    naive_results_unweighted = naive_wasserstein(u_values_batch, v_values_batch).numpy()
    naive_batched_time = time.time() - start_time

    print(f"SciPy Loop Time: {scipy_loop_time} seconds")
    print(f"PyTorch Batched Time: {pytorch_batched_time} seconds")
    print(f"PyTorch Hybrid Time: {hybrid_batched_time} seconds")
    print(
        f"Results are close: {np.allclose(scipy_results_unweighted, pytorch_results_unweighted)}\n"
    )
    print(
        f"Results are close (hybrid): {np.allclose(scipy_results_unweighted, hybrid_results_unweighted)}\n"
    )
    print(
        f"Results are close (naive): {np.allclose(scipy_results_unweighted, naive_results_unweighted)}\n"
    )

    # --- Test 2: Weighted Distributions ---
    print("--- Testing Weighted Distributions (p=2) ---")
    p_order = 2

    # Method 1: SciPy with a for loop
    start_time = time.time()
    scipy_results_weighted = []
    for i in range(BATCH_SIZE):
        dist = wasserstein_distance(
            p_order,
            u_values_batch[i],
            v_values_batch[i],
            u_weights=u_weights_batch[i],
            v_weights=v_weights_batch[i],
        )
        scipy_results_weighted.append(dist)
    scipy_results_weighted = np.array(scipy_results_weighted)
    scipy_loop_time = time.time() - start_time

    # Method 2: Batched PyTorch function
    start_time = time.time()
    pytorch_results_weighted = wasserstein_distance_pytorch(
        u_values_batch, v_values_batch, u_weights_batch, v_weights_batch, p=p_order
    ).numpy()
    pytorch_batched_time = time.time() - start_time

    # Method 3: Hybrid PyTorch function
    start_time = time.time()
    hybrid_results_weighted = wasserstein_distance_fast_v2(
        u_values_batch.T,
        v_values_batch.T,
        u_weights_batch.T,
        v_weights_batch.T,
        p=p_order,
    ).numpy()

    hybrid_batched_time = time.time() - start_time
    print(f"SciPy Loop Time: {scipy_loop_time} seconds")
    print(f"PyTorch Batched Time: {pytorch_batched_time} seconds")
    print(f"PyTorch Hybrid Time: {hybrid_batched_time} seconds")
    print(
        f"Results are close: {np.allclose(scipy_results_weighted, pytorch_results_weighted)}\n"
    )
    print(
        f"Results are close (hybrid): {np.allclose(scipy_results_weighted, hybrid_results_weighted)}\n"
    )
