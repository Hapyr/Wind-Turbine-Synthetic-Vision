# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def align_tensors(A, B):
    """
    Aligns tensor B to tensor A by permuting the points in B to minimize MSE.
    
    Args:
        A (torch.Tensor): Tensor of shape (batch_size, num_points, coord_dim)
        B (torch.Tensor): Tensor of same shape as A
    
    Returns:
        torch.Tensor: Aligned version of B with minimal MSE to A
    """
    device = A.device
    batch_size, num_points, coord_dim = A.shape
    import itertools
    # Generate all permutations of the points (only for 3 points)
    assert num_points == 3, "This implementation only supports 3 points"
    permutations = torch.tensor(list(itertools.permutations(range(num_points))), device=device)
    
    # Expand B to all possible permutations (batch_size, num_perms, num_points, coord_dim)
    B_expanded = B[:, permutations, :]  # Shape: (batch_size, 6, 3, 2)
    
    # Compute MSE between A and all permuted versions of B
    A_expanded = A[:,:,:2].unsqueeze(1)  # Shape: (batch_size, 1, 3, 2)

    mse = ((A_expanded - B_expanded) ** 2).mean(dim=(2, 3)).squeeze()  # Shape: (batch_size, 6)

    # Find the best permutation for each batch
    best_perm_indices = torch.argmin(mse, dim=-1)  # Shape: (batch_size,)
    best_perms = permutations[best_perm_indices]  # Shape: (batch_size, 3)
    
    # Apply the best permutation to B
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1)
    B_aligned = B[batch_indices, best_perms, :]
    return B_aligned

def get_aligned_d(kpt1, kpt2):

    # Reshape and broadcast
    a_expanded = kpt1.unsqueeze(1)
    b_expanded = kpt2.unsqueeze(0)

    # Expand to final dimensions [M, N, 1] for both
    a_final = a_expanded.expand(kpt1.size(0), kpt2.size(0), 7, 2)
    b_final = b_expanded.expand(kpt1.size(0), kpt2.size(0), 7, 2)

    # Concatenate along the last dimension to create pairs
    result = torch.stack((a_final, b_final), dim=3)

    result = result.view(kpt1.size(0)*kpt2.size(0),7,2,2)

    A = result[:,4:,0,:]
    B = result[:,4:,1,:]

    B_al = align_tensors(A,B)

    A = result[:,:,0,:]
    B = torch.cat([result[:,:4,1,:],B_al],dim=1)

    bb = torch.cat([A,B],dim=2).view(kpt1.size(0),kpt2.size(0),7,2,2)

    return (bb[:,:,:,0,0] - bb[:,:,:,1,0]).pow(2) + (bb[:,:,:,0,1] - bb[:,:,:,1,1]).pow(2)

    
def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """

    d = get_aligned_d(kpt1[:,:,:2],kpt2)

    # d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)

