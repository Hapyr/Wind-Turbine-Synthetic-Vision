# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def align_tensors(self, A, B):
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
        mse = ((A_expanded - B_expanded) ** 2).mean(dim=(2, 3))  # Shape: (batch_size, 6)
        
        # Find the best permutation for each batch
        best_perm_indices = torch.argmin(mse, dim=1)  # Shape: (batch_size,)
        best_perms = permutations[best_perm_indices]  # Shape: (batch_size, 3)
        
        # Apply the best permutation to B
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1)
        B_aligned = B[batch_indices, best_perms, :]
        return B_aligned

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""

        pred_kpts_1234 = pred_kpts[:, :4, :]
        gt_kpts_1234 = gt_kpts[:, :4, :]
        d_first_4 = (pred_kpts_1234[..., 0] - gt_kpts_1234[..., 0]).pow(2) + (pred_kpts_1234[..., 1] - gt_kpts_1234[..., 1]).pow(2)
        
        pred_kpts_567 = pred_kpts[:, [4,5,6], :]
        gt_kpts_567 = gt_kpts[:, [4,5,6], :]

        pred_kpts_aligned = self.align_tensors(gt_kpts_567, pred_kpts_567)

        d_567 = (pred_kpts_aligned[..., 0] - gt_kpts_567[..., 0]).pow(2) + (pred_kpts_aligned[..., 1] - gt_kpts_567[..., 1]).pow(2)

        d = torch.cat((d_first_4, d_567), dim=1)

        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval

        r = (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

        return r