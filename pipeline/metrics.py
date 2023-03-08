"""Metric utilities

All metrics take a reference scan (ref) and an output/reconstructed scan (x).
"""

import torch


def l2(ref: torch.Tensor, pred: torch.Tensor):
    """L2 loss.
    """
    return torch.sqrt(torch.mean(torch.abs(ref - pred)**2))


def l1(ref: torch.Tensor, pred: torch.Tensor):
    """L1 loss.
    """
    return torch.mean(torch.abs(ref - pred))


def psnr(ref: torch.Tensor, pred: torch.Tensor):
    """Peak signal-to-noise ratio (PSNR)
    """
    scale = torch.norm(ref, p=float('inf'))
    return 20 * torch.log10(scale / l2(ref, pred))
