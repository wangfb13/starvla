"""Spatial, two-finger depth/deformation conditioning for recovery policies.

Inputs are decoded SDK fields, NOT RGB: [batch, finger, (D, Dx, Dy), H, W].
Finger order must remain right-left, right-right at training and inference.
"""

from collections.abc import Sequence

import torch
from torch import nn


class DDConditioner(nn.Module):
    """Shared finger CNN, spatial flattening, and zero-initialized residual.

    Keeping a 6 x 8 spatial grid before the learned projection preserves
    contact location. Fixed channel scaling preserves physical amplitude and
    signed deformation; there is no per-frame recentering or normalization.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 128,
        channel_scale: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        scale = torch.as_tensor(list(channel_scale), dtype=torch.float32)
        if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("DD channel_scale must contain three finite, positive values.")
        if hidden_dim < 1 or latent_dim < 1:
            raise ValueError("DD hidden_dim and latent_dim must be positive.")
        self.register_buffer("channel_scale", scale.reshape(1, 1, 3, 1, 1))
        self.finger_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((6, 8)),
            nn.Flatten(),
            nn.Linear(48 * 6 * 8, latent_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.GELU())
        self.projector = nn.Linear(latent_dim, hidden_dim)
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

    def encode(self, tactile_dd: torch.Tensor) -> torch.Tensor:
        if tactile_dd.ndim != 5 or tuple(tactile_dd.shape[1:]) != (2, 3, 96, 128):
            raise ValueError(
                "Expected decoded tactile_dd [B,2,3,96,128], "
                f"got {tuple(tactile_dd.shape)}"
            )
        if not tactile_dd.is_floating_point():
            raise ValueError("tactile_dd must be floating point decoded D, Dx, Dy, not uint16/RGB.")
        dtype = self.finger_encoder[0].weight.dtype
        x = tactile_dd.to(device=self.channel_scale.device, dtype=dtype)
        x = x / self.channel_scale.to(dtype=dtype)
        batch_size = x.shape[0]
        fingers = self.finger_encoder(x.reshape(batch_size * 2, 3, 96, 128))
        return self.fusion(fingers.reshape(batch_size, -1))

    def forward(self, tactile_dd: torch.Tensor) -> torch.Tensor:
        # Only action conditioning; no classifier or auxiliary objective.
        return self.projector(self.encode(tactile_dd))
