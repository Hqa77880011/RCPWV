from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spectral_simulator import SpectralSimulator, embed_band, synthesize_intensity


def mask_lowpass_grid(mask_band: torch.Tensor, grid_size: int = 128) -> torch.Tensor:

    return torch.fft.ifft2(embed_band(mask_band, grid_size), norm="forward").real


class LowRankResidualSurrogate(nn.Module):
    def __init__(self, kernels: int = 4, width: int = 16) -> None:
        super().__init__()
        if kernels not in (0, 1, 4) or width < 1:
            raise ValueError("kernels must be 0, 1 or 4 and width must be positive")
        self.kernels = kernels
        self.width = width
        self.physics = SpectralSimulator(device="cpu")
        layers = []
        channels = kernels + 2
        for dilation in (1, 2, 4, 8):
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        width,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        padding_mode="circular",
                    ),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
            )
            channels = width
        layers.append(nn.Conv2d(width, 1, 1))
        self.residual = nn.Sequential(*layers)
        nn.init.normal_(self.residual[-1].weight, std=0.01)
        nn.init.constant_(self.residual[-1].bias, -3.0)

    def _raw_grid(self, features: torch.Tensor, conditions) -> torch.Tensor:
        modes = features[:, : self.kernels].float()
        mask_low = features[:, -1:].float()
        condition_plane = torch.as_tensor(
            conditions, device=features.device, dtype=torch.float32
        )
        if condition_plane.ndim == 0:
            condition_plane = condition_plane.expand(len(features))
        condition_plane = condition_plane.reshape(-1, 1, 1, 1).expand(-1, 1, 128, 128)
        inputs = torch.cat(
            (modes / self.physics.threshold, mask_low, condition_plane), dim=1
        )
        correction = (
            F.softplus(self.residual(inputs).float())[:, 0] * self.physics.threshold
        )

        return modes.sum(dim=1) + correction

    def forward_features(self, features: torch.Tensor, conditions=0) -> torch.Tensor:

        raw = self._raw_grid(features, conditions)
        return synthesize_intensity(raw, size=self.physics.grid_size)

    def extract_features(
        self, mask: torch.Tensor, defocus: bool = False
    ) -> torch.Tensor:

        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        band = self.physics.mask_band(mask)
        mask_low = mask_lowpass_grid(band, self.physics.grid_size)[:, None]
        if self.kernels == 0:
            return mask_low
        name = "defocus" if defocus else "focus"
        kernel = getattr(self.physics, f"{name}_kernel")[: self.kernels]
        scales = getattr(self.physics, f"{name}_scales")[: self.kernels]
        fields = torch.fft.ifft2(
            embed_band(band[:, None] * kernel[None], self.physics.grid_size),
            norm="forward",
        )
        modes = fields.abs().square() * scales[None, :, None, None]
        return torch.cat((modes, mask_low), dim=1)

    def aerial_grid(
        self, mask: torch.Tensor, dose: float = 1.0, defocus: bool = False
    ) -> torch.Tensor:

        if dose <= 0:
            raise ValueError("dose must be positive")
        features = self.extract_features(mask, defocus=defocus)
        conditions = torch.full((len(features),), int(defocus), device=features.device)
        return self.forward_features(features, conditions) * dose**2

    def aerial(
        self, mask: torch.Tensor, dose: float = 1.0, defocus: bool = False
    ) -> torch.Tensor:

        if dose <= 0:
            raise ValueError("dose must be positive")
        squeeze = mask.ndim == 2
        height, width = mask.shape[-2:]
        features = self.extract_features(mask, defocus=defocus)
        conditions = torch.full((len(features),), int(defocus), device=features.device)
        raw = self._raw_grid(features, conditions)
        full = synthesize_intensity(raw, size=self.physics.canvas_size) * dose**2
        top = (self.physics.canvas_size - height) // 2
        left = (self.physics.canvas_size - width) // 2
        crop = full[:, top : top + height, left : left + width]
        return crop[0] if squeeze else crop

    def forward(
        self, mask: torch.Tensor, dose: float = 1.0, defocus: bool = False
    ) -> torch.Tensor:
        return self.aerial(mask, dose=dose, defocus=defocus)


def load_model(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[LowRankResidualSurrogate, dict]:

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = LowRankResidualSurrogate(kernels=config["kernels"], width=config["width"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
