from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


VENDOR = Path(__file__).resolve().parent / "vendor" / "lithobench"


class Simulator(nn.Module):
    canvas_size = 2048
    pixel_size_nm = 1.0
    threshold = 0.225

    def __init__(self, device="cpu", kernel_chunk=4):
        super().__init__()
        if not 1 <= kernel_chunk <= 24:
            raise ValueError("kernel_chunk must be between 1 and 24")
        self.kernel_chunk = kernel_chunk
        for name in ("focus", "defocus"):
            kernel = (
                torch.load(
                    VENDOR / "kernel" / "kernels" / f"{name}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                .permute(2, 0, 1)
                .contiguous()
            )
            scales = torch.load(
                VENDOR / "kernel" / "scales" / f"{name}.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.register_buffer(f"{name}_kernel", kernel)
            self.register_buffer(f"{name}_scales", scales)
        self.to(device)

    @torch.no_grad()
    def aerial(self, mask, dose=1.0, defocus=False):

        squeeze = mask.ndim == 2
        if squeeze:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise ValueError("mask must have shape [B,H,W] or [H,W]")
        height, width = mask.shape[-2:]
        if not (1 <= height <= self.canvas_size and 1 <= width <= self.canvas_size):
            raise ValueError("mask sides must be between 1 and 2048 pixels")
        if dose <= 0:
            raise ValueError("dose must be positive")
        name = "defocus" if defocus else "focus"
        kernel = getattr(self, f"{name}_kernel")
        scales = getattr(self, f"{name}_scales")
        mask = mask.to(device=kernel.device, dtype=torch.float32)
        top = (self.canvas_size - height) // 2
        left = (self.canvas_size - width) // 2
        padded = F.pad(
            mask,
            (
                left,
                self.canvas_size - width - left,
                top,
                self.canvas_size - height - top,
            ),
        )
        complex_mask = padded.to(torch.complex64) * dose
        spectrum = torch.fft.fft2(complex_mask, norm="backward") / (self.canvas_size**2)
        image = torch.zeros_like(padded)

        half_y, half_x = kernel.shape[-2] // 2, kernel.shape[-1] // 2
        for start in range(0, len(scales), self.kernel_chunk):
            stop = min(start + self.kernel_chunk, len(scales))
            part = kernel[start:stop]
            field = torch.zeros(
                (mask.shape[0], stop - start, self.canvas_size, self.canvas_size),
                dtype=torch.complex64,
                device=mask.device,
            )
            field[:, :, : half_y + 1, : half_x + 1] = (
                spectrum[:, None, : half_y + 1, : half_x + 1]
                * part[None, :, -(half_y + 1) :, -(half_x + 1) :]
            )
            field[:, :, : half_y + 1, -half_x:] = (
                spectrum[:, None, : half_y + 1, -half_x:]
                * part[None, :, -(half_y + 1) :, :half_x]
            )
            field[:, :, -half_y:, : half_x + 1] = (
                spectrum[:, None, -half_y:, : half_x + 1]
                * part[None, :, :half_y, -(half_x + 1) :]
            )
            field[:, :, -half_y:, -half_x:] = (
                spectrum[:, None, -half_y:, -half_x:] * part[None, :, :half_y, :half_x]
            )
            coherent = torch.fft.ifft2(field, norm="forward")
            image += (
                scales[None, start:stop, None, None] * coherent.abs().square()
            ).sum(dim=1)
        result = image[:, top : top + height, left : left + width]
        return result[0] if squeeze else result

    def forward(self, mask, dose=1.0, defocus=False):

        return self.aerial(mask, dose=dose, defocus=defocus) > self.threshold
