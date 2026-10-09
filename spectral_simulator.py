import torch
from torch.nn import functional as F

from simulator import Simulator


def fft_forward(value):

    return torch.fft.fft2(value, norm="backward") / (value.shape[-2] * value.shape[-1])


def centered_band(spectrum, radius):

    indices_y = (
        torch.arange(-radius, radius + 1, device=spectrum.device) % spectrum.shape[-2]
    )
    indices_x = (
        torch.arange(-radius, radius + 1, device=spectrum.device) % spectrum.shape[-1]
    )
    return spectrum.index_select(-2, indices_y).index_select(-1, indices_x)


def embed_band(band, size):

    radius = band.shape[-1] // 2
    indices = torch.arange(-radius, radius + 1, device=band.device) % size
    result = band.new_zeros((*band.shape[:-2], size, size))
    result[..., indices[:, None], indices[None, :]] = band
    return result


def synthesize_intensity(image, size=2048):

    band = centered_band(fft_forward(image), 34)
    return torch.fft.ifft2(embed_band(band, size), norm="forward").real


class SpectralSimulator(Simulator):
    grid_size = 128

    def mask_band(self, mask):
        if mask.ndim == 2:
            mask = mask[None]
        height, width = mask.shape[-2:]
        if mask.ndim != 3 or not (1 <= height <= 2048 and 1 <= width <= 2048):
            raise ValueError("Expected B,H,W masks with dimensions at most 2048")
        mask = mask.to(self.focus_kernel.device, dtype=torch.float32)
        top, left = (2048 - height) // 2, (2048 - width) // 2
        padded = F.pad(mask, (left, 2048 - width - left, top, 2048 - height - top))
        return centered_band(fft_forward(padded), 17)

    def coherent_intensities(self, mask, defocus=False, kernels=24):

        if not 1 <= kernels <= 24:
            raise ValueError("kernels must be between 1 and 24")
        band = self.mask_band(mask)
        name = "defocus" if defocus else "focus"
        kernel = getattr(self, f"{name}_kernel")[:kernels]
        scales = getattr(self, f"{name}_scales")[:kernels]
        fields = torch.fft.ifft2(
            embed_band(band[:, None] * kernel[None], self.grid_size), norm="forward"
        )
        return fields.abs().square() * scales[None, :, None, None]

    @torch.no_grad()
    def aerial_grid(self, mask, dose=1.0, defocus=False, kernels=24):
        if dose <= 0:
            raise ValueError("dose must be positive")
        return self.coherent_intensities(mask, defocus, kernels).sum(1) * dose**2

    @torch.no_grad()
    def aerial(self, mask, dose=1.0, defocus=False, kernels=24):
        squeeze = mask.ndim == 2
        height, width = mask.shape[-2:]
        full = synthesize_intensity(self.aerial_grid(mask, dose, defocus, kernels))
        top, left = (2048 - height) // 2, (2048 - width) // 2
        result = full[:, top : top + height, left : left + width]
        return result[0] if squeeze else result
