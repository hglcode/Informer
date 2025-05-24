import torch


class TriangularCausalMask:
    def __init__(self, b: int, l: int, device: str = "cpu") -> None:
        mask_shape = [b, 1, l, l]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    def __init__(self, b: int, h: int, l: int, index: torch.Tensor, scores: torch.Tensor, device: str = "cpu") -> None:
        _mask = torch.ones(l, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(b, h, l, scores.shape[-1])
        indicator = _mask_ex[torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask
