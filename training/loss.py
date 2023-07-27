import torch


class Loss:
    """Loss function for training: sum of coordinates absolute error."""

    def __init__(self):
        pass

    def __call__(
        self, x: type[torch.Tensor], y: type[torch.Tensor]
    ) -> type[torch.Tensor]:
        return torch.sum(torch.abs(x - y), axis=1)
