import torch
from torch.utils.data import DataLoader


def is_correct(y_pred: type[torch.Tensor], y_true: type[torch.Tensor]) -> bool:
    """A prediction is correct if the predicted coordinates are within y_pred +- 0.05
    Args:
        y_pred (torch.Tensor): dim. (batch_size, 2)
        y_true (torch.Tensor): dim. (batch_size, 2)
    """

    dif = abs(y_pred - y_true)
    within_range = dif <= 0.05
    return torch.all(within_range, dim=1)


def compute_accuracy(model: callable, dataloader: type[DataLoader]) -> float:
    """Runs the model and computes the accuracy."""

    model.eval()
    model.cpu()

    result = []
    for x, y in dataloader:
        with torch.no_grad():
            y_pred = model(x)
        result.append(is_correct(y_pred, y))
    result = torch.hstack(result)
    return float(result.sum() / len(result))
