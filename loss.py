from typing import Optional

from torch import Tensor, nn
from torch.nn import SmoothL1Loss, CrossEntropyLoss


class ILoss(nn.Module):

    def __init__(self, interval_size_average=None, interval_reduce=None, interval_reduction: str = 'mean',
                 interval_beta: float = 1.0) -> None:
        super().__init__()

        self.interval_loss = SmoothL1Loss(interval_size_average, interval_reduce, interval_reduction, interval_beta)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.interval_loss(input, target)


class ICLoss(ILoss):

    def __init__(self, num_classes, interval_weight, interval_size_average=None, interval_reduce=None,
                 interval_reduction: str = 'mean',
                 interval_beta: float = 1.0, class_weight: Optional[Tensor] = None, class_size_average=None,
                 class_ignore_index: int = -100, class_reduce=None, class_reduction: str = 'mean') -> None:
        super().__init__(interval_size_average, interval_reduce, interval_reduction, interval_beta)

        self.num_classes = num_classes
        self.interval_weight = interval_weight

        self.class_loss = CrossEntropyLoss(class_weight, class_size_average, class_ignore_index, class_reduce,
                                           class_reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_intervals, input_classes = input

        input_classes = input_classes.reshape(-1, self.num_classes)  # (16*10, 2) = (160, 2)
        target_classes = target[..., 2].reshape(-1).long()  # (16*10,) = (160,)
        class_loss = self.class_loss(input_classes, target_classes)

        target_intervals = target[..., :2]
        mask = (target[..., 2] > 0).unsqueeze(-1).expand(-1, -1, 2)
        interval_loss = super().forward(input_intervals[mask],
                                        target_intervals[mask]) if mask.any() else input_intervals.new_tensor(0.0)

        return self.interval_weight * interval_loss + class_loss
