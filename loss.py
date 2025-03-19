from typing import Optional

from torch import Tensor, nn
from torch.nn import SmoothL1Loss, CrossEntropyLoss


class IntervalClassLoss(nn.Module):

    def __init__(self, interval_size_average=None, interval_reduce=None, interval_reduction: str = 'mean',
                 interval_beta: float = 1.0, class_weight: Optional[Tensor] = None, class_size_average=None,
                 class_ignore_index: int = -1, class_reduce=None, class_reduction: str = 'mean') -> None:
        super().__init__()

        self.interval_loss = SmoothL1Loss(interval_size_average, interval_reduce, interval_reduction, interval_beta)
        self.class_loss = CrossEntropyLoss(class_weight, class_size_average, class_ignore_index, class_reduce,
                                           class_reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_interval = input[0]
        target_interval = target[..., :2]
        interval_loss = self.interval_loss(input_interval, target_interval)

        input_class = input[1].reshape(-1, 2)  # (16*10, 2) = (160, 2)
        target_class = target[..., 2].reshape(-1).long()  # (16*10,) = (160,)
        class_loss = self.class_loss(input_class, target_class)

        return interval_loss + class_loss
