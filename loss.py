from typing import Optional

from torch import Tensor
from torch.nn import SmoothL1Loss, CrossEntropyLoss


class IntervalClassLoss:

    def __init__(self, interval_size_average=None, interval_reduce=None, interval_reduction: str = 'mean',
                 interval_beta: float = 1.0, class_weight: Optional[Tensor] = None, class_size_average=None,
                 class_ignore_index: int = -100, class_reduce=None, class_reduction: str = 'mean') -> None:
        self.interval_loss = SmoothL1Loss(interval_size_average, interval_reduce, interval_reduction, interval_beta)
        self.class_loss = CrossEntropyLoss(class_weight, class_size_average, class_ignore_index, class_reduce,
                                           class_reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        print(input, target)
        interval_loss = self.interval_loss.forward(input, target)
        class_loss = self.class_loss.forward(input, target)
        return interval_loss + class_loss
