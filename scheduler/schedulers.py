from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1,
                 gamma=0.9, last_epoch=-1):
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]
