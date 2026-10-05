import math

import torch


class WarmupStableDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 1e-5,
        base_lr: float = 1e-3,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps

        self.min_lr = min_lr
        self.base_lr = base_lr
        self.total_steps = warmup_steps + stable_steps + decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            factor = step / self.warmup_steps
            return [self.min_lr + factor * (self.base_lr - self.min_lr) for _ in self.optimizer.param_groups]
        elif step < self.warmup_steps + self.stable_steps:
            return [self.base_lr for _ in self.optimizer.param_groups]
        elif step < self.total_steps:
            decay_progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
            factor = 1 - math.sqrt(decay_progress)
            return [self.min_lr + factor * (self.base_lr - self.min_lr) for _ in self.optimizer.param_groups]
        else:
            return [self.min_lr for _ in self.optimizer.param_groups]
