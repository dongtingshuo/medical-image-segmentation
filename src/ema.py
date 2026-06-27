from __future__ import annotations

import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {self.decay}")
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                self.shadow[name] = value.detach().clone()
            elif value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    def store(self, model):
        self.backup = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("EMA restore called before store.")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.shadow = {name: value.detach().clone() for name, value in state["shadow"].items()}
