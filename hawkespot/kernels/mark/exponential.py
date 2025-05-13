import torch
import torch.nn as nn
from .base import MarkKernel

class ExponentialMarkKernel(MarkKernel):
    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, device=self.device, dtype=torch.float32))
        
    def mark(self, marks: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.gamma * marks)

    def apply_constraints(self):
        with torch.no_grad():
            self.gamma.data = torch.clamp(self.gamma, min=0.0)
