import torch
import torch.nn as nn
from .base import TemporalKernel

class ExponentialTemporalKernel(TemporalKernel):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta, device=self.device, dtype=torch.float32))
        
    def intensity(self, t_diff: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.beta * t_diff)

    def integral(self, t_diff: torch.Tensor) -> torch.Tensor:
        return (1 - torch.exp(-self.beta * t_diff)) / self.beta

    def apply_constraints(self):
        with torch.no_grad():
            self.beta.data = torch.clamp(self.beta, min=1e-8)
