import torch
import torch.nn as nn

class TemporalKernel(nn.Module):
    """Абстрактный базовый класс для временных ядер"""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def intensity(self, t_diff: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def integral(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_constraints(self):
        pass
