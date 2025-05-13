import torch
import torch.nn as nn

class MarkKernel(nn.Module):
    """Базовый класс для ядер меток"""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def mark(self, marks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_constraints(self):
        pass  # Базовый метод для ограничений
