import torch
import torch.nn as nn

class MarkedHawkesProcess(nn.Module):
    def __init__(self, temporal_kernel,
                mark_kernel,
                base_intensity: float = 0.1,
                alpha: float = 0.5):
        super().__init__()
        self.device = temporal_kernel.device
        self.temporal_kernel = temporal_kernel
        self.mark_kernel = mark_kernel
        self.base_intensity = nn.Parameter(torch.tensor(base_intensity, device=self.device, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha, device=self.device, dtype=torch.float32))
    
    def intensity(self, time_diff_filtered: torch.Tensor, marks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total_impact = self.temporal_kernel.intensity(time_diff_filtered) * self.mark_kernel.mark(marks) * mask
        return self.base_intensity + self.alpha * torch.sum(total_impact, dim=1)

    def integral(self, t_start: torch.Tensor, t_end: torch.Tensor, history: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_kernel.integral(t_end - history) - self.temporal_kernel.integral(t_start)
        return self.base_intensity * (t_end - t_start) + self.alpha * torch.sum(temporal * self.mark_kernel.mark(marks))

    def apply_constraints(self):
        with torch.no_grad():
            self.base_intensity.data = torch.clamp(self.base_intensity, min=1e-8)
            self.alpha.data = torch.clamp(self.alpha, min=0.0)
        self.temporal_kernel.apply_constraints()
        self.mark_kernel.apply_constraints()
