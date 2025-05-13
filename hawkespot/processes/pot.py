import torch
import torch.nn as nn

class DynamicPOT(nn.Module):
    def __init__(self,
                sigma0: float = 1.0, 
                phi: float = 0.2, 
                xi: float = 0.15):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigma0 = nn.Parameter(torch.tensor(sigma0, device=self.device, dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor(phi, device=self.device, dtype=torch.float32))
        self.xi = nn.Parameter(torch.tensor(xi, device=self.device, dtype=torch.float32))
    
    def log_density(self, data: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma0 + self.phi * data
        z = marks / sigma
        log_arg = 1 + self.xi * z
        return -torch.log(sigma) - (1 + 1/self.xi) * torch.log(log_arg)

    def apply_constraints(self):
        with torch.no_grad():
            self.sigma0.data = torch.clamp(self.sigma0, min=1e-8)
            self.phi.data = torch.clamp(self.phi, min=0.0)
            self.xi.data = torch.clamp(self.xi, min=1e-8, max=0.49999999)
