import torch
import torch.nn as nn

class POTModel(nn.Module):
    def __init__(self, 
                pot_process,
                vec_data: torch.Tensor, 
                marks: torch.Tensor):
        super().__init__()
        self.device = pot_process.device
        self.pot = pot_process
        self.marks = torch.tensor(marks, device=self.device, dtype=torch.float32)
        self.vec_data = torch.tensor(vec_data, device=self.device, dtype=torch.float32)
        self.loss_history = []
        
    def log_likelihood(self) -> torch.Tensor:
        return torch.sum(self.pot.log_density(self.vec_data, self.marks))
    
    def fit(self, lr=0.01, max_iter=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(max_iter):
            optimizer.zero_grad()
            loss = -self.log_likelihood()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.pot.apply_constraints()
            self.loss_history.append(loss.item())
        
        loss = -self.log_likelihood()
        self.loss_history.append(loss.item())
