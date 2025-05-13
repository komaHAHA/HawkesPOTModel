import torch
import torch.nn as nn

class HawkesModel(nn.Module):
    def __init__(self, hawkes_process, start: float, end: float, vec_times: torch.Tensor):
        super().__init__()
        self.hawkes = hawkes_process
        self.device = hawkes_process.device
        self.start = torch.tensor(start, device=self.device, dtype=torch.float32)
        self.end = torch.tensor(end, device=self.device, dtype=torch.float32)
        self.vec_times = torch.tensor(vec_times, device=self.device, dtype=torch.float32)
        self.data_matr = self.vec_times.unsqueeze(1) - self.vec_times.unsqueeze(0)
        self.mask = self.data_matr > 0
        self.data_matr_filtered = self.data_matr * self.mask
        self.loss_history = []
        
    def log_likelihood(self) -> torch.Tensor:
        intensities = self.hawkes.intensity(self.data_matr_filtered, self.mask)
        hawkes_ll = torch.sum(torch.log(intensities)) - self.hawkes.integral(self.start, self.end, self.vec_times)
        return hawkes_ll
    
    def fit(self, lr=0.01, max_iter=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(max_iter):
            optimizer.zero_grad()
            loss = -self.log_likelihood()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.hawkes.apply_constraints()
            self.loss_history.append(loss.item())
        
        loss = -self.log_likelihood()
        self.loss_history.append(loss.item())
