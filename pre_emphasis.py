import torch.nn as nn
import torch.nn.functional as F
import torch

class PreEmphasis(nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))
        

    def forward(self, input: torch.tensor) -> torch.tensor:
        device = input.device
        self.flipped_filter = self.flipped_filter.to(device)
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)
