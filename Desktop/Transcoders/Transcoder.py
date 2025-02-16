import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

#have to fix MLP to make it more like the original paper

class Transcoders(nn.Module):
  def __init__(self, sparsity_penalty,input_dim: int, expansion_factor: float = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)
        self.sparsity_penalty = 1e-2

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        MLP= nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.input_dim, bias=True),
        )

        loss= (((MLP(x) - decoded) ** 2).mean(dim=1) / (x**2).mean(dim=1)
      ).mean()+self.sparsity_penalty*torch.norm(encoded, p=1)
        return decoded, encoded, loss
     
  def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.encoder(x))
            
  def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)