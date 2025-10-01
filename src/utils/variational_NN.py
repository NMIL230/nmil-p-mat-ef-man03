import sys

sys.path.append("../")
import torch
from torch import nn
import os

os.environ["WANDB_SILENT"] = "True"


class variationalNN(nn.Module):
    def __init__(self, n, latent_dim, out_sz):
        super(variationalNN, self).__init__()
        self.meu_z = torch.nn.Parameter(torch.randn(n, latent_dim))
        self.sigma_z = torch.nn.Parameter(torch.randn(n, latent_dim))
        self.fc1 = nn.Linear(latent_dim, out_sz)
        self.fc2 = nn.Linear(out_sz, out_sz)
        self.fc3 = nn.Linear(out_sz, out_sz)
        self.out_sz = out_sz

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x