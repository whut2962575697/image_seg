import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, input):
        return input*torch.tanh(F.softplus(input))