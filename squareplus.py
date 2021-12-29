import torch
import torch.nn as nn


def squareplus(x, b=1.52382103):
    return torch.mul(0.5, torch.add(x, torch.sqrt(torch.add(torch.square(x), b))))


class SquarePlus(nn.Module):
    def __init__(self, b=1.52382103):
        super().__init__()
        self.b = b

    def forward(self, x):
        return squareplus(x, self.b)
