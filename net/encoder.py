import torch
from torch import nn
import net.utils


class GMEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.input_dim = 15
        self.sin = net.utils.CatMlpSineLayer(self.input_dim, d_model, is_first=True)

    def forward(self, x, mask=None):
        output = self.sin(x)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, dtype=torch.bool)
            output[mask] = 0
        return output


class PointEncoder(nn.Module):
    def __init__(self, d_point):
        super().__init__()
        self.input_dim = 3
        self.sin = net.utils.CatSineLayer(self.input_dim, d_point, is_first=True)

    def forward(self, x):
        output = self.sin(x)
        return output
