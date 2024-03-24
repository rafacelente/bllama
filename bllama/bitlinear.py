import torch.nn as nn
import torch.nn.functional as F
from .utils import RMSNorm

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1,1) / scale
    return u

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    u = (x * scale).round().clamp_(-128,127) / scale
    return u

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.rms_norm = RMSNorm(in_features)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        w = self.weight
        x_norm = self.rms_norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant, self.bias)
        return y