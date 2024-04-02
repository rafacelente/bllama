import torch.nn as nn
import torch.nn.functional as F
from .utils import RMSNorm
from .quantization import weight_quant, activation_quant, activation_post_quant
from typing import Optional

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.rms_norm = RMSNorm(in_features)
        self.weight_scale = None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, inference: Optional[bool]=False):
        w = self.weight
        if not inference:
            x_norm = self.rms_norm(x)
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            return F.linear(x_quant, w_quant, self.bias)
        else:
            # in case of inference, the weights are offline quantized to int8, so we assume w = w_quant
            x_norm = self.rms_norm(x)
            x_quant, x_scale = activation_post_quant(x_norm)
            w_scale = self.weight_scale
            # according to the paper, this linear layer may have to be replaced by a gemm_lowbit_kernel,
            # but no such kernel is available, nor any directions on how to implement it, so we'll just use linear
            return F.linear(x_quant, w.float(), self.bias) / (x_scale * w_scale)