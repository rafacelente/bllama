import torch

def weight_quant(w: torch.Tensor):
    """
    Quantize a set of weights to ternary based on the mean of the absolute values described by
    https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

    This doesn't change the type of the weights, but rather the values themselves.
    Args:
        w (torch.Tensor): weights
    Returns:
        u (torch.Tensor): quantized weights
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1,1) / scale
    return u

def activation_quant(x: torch.Tensor):
    """
    Quantize the activation tensor to scaled 8 bit based on the mean of the absolute values described by
    https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

    This doesn't change the type of the activations, but rather the values themselves.
    Args:
        x (torch.Tensor): activations
    Returns:
        u (torch.Tensor): quantized weights
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    u = (x * scale).round().clamp_(-128,127) / scale
    return u

def activation_post_quant(x_norm: torch.Tensor):
    """
    Quantize the layer-normalized activations to 8 bit based on the mean of the absolute values described by
    https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

    In constrast with the add-on of the paper, this function does not do its own RMSNorm. In turn, the 
    RMSNorm will be done directly on the BitLinear layer.

    This doesn't change the type of the activations, but rather the values themselves.
    Args:
        w (torch.Tensor): weights
    Returns:
        y (torch.Tensor): quantized activations
        scale (torch.Tensor): scale factor
    """
    scale = 127.0 / x_norm.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x_norm * scale).round().clamp_(-128,127)
    return y, scale

def quantize_weights_to_int8(w: torch.Tensor):
    """
    Offline quantization of a set of weights to int8 based on the mean of the absolute values.

    This operation casts the weights to int8.
    Args:
        w (torch.Tensor): weights
    Returns:
        w_quant (torch.Tensor): quantized weights
        scale (torch.Tensor): scale factor
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    w_quant = (w * scale).round().clamp_(-1,1).to(torch.int8)
    w_quant.requires_grad = False
    return w_quant, scale