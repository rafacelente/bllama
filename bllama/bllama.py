import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .config import bLlamaConfig
from .transformer import Transformer
from .quantization import quantize_weights_to_int8
from typing import Tuple
from .bitlinear import BitLinear

class bLlama(pl.LightningModule):
    def __init__(
            self,
            config: bLlamaConfig,
        ):
        super().__init__()
        self.config = config
        self.model = Transformer(config)

    def quantize_weights_to_ternary(self, verbose=True):
        """
            Quantize all BitLinear layers to ternary in int8.
        """
        if verbose:
            size_before_quant = self.get_model_size_in_bytes()
            print(f'Mode size before quantization: {size_before_quant} MB')
        for name, layer in self.model.named_modules():
            if isinstance(layer, BitLinear):
                for k, v in layer.state_dict().items():
                    if 'weight' in k and 'norm' not in k:
                        w_quant, scale = quantize_weights_to_int8(v)
                        layer.weight.requires_grad = False
                        layer.weight.data = w_quant
                        layer.weight_scale = scale
        if verbose:
            size_after_quant = self.get_model_size_in_bytes()
            print(f'Mode size after quantization: {size_after_quant} MB')
            print(f'Quantization ratio: {size_after_quant / size_before_quant}')

    def get_model_size_in_bytes(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        size_all_mb = (param_size) / 1024**2
        return size_all_mb

    # FIXME: This is overly complicated, but works for now.
    def generate(
            self,
            prompt: str,
            tokenizer: callable,
            max_len: int = 50,
            do_sample: bool = True, 
            temperature: float = 0.1, 
            top_k: int = 0,
            repetition_penalty : float = 1.0, 
            num_return_sequences : int = 1, 
            device : str = "cuda",
        ):
        assert hasattr(tokenizer, 'encode'), f"Tokenizer {tokenizer.__name__} must have an encode method"
        assert hasattr(tokenizer, 'decode'), f"Tokenizer {tokenizer.__name__} must have an decode method"
        prompt_tokens = tokenizer.encode(prompt)
        self.model = self.model.to(device)
        self.model.eval()

        # asserting that all bitlinear layers have weight scales...
        unquantized_layers = [name for name, layer in self.model.named_modules() if isinstance(layer, BitLinear) and layer.weight_scale is None]
        if len(unquantized_layers):
            raise ValueError(f"Layers {unquantized_layers} have not been quantized to int8. Please call `quantize_weights_to_int8` before generating text.")
        
        def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value

            return logits

        for _ in range(num_return_sequences):
            generated = torch.tensor([prompt_tokens])
            generated = generated.to(device)

            for _ in range(max_len):
                with torch.no_grad():
                    outputs = self.model(generated, inference=True)
                    next_token_logits = outputs[:, -1, :]
                    for token in set(generated[0].tolist()):
                        next_token_logits[:, token] /= repetition_penalty
                    next_token_logits = next_token_logits / temperature
                    filtered_logits = top_k_filtering(next_token_logits, top_k=top_k)
                    if do_sample:
                        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    else:
                        next_token = torch.argmax(F.softmax(filtered_logits, dim=-1), dim=-1, keepdims=True)
                    generated = torch.cat((generated, next_token), dim=-1)

            result = generated[0].tolist()
            text = tokenizer.decode(result)
        return text
    
    def training_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int,
        ):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(
            self, 
            batch: Tuple[torch.Tensor], 
            batch_idx: int):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        return optimizer
    