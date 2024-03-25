import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .config import bLlamaConfig
from .llama import Transformer
from .quantization import quantize_weights_to_int8
from typing import Tuple

class bLlama(pl.LightningModule):
    def __init__(
            self,
            config: bLlamaConfig,
        ):
        super().__init__()
        self.config = config
        self.model = Transformer(config)

    def quantize_weights_to_int8(self):
        current_state_dict = self.model.state_dict()
        new_state_dict = {}
        list_of_bitlinear_layers = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "ff.w1", "ff.w2", "ff.w3"]
        for k, v in current_state_dict.items():
            quantize_weights = False
            for bitlinear_layer in list_of_bitlinear_layers:
                if bitlinear_layer in k:
                    quantize_weights = True
                    break
            new_state_dict[k] = quantize_weights_to_int8(v)[0] if quantize_weights else v
        self.model.load_state_dict(new_state_dict)

    def get_model_size_in_bytes(self, verbose=True):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        size_all_mb = (param_size) / 1024**2
        if verbose: 
            print(f'Model size:= {size_all_mb} MB')
        return size_all_mb

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
                    outputs = self.model(generated)
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
    