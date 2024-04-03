import pytorch_lightning as pl
import torch
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import wandb

class CalculateDistributionCallback(pl.Callback):
    def __init__(
            self, 
            every_n_step: Optional[int] = 5000,
            to_wandb: Optional[bool] = False,
            save_path: Optional[str] = None,
            ):
        self.every_n_step = every_n_step
        self.to_wandb = to_wandb
        self.save_path = save_path

    def on_batch_end(self, trainer: pl.Trainer):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            self._produce_distribution(trainer)
        
    @torch.no_grad()
    def _quantize_weights_to_int8_with_no_clamp(w: torch.Tensor):
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w_quant = (w * scale).round()
        w_quant.requires_grad = False
        return w_quant, scale

    @torch.no_grad()
    def _produce_distribution(
            self,
            trainer: pl.Trainer,
            ):
        model = trainer.model.model.copy()
        fig, axs = plt.subplots(4,4, figsize=(10,10))

        i = 0
        for k,v in model.state_dict().items():
            if "attn.wq" in k and "weight" in k and "rms_norm" not in k:
                w_quant, _ = self._quantize_weights_to_int8_with_no_clamp(v)
                line = int(i / 4)
                column = i % 4
                counts, bins = np.histogram(w_quant.view(1,-1).cpu().numpy(), bins=20)
                axs[line, column].stairs(counts, bins, fill=True)
                axs[line, column].axvline(x=1, color='r', ls='--')
                axs[line, column].axvline(x=-1, color='r', ls='--')
                axs[line, column].set_title(f"Quantized wq layer {i}")
                i += 1
        plt.suptitle(f"Weight distribution of quantized wq layers at step {trainer.global_step}")
        if self.to_wandb:
            wandb.log({"weight_distribution": fig})
        if self.save_path is not None:
            plt.savefig(self.save_path)