import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .config import bLlamaConfig
from .llama import Transformer
from typing import Tuple

class bLlama(pl.LightningModule):
    def __init__(
            self,
            config: bLlamaConfig,
        ):
        super().__init__()
        self.config = config
        self.model = Transformer(config)

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
    