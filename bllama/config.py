from dataclasses import dataclass

@dataclass
class bLlamaConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    hidden_size: int = 2048
    num_heads: int = 16
    num_layers: int = 24
    dropout: float = 0.0
    bias: bool = False

@dataclass
class trainerConfig:
    max_steps: int = 10000
    gpus: int = 1
    precision: int = 16
    gradient_clip_val: float = 1.0
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1
    limit_val_batches: float = 1.0
    limit_train_batches: float = 1.0
    num_warmup_steps: int = 1000