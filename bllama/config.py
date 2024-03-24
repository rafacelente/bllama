from dataclasses import dataclass

@dataclass
class bLlamaConfig:
    vocab_size: int = 50257
    seq_len: int = 1024
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    batch_size: int = 8
    bias: bool = False