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