import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear
from .utils import apply_rotary_emb, precompute_freqs_cis, RMSNorm
from .config import bLlamaConfig


class Attention(nn.Module):
    def __init__(self, config: bLlamaConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        self.n_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.attn_dropout = config.dropout
        self.proj_dropout = config.dropout
        self.flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # consider self.n_heads == self.n_kv_heads
        

        self.wq = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.wk = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.wv = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.wo = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_drop = nn.Dropout(self.proj_dropout)

    def forward(
            self, 
            x: torch.Tensor,
            freq_cis: torch.Tensor,
            mask: torch.Tensor = None,
            inference: bool = False,
        ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x, inference), self.wk(x, inference), self.wv(x, inference)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freq_cis)

        keys = xk.transpose(1,2)
        values = xv.transpose(1,2)
        queries = xq.transpose(1,2)

        if self.flash_attention:
            attn_output = F.scaled_dot_product_attention(
                queries, 
                keys, 
                values,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
                )
        else:
            attn = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = attn + mask[:,:, :seqlen, :seqlen]
            attn = F.softmax(attn.float(), dim=-1).type_as(queries)
            attn_output = torch.matmul(attn, values)

        attn_output = self.attn_drop(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_output, inference)
    
class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
        ):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim)
        self.w2 = BitLinear(hidden_dim, dim)
        self.w3 = BitLinear(dim, hidden_dim)

    def forward(self, x, inference: bool = False):
        return self.w2(F.silu(self.w1(x, inference)) * self.w3(x, inference), inference=inference)
    
class TransformerBlock(nn.Module):
    def __init__(
            self,
            config: bLlamaConfig,
        ):
        super().__init__()
        self.attn = Attention(config)
        self.ff = FeedForward(config.hidden_size, config.hidden_size * 4)
        # self.attn_norm = RMSNorm(config.hidden_size)
        # self.ff_norm = RMSNorm(config.hidden_size)
        # BitLinear has built-in RMSNorm, but I'll keep as Identity in case
        # we want to make it a choice to use BitLinear or Linear later
        self.attn_norm = nn.Identity()
        self.ff_norm = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            freq_cis: torch.Tensor,
            inference: bool = False,
        ):
        h = x + self.attn(self.attn_norm(x), freq_cis, mask=None, inference=inference)
        return h + self.ff(self.ff_norm(h), inference=inference)
    
class Transformer(nn.Module):
    """
    Transformer module based on the LLaMa architecture.

    Args:
        config (bLlamaConfig): Configuration object containing model hyperparameters.

    Attributes:
        config (bLlamaConfig): Configuration object containing model hyperparameters.
        embed (nn.Embedding): Embedding layer for input tokens.
        blocks (nn.ModuleList): List of TransformerBlock instances.
        norm (RMSNorm): Normalization layer.
        freq_cis (torch.Tensor): Precomputed frequency tensor.
        vocab_proj (nn.Linear): Linear projection layer for output vocabulary.
        mask (torch.Tensor): Mask for attention mechanism.
    """

    def __init__(
            self,
            config: bLlamaConfig,
        ):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.freq_cis = precompute_freqs_cis(config.hidden_size // config.num_heads, config.seq_len * 2)
        self.vocab_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed.weight = self.vocab_proj.weight # tie weights

        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            mask = torch.full((1,1,config.seq_len,config.seq_len), float('-inf'), device=self.freq_cis.device, dtype=self.freq_cis.dtype)
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)
        else:
            mask = None


    def forward(
            self,
            x: torch.Tensor,
            inference: bool = False,
        ):
        """
        Forward pass of the Transformer module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            inference (bool, optional): Flag indicating whether the forward pass is for inference or training. This is needed because the BitLinear layers have different behavior during inference.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).

        """
        bsz, seqlen = x.shape
        x = self.embed(x)
        freq_cis = self.freq_cis[:seqlen].to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, freq_cis, inference=inference)
        x = self.norm(x)
        return self.vocab_proj(x)
    
