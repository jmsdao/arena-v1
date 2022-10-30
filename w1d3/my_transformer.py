from dataclasses import dataclass

import torch as t
import torch.nn as nn

from fancy_einsum import einsum


def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    '''
    batch = Q.shape[0]
    seq_len = Q.shape[1]
    headsize = Q.shape[2] // num_heads

    Q = Q.reshape(batch, seq_len, num_heads, headsize)
    K = K.reshape(batch, seq_len, num_heads, headsize)
    V = V.reshape(batch, seq_len, num_heads, headsize)

    scale = t.sqrt(t.tensor(K.shape[-1]).type(t.float32))
    raw_attention_filter = einsum('b sl_Q nh hs, b sl_K nh hs -> b nh sl_Q sl_K', Q, K)
    mask_filter = t.triu(t.full_like(raw_attention_filter, -t.inf), 1)
    masked_attention_filter = t.softmax((raw_attention_filter + mask_filter) / scale, dim=-1)
    attention_values = einsum('b nh sl_Q sl_K, b sl_K nh hs -> b sl_Q nh hs', masked_attention_filter, V)
    return attention_values.reshape(batch, seq_len, num_heads * headsize)


class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        assert hidden_size % num_heads == 0, "num_heads should be divisible by hidden_size"
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.W_O = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        headsize = self.hidden_size // self.num_heads

        QKV = self.W_QKV(x)        
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:2*self.hidden_size]
        V = QKV[..., 2*self.hidden_size:3*self.hidden_size]
        attention_values = multihead_masked_attention(Q, K, V, self.num_heads)
        return self.W_O(attention_values)


@dataclass
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''
    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_seq_len: int = 5000):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        position = t.arange(max_seq_len).unsqueeze(1)
        denominator = 10000 ** (2 * t.arange(embedding_dim // 2).type(t.float32) / embedding_dim)

        pe = t.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = t.sin(position / denominator)
        pe[:, 1::2] = t.cos(position / denominator)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq_len, embedding_dim)
        '''
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]
    
    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f'embedding_dim={self.embedding_dim}, max_seq_len={self.max_seq_len}'


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.hidden_size, 4*config.hidden_size),
            nn.GELU(),
            nn.Linear(4*config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (..., hidden_size)

        Return: shape (..., hidden_size)
        '''
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.att = MultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.ln1(self.att(x))
        return x + self.ln2(self.mlp(x))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = PositionalEncoding(config.hidden_size, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.final_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x is a tensor of token ids

        x: shape (batch, seq_len)

        Return: shape (batch, seq_len, vocab_size)
        '''
        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        x = self.decoder_blocks(x)
        x = self.final_ln(x)
        x = x @ self.token_embedding.weight.T
        return x
