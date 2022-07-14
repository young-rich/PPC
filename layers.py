import math

import torch
import einops
from torch import nn, einsum, Tensor
from typing import Optional,Union,List

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        keep_attn_weights: bool,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super(TransformerEncoder, self).__init__()

        self.self_attn = MultiHeadedAttention(
            input_dim,
            n_heads,
            keep_attn_weights,
            dropout,
        )
        self.ff = PositionwiseFF(input_dim, ff_hidden_dim, dropout, activation)
        self.attn_addnorm = AddNorm(input_dim, dropout)
        self.ff_addnorm = AddNorm(input_dim, dropout)

    def forward(self, X: Tensor) -> Tensor:
        x = self.attn_addnorm(X, self.self_attn(X))
        return self.ff_addnorm(x, self.ff(x))

class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        keep_attn_weights: bool,
        dropout: float,
    ):
        super(MultiHeadedAttention, self).__init__()

        assert input_dim % n_heads == 0, "'input_dim' must be divisible by 'n_heads'"
        # Consistent with other implementations I assume d_v = d_k
        self.d_k = input_dim // n_heads
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.inp_proj = nn.Linear(input_dim, input_dim * 3)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.keep_attn_weights = keep_attn_weights

    def forward(self, X: Tensor) -> Tensor:
        # b: batch size, s: src seq length (num of categorical features
        # encoded as embeddings), l: target sequence (l = s), e: embeddings
        # dimensions, h: number of attention heads, d: d_k
        q, k, v = self.inp_proj(X).chunk(3, dim=2)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b s (h d) -> b h s d", h=self.n_heads),
            (q, k, v),
        )
        scores = einsum("b h s d, b h l d -> b h s l", q, k) / math.sqrt(self.d_k)
        attn_weights = self.dropout(scores.softmax(dim=-1))
        if self.keep_attn_weights:
            self.attn_weights = attn_weights
        attn_output = einsum("b h s l, b h l d -> b h s d", attn_weights, v)
        output = einops.rearrange(attn_output, "b h s d -> b s (h d)", h=self.n_heads)

        return self.out_proj(output)

class PositionwiseFF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ff_hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super(PositionwiseFF, self).__init__()
        self.w_1 = nn.Linear(
            input_dim, ff_hidden_dim * 2 if activation == "geglu" else ff_hidden_dim
        )
        self.w_2 = nn.Linear(ff_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, X: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(X))))


class AddNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return self.ln(self.dropout(Y) + X)

class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        dropout: Optional[Union[float, List[float]]],
        batchnorm: bool,
        batchnorm_last: bool,
        linear_first: bool,

    ):
        super(MLP, self).__init__()

        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)

def dense_layer(
    inp: int,
    out: int,
    activation: str,
    p: float,
    bn: bool,
    linear_first: bool,
):
    act_fn = nn.ReLU(inplace=True)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  # type: ignore[arg-type]
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)
