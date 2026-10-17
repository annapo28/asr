import math

import torch
from torch import nn


class ConvSubsampling(nn.Module):
    def __init__(self, n_feats, d_model, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        freq = ((n_feats - 1) // 2 - 1) // 2
        self.linear = nn.Linear(d_model * freq, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x = self.conv(x.unsqueeze(1))
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        lengths = ((lengths - 1) // 2 - 1) // 2
        return self.dropout(self.linear(x)), lengths


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.pos = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def rel_shift(x):
        b, h, t, n = x.shape
        x = torch.cat([x.new_zeros(b, h, t, 1), x], dim=-1)
        x = x.view(b, h, n + 1, t)[:, :, 1:].view(b, h, t, n)
        return x[..., :t]

    def forward(self, x, pos_emb, mask):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.d_head)
        k = self.k(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        p = self.pos(pos_emb).view(-1, self.n_heads, self.d_head).permute(1, 2, 0)

        content = (q + self.pos_bias_u).transpose(1, 2) @ k.transpose(-2, -1)
        position = self.rel_shift((q + self.pos_bias_v).transpose(1, 2) @ p)
        scores = (content + position) / math.sqrt(self.d_head)

        scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
        attn = self.dropout(scores.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        return self.out(out)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = RelPositionMultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, mask):
        return self.dropout(self.attention(self.norm(x), pos_emb, mask))


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding="same", groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.glu(self.pointwise_in(self.norm(x).transpose(1, 2)))
        x = x.masked_fill(mask[:, None, :], 0.0)
        x = self.swish(self.batch_norm(self.depthwise(x)))
        return self.dropout(self.pointwise_out(x).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_expansion, kernel_size, dropout):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, mask):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, pos_emb, mask)
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Conformer(nn.Module):
    def __init__(
        self,
        n_feats,
        n_tokens,
        d_model=144,
        n_layers=16,
        n_heads=4,
        ff_expansion=4,
        kernel_size=32,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.subsampling = ConvSubsampling(n_feats, d_model, dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads, ff_expansion, kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(d_model, n_tokens)

    def relative_positions(self, t, device):
        pos = torch.arange(t - 1, -t, -1, device=device, dtype=torch.float32)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(2 * t - 1, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos[:, None] * div)
        pe[:, 1::2] = torch.cos(pos[:, None] * div)
        return pe

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = torch.log(spectrogram + 1e-6)
        x, lengths = self.subsampling(x, spectrogram_length)

        t = x.shape[1]
        mask = (
            torch.arange(t, device=x.device)[None, :] >= lengths.to(x.device)[:, None]
        )
        pos_emb = self.relative_positions(t, x.device)

        for block in self.blocks:
            x = block(x, pos_emb, mask)

        log_probs = self.head(x).log_softmax(dim=-1)
        return {"log_probs": log_probs, "log_probs_length": lengths}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
