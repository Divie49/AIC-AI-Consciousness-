# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (B,T,H,dh)
        q = q.permute(0,2,1,3)  # (B,H,T,dh)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B,H,T,T)
        if mask is None:
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            mask = causal.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,T,dh)
        out = out.permute(0,2,1,3).contiguous().view(B,T,D)
        return self.o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = FeedForward(d_model, d_ff, resid_dropout)
        self.drop = nn.Dropout(resid_dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class GPTlikeModel(nn.Module):
    def __init__(self, vocab_size, d_model=2048, n_layers=9, n_heads=16, d_ff=None,
                 max_seq_len=2048, dropout=0.1, tie_word_embeddings=True, use_checkpoint=False):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, resid_dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.use_checkpoint = use_checkpoint
        self.tie_word_embeddings = tie_word_embeddings
        self.apply(self._init_weights)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError("Sequence length exceeds model max_seq_len")
        tok = self.token_emb(input_ids)
        pos = self.pos_emb[:, :T, :].expand(B, -1, -1)
        x = tok + pos
        # build causal mask
        causal = torch.tril(torch.ones(T, T, device=input_ids.device, dtype=torch.bool))
        mask = causal.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask & key_mask

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # checkpointing reduces memory by re-computing block forward on backward
                x = checkpoint(lambda y, m=mask, b=block: b(y, mask=m), x)
            else:
                x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
