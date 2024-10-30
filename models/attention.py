import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class MaskAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(MaskAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # scores = scores if mask == None else scores * mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        A = A if mask == None else A * mask
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class MaskAttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(MaskAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out) # B, L, d_model
    
    
    
# class MaskAttention(nn.Module):
#     '''
#     The Attention operation
#     '''
#     def __init__(self, scale=None, attention_dropout=0.1):
#         super(MaskAttention, self).__init__()
#         self.scale = scale
#         self.dropout = nn.Dropout(attention_dropout)
        
#     def forward(self, queries, keys, values, mask=None):
#         B,  H, L, E = queries.shape
#         _,  _, D, S = values.shape
#         scale = self.scale or 1./sqrt(E)

#         attn_scores = torch.matmul(queries, keys) * scale
#         attn_scores = attn_scores if mask == None else attn_scores * mask
#         attn_weights = F.softmax(attn_scores, dim=-1) 
#         attn_weights = self.dropout(attn_weights)
#         output = torch.matmul(attn_weights, values).transpose(2,1)  # [bs, L, n_head, d_v]
#         return output.contiguous()


# class MaskAttentionLayer(nn.Module):
#     '''
#     The Multi-head Self-Attention (MSA) Layer
#     input:
#         queries: (bs, L, d_model)
#         keys: (_, S, d_model)
#         values: (bs, S, d_model)
#         mask: (L, S)
#     return: (bs, L, d_model)

#     '''
#     def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
#         super(MaskAttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model//n_heads)
#         d_values = d_values or (d_model//n_heads)

#         self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Sequential(nn.Linear(d_values * n_heads, d_model), nn.Dropout(dropout))
#         self.n_heads = n_heads
#         self.mix = mix

#     def forward(self, queries, keys, values, mask=None):
#         # input dim: d_model
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1).transpose(1,2) 
#         keys = self.key_projection(keys).view(B, S, H, -1).permute(0,2,3,1)  
#         values = self.value_projection(values).view(B, S, H, -1).transpose(1,2)    

#         out = self.inner_attention(
#             queries,
#             keys,
#             values,
#             mask,
#         )
#         if self.mix:
#             out = out.transpose(2,1).contiguous()
#         out = out.view(B, L, -1)

#         return self.out_projection(out) # B, L, d_model