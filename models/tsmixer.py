import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.patch_layer import *


class RevNorm(nn.Module):
    """Reversible Instance Normalization in PyTorch."""

    def __init__(self, num_features, axis=-2, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features)) 

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = x.mean(dim=self.axis, keepdim=True).detach() # [Batch, Input Length, Channel]
        self.stdev = torch.sqrt(x.var(dim=self.axis, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x

class TSLinear(nn.Module):
    def __init__(self, L, T):
        super(TSLinear, self).__init__()
        self.fc = nn.Linear(L, T)

    def forward(self, x):
        return self.fc(x)

class TSMixerC(nn.Module):
    def __init__(self, args):
        super(TSMixerC, self).__init__()
        self.n_vars = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.in_len = args.in_len
        self.out_len = args.out_len
        self.n_cluster = args.n_cluster
        self.d_ff = args.d_ff
        self.d_model = args.d_model
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.individual = args.individual
        self.mixer_layers = []
        self.n_mixer = args.n_layers
        for i in range(self.n_mixer):
            self.mixer_layers.append(MixerLayer(self.n_cluster, self.n_vars, self.in_len, self.device, self.individual, self.d_ff, args.dropout)) 
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.temp_proj = TemporalProj(self.n_cluster, self.n_vars, self.in_len, self.out_len, self.device, self.individual)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.n_vars, self.n_cluster, self.in_len, self.d_ff, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        self.rev_in = RevIN(num_features=self.n_vars)
        
    def forward(self, x, if_update=False):
        if self.individual == "c":
            self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)
        else:
            self.cluster_prob = None
        # x = self.rev_in(x, mode = "norm")

        for i in range(self.n_mixer):
            x = self.mixer_layers[i](x, self.cluster_prob)
        x = self.temp_proj(x, self.cluster_prob)
        # x = self.rev_in(x, mode="denorm")
        if if_update and self.individual == "c":
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)
        return x


class MLPTime(nn.Module):
    def __init__(self, n_cluster, n_vars, seq_len, device, individual, dropout_rate):
        super(MLPTime, self).__init__()
        if individual == "c":
            self.fc = Cluster_wise_linear(n_cluster, n_vars, seq_len, seq_len, device)
        else:
            self.fc = nn.Linear(seq_len, seq_len)
        self.individual = individual
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, prob):
        if self.individual == "c":
            x = self.fc(x, prob)
        else:
            x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MLPFeat(nn.Module):
    def __init__(self, C, ff_dim, dropout_rate=0.1):
        super(MLPFeat, self).__init__()
        self.fc1 = nn.Linear(C, ff_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, C)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, n_cluster, n_vars, seq_len, device, individual, ff_dim, dropout):
        super(MixerLayer, self).__init__()
        self.mlp_time = MLPTime(n_cluster, n_vars, seq_len, device, individual, dropout)
        self.mlp_feat = MLPFeat(n_vars, ff_dim, dropout)
        
    def batch_norm_2d(self, x):
        """ x has shape (B, L, C) """
        return (x - x.mean()) / x.std()
    
    def forward(self, x, prob):
        """ x has shape (B, L, C) """
        res_x = x
        x = self.batch_norm_2d(x)
        x = x.transpose(1, 2)
        x = self.mlp_time(x, prob)
        x = x.transpose(1, 2) + res_x
        res_x = x
        x = self.batch_norm_2d(x)
        x = self.mlp_feat(x) + res_x
        return x

class TemporalProj(nn.Module):
    def __init__(self, n_cluster, n_vars, in_dim, out_dim, device, individual):
        super(TemporalProj, self).__init__()
        if individual == "c":
            self.fc = Cluster_wise_linear(n_cluster, n_vars, in_dim, out_dim, device)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
        self.individual = individual
    def forward(self, x, prob):
        # x: [bs, seq_len, n_vars]
        # mask: [n_vars, n_cluster]
        x = x.transpose(1, 2)
        if self.individual == "c":
            x = self.fc(x, prob)
        else:
            x = self.fc(x)
        x = x.transpose(1, 2)
        return x  #[n_cluster,seq_len]
    
    

        