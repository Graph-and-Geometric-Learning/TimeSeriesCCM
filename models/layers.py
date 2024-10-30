import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from models.attention import *


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: BLC
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  #output: BLC
    
    
    
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x




class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model):
        super(_Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster = nn.Linear(d_model*2, 1)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, cluster_emb.shape[-1])      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = int(bn/self.n_vars)
        x_emb_batch = x_emb.repeat(self.n_cluster, 1)   
        cluster_emb_batch = torch.repeat_interleave(cluster_emb, bn, dim=0)
        out = torch.cat([x_emb_batch, cluster_emb_batch], dim=-1)
        prob = F.sigmoid(self.cluster(out)).squeeze(-1).reshape(self.n_cluster, bs, self.n_vars).permute(1,2,0)
        # prob: [bs, n_vars, n_cluster]
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = F.softmax(prob_avg, dim=-1)
        return prob_avg


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        # self.fc3 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        # linear_layer = [nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        # self.linear = MLP(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_temp)   #[bs*n_vars, n_cluster]
        num_var_pc = torch.sum(mask, dim=0)
        adpat_cluster = torch.matmul(x_emb.transpose(0,1), mask)/(num_var_pc + 1e-6)  #[d_model, n_cluster]
        cluster_emb = cluster_emb + adpat_cluster.transpose(0,1)
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        return prob_avg, cluster_emb
    
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern

class Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        # linear_layer = [nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        # self.linear = MLP(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        prob_avg = torch.mean(prob, dim=0)    #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)   #[bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars,-1)
        cluster_emb_ = cluster_emb.repeat(bs,1,1)
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0,1))

        return prob_avg, cluster_emb
    
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern


def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):   #[n_vars, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q


# def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
#     Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
#     B = Q.shape[1] # number of samples to assign
#     K = Q.shape[0] # how many prototypes
#     # Q = Q / torch.sum(Q, dim=0, keepdim=True)
#     # make the matrix sums to 1
#     sum_Q = torch.sum(Q)
#     Q = Q / sum_Q
    
#     for it in range(sinkhorn_iterations):
#         # normalize each row: total weight per prototype must be 1/K
#         Q = Q / torch.sum(Q, dim=1, keepdim=True)
#         Q = Q / K
        
#         # normalize each column: total weight per sample must be 1/B
#         Q = Q / torch.sum(Q, dim=0, keepdim=True)
#         Q = Q / B

#     Q = Q * B # the colomns must sum to 1 so that Q is an assignment
#     return Q.t()


def cluster_aggregator(var_emb, mask):
    '''
        var_emb: (bs*patch_num, nvars, d_model)
        mask: (nvars, n_cluster)
        return: (bs*patch_num, n_cluster, d_model)
    '''
    num_var_pc = torch.sum(mask, dim=0)
    var_emb = var_emb.transpose(1,2)
    cluster_emb = torch.matmul(var_emb, mask)/(num_var_pc + 1e-6)
    cluster_emb = cluster_emb.transpose(1,2)
    return cluster_emb


    
class CrossAttention(nn.Module):
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
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return out # B, L, d_model