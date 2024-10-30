import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from models.emb_layers import DataEmbedding
from models.conv_layers import Inception_Block_V1
from models.patch_layer import Cluster_wise_linear
from models.layers import Cluster_assigner


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, args):
        super(TimesBlock, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.k = 5
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(args.d_model, args.d_ff,
                               num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(args.d_ff, args.d_model,
                               num_kernels=6)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res  #[B, in_len+out_len, C]


class TimesNetC(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, args):
        super(TimesNetC, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.n_cluster = args.n_cluster
        self.channels = args.batch_size if args.data in ["M4", "stock"] else args.data_dim
        self.d_ff = args.d_ff
        self.individual = args.individual
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        
        
        self.model = nn.ModuleList([TimesBlock(args)
                                    for _ in range(args.n_layers)])
        self.enc_embedding = DataEmbedding(args.data_dim, args.d_model, freq=args.freq, dropout=args.dropout)
        self.layer = args.n_layers
        self.layer_norm = nn.LayerNorm(args.d_model)
        if self.individual == "c":
            self.predict_linear = Cluster_wise_linear(self.n_cluster, self.channels,self.seq_len, self.pred_len+self.seq_len, self.device)
        else:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            
        self.projection = nn.Linear(
            args.d_model, args.data_dim, bias=True)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.channels, self.n_cluster, self.seq_len, self.d_ff, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        
    def forward(self, x, if_update=False):
        # Normalization from Non-stationary Transformer
        if self.individual == "c":
            self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)
        else:
            self.cluster_prob = None
            
        if if_update and self.individual == "c":
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)
        
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        if self.individual == "c":
            x = self.predict_linear(x.permute(0, 2, 1), self.cluster_prob).permute(0, 2, 1)  # align temporal dimension
        else:
            x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            
        # embedding
        enc_out = self.enc_embedding(x, x_mark=None)  # [B, in_len, d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  #[B, in_len+out_len, d_model]
        # porject back
        dec_out = self.projection(enc_out)      #[B, in_len+out_len, C]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out[:, -self.pred_len:, :]



