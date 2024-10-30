import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import *
from models.patch_layer import *

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
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
        return res, moving_mean

class DLinearC(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args):
        super(DLinearC, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len
        self.n_cluster = args.n_cluster
        self.d_ff = args.d_ff
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = args.individual
        self.channels = args.batch_size if args.data == "M4" else args.data_dim

        if self.individual == "i":
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
        elif self.individual == "c":
            self.Linear_Seasonal = Cluster_wise_linear(self.n_cluster, self.channels, self.seq_len, self.pred_len, self.device)
            self.Linear_Trend = Cluster_wise_linear(self.n_cluster, self.channels,self.seq_len, self.pred_len, self.device)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.channels, self.n_cluster, self.seq_len, self.d_ff, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
            

    def forward(self, x, if_update=False):
        # x: [Batch, Input length, Channel]
        if self.individual == "c":
            self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)
        else:
            self.cluster_prob = None
        if if_update and self.individual == "c":
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual == "i":
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        elif self.individual == "c":
            seasonal_output = self.Linear_Seasonal(seasonal_init, self.cluster_prob)
            trend_output = self.Linear_Trend(trend_init, self.cluster_prob)
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]