import argparse
import os
import torch
import wandb
from exp.exp_stock import Exp_Stock
from utils.tools import string_split
import math
from torchinfo import summary

parser = argparse.ArgumentParser(description='CCM')

# model: PatchTST / DLinear / TSMixer / TimesNet

parser.add_argument('--data', type=str, default='stock', help='data')
parser.add_argument('--freq', type=str, default='h')  #[s, t, h, d, b, w, m]
parser.add_argument('--features', type=str, default='S')  
parser.add_argument('--data_path', type=str, default='stock.csv')  
parser.add_argument('--model', type=str, default='PatchTST', help='model')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer initial learning rate')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--individual', type=str, default="c", help="i: individual; c: cluster, else: all-in dimension")
parser.add_argument('--beta', type=float, default=0.1, help="loss weight for similarity loss")
parser.add_argument('--in_len', type=int, default=21, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=7, help='output MTS length (\tau)')


parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--n_layers', type=int, default=4, help='num of encoder layers (N)')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of MLP in transformer')
parser.add_argument('--cluster_ratio', type=float, default=0.3, help="ratio of clusters")


## model parameters
parser.add_argument('--d_model', type=int, default=512, help='dimension of hidden states (d_model)')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--attn_dropout', type=float, default=0.3, help='attention dropout')
parser.add_argument('--pre_norm', type=bool, default=False, help='pre normalization')
parser.add_argument('--stride', type=int, default=8, help="stride")
parser.add_argument('--pretrain_head', type=bool, default=False, help='pretrain head')
parser.add_argument('--patch_len', type=int, default=16, help='patch length (L_seg)')
parser.add_argument('--max_seq_len', type=int, default=1024, help="maximum number of sequence_length")
parser.add_argument('--padding_patch', type=str, default='end', help='None: None; end: padding on the end')



parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=True)
parser.add_argument('--cuda', type=int, default=3, help='GPU device.')

args = parser.parse_args()

args.data_dim = 1

if args.individual == "i":
    args.beta = 0.0

args.n_cluster = 4


print('Args in experiment:')
print(args)


wandb.login(key="")
wandb.init(project="time_series_forecasting", mode="disabled")  #  mode="disabled"
wandb.config.update(args)

Exp = Exp_Stock
setting = f'{args.model}_Stocks'
exp = Exp(args) 
print('>>>>>>>start training on: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
exp.train(setting)

    
    
