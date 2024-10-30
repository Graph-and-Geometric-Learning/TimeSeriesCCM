from data.data_loader import Dataset_MTS, Dataset_stock
from exp.exp_basic import Exp_Basic
from models.patchtst import PatchTSTC
from models.tsmixer import TSMixerC
from models.Dlinear import DLinearC
from models.timesnet import TimesNetC
import torch.nn.functional as F
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, MSE_dim

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import wandb
import os
import time
import json
import pickle
from torchinfo import summary
import warnings
warnings.filterwarnings('ignore')

class Exp_Stock(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTSTC,
            'TSMixer': TSMixerC,
            'DLinear': DLinearC,
            'TimesNet': TimesNetC
        }
        self.args.pred_len = self.args.out_len
        self.args.seq_len = self.args.in_len  # input_len = 2*pred_len
        self.args.label_len = self.args.pred_len
        # self.args.in_len = self.args.seq_len
        
        model = model_dict[self.args.model](self.args).float()
        summary(model)
        # model = TVModel(self.args).float()
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
            data_set = Dataset_stock(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                out_len = args.out_len,
                in_len = args.in_len
            )
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
            data_set = Dataset_stock(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                out_len = args.out_len,
                in_len = args.in_len
            )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=True)
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        loss_f_list = []
        loss_s_list = []
            
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        f_dim = -1 if self.args.features == 'MS' else 0
        criterion_ts =  nn.MSELoss()
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device)  #[bsz, in_len, 1]
                batch_y = batch_y.float().to(self.device)
                pred = self.model(batch_x.permute(2,1,0), if_update=False)  #[1, out_len, bsz]
                pred = pred.permute(2,1,0)  #[bsz, out_len, 1]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss_f = criterion_ts(pred, batch_y) 
                simMatrix = self._get_similarity_matrix(batch_x)
                loss_s = self._similarity_loss_batch(self.model.cluster_prob, simMatrix)

                loss = loss_f + self.args.beta * loss_s
                total_loss.append(loss.detach().item())
                loss_f_list.append(loss_f.detach().item())
                loss_s_list.append(loss_s.detach().item())
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), batch_y.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
        total_loss = np.average(total_loss)
        loss_f = np.average(loss_f_list)
        loss_s = np.average(loss_s_list)
        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num
        mae, mse, rmse, mape, mspe = metrics_mean
        self.model.train()
        return mse, total_loss, loss_f, loss_s, mae
    

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        val_data, val_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        # with open(os.path.join(path, "args.json"), 'w') as f:
        #     json.dump(vars(self.args), f, indent=True)
        # scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        # with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
        #     pickle.dump(scale_statistic, f)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion_ts =  nn.MSELoss()
        
        f_dim = -1 if self.args.features == 'MS' else 0
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            tl_f = []
            tl_s = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                # pred, true = self._process_one_batch(
                #     train_data, batch_x, batch_y, if_update=True)
                
                
                batch_x = batch_x.float().to(self.device)  #[bsz, in_len, 1]
                batch_y = batch_y.float().to(self.device)

                pred = self.model(batch_x.permute(2,1,0), if_update=True)  #[1, out_len, bsz]
                pred = pred.permute(2,1,0)  #[bsz, out_len, 1]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss_f = criterion_ts(pred, batch_y) 
                simMatrix = self._get_similarity_matrix(batch_x)
                loss_s = self._similarity_loss_batch(self.model.cluster_prob, simMatrix)
                
                loss = loss_f + self.args.beta * loss_s
                train_loss.append(loss.item())
                tl_f.append(loss_f.item())
                tl_s.append(loss_s.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                
                # if epoch % 10 == 0:
            # self.vis_linear_weight(epoch)
            
            train_loss = np.average(train_loss)
            train_loss_f = np.average(tl_f)
            train_loss_s = np.average(tl_s)
            vali_mse, vali_loss, vali_loss_f, vali_loss_s, vali_mae = self.vali(val_data, val_loader)
            test_mse, test_loss, test_loss_f, test_loss_s, test_mae = self.vali(test_data, test_loader)
            print("prob", self.model.cluster_prob)

            print("Epoch: {0}, Steps: {1}, Cost time: {2:.3f} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f} Test MSE: {6:.3f} Test MAE: {7:.3f}".format(
                epoch + 1, train_steps, time.time()-epoch_time, train_loss, vali_loss, test_loss, test_mse, test_mae))
            
            
            wandb.log({"Train_loss":train_loss, "Train_forecast_loss":train_loss_f ,"Train_similarity_loss": train_loss_s,
                "Vali_loss": vali_loss, "Vali_forecast_loss":vali_loss_f , "Vali_similarity_loss": vali_loss_s, "Vali_mse": vali_mse,
                    "Test_loss": test_loss ,"Test_forecast_loss": test_loss_f,"Test_similarity_loss":test_loss_s, "Test_mse": test_mse,
                    "Test_mae": test_mae, "Vali_mae": vali_mae,})
            wandb.log({"Cluster_prob": wandb.Histogram(self.model.cluster_prob.detach().cpu().numpy())})

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model
    
    
    def _similarity_loss_batch(self, prob, simMatrix):
        def concrete_bern(prob, temp = 0.07):
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
            return prob_bern
        membership = concrete_bern(prob)  #[n_vars, n_clusters]
        # membership = prob
        temp_1 = torch.mm(membership.t(), simMatrix) 
        SAS = torch.mm(temp_1, membership)
        _SS = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss
    
    def _get_similarity_matrix(self, batch_x):
        sample = batch_x.squeeze(-1)  #[bsz, in_len]
        diff = sample.unsqueeze(1) - sample.unsqueeze(0)
        dist_squared = torch.sum(diff ** 2, dim=-1)  #[bsz, bsz]
        param = torch.max(dist_squared)
        euc_similarity = torch.exp(-5 * dist_squared /param )
        return euc_similarity
        
        
    def test(self, setting, save_pred = True, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse, if_update=False)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        return

        
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(self.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern
    
    
