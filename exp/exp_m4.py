from data.data_loader import Dataset_MTS, Dataset_M4
from data.m4 import M4Meta
from utils.m4_summary import M4Summary
from math import exp
from exp.exp_basic import Exp_Basic
from models.patchtst import PatchTSTC
from models.tsmixer import TSMixerC
from models.Dlinear import DLinearC
from models.timesnet import TimesNetC
import torch.nn.functional as F
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, MSE_dim
from utils.losses import mape_loss, mase_loss, smape_loss
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
import pandas
from torchinfo import summary

class Exp_M4(Exp_Basic):
    def __init__(self, args):
        super(Exp_M4, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTSTC,
            'TSMixer': TSMixerC,
            'DLinear': DLinearC,
            'TimesNet': TimesNetC
        }
        
        self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
        self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
        self.args.label_len = self.args.pred_len
        self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        
        
        self.args.in_len = self.args.seq_len
        self.args.out_len = self.args.pred_len
        model = model_dict[self.args.model](self.args).float()
        
        # summary(model)
        # model = TVModel(self.args).float()
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
            data_set = Dataset_M4(
                root_path=args.root_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                freq=args.freq,
                seasonal_patterns=args.seasonal_patterns
            )

            print(flag, len(data_set))
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size
            data_set = Dataset_M4(
                root_path=args.root_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                freq=args.freq,
                seasonal_patterns=args.seasonal_patterns
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



    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')

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
        criterion = self._select_criterion(self.args.loss)
        criterion_ts =  nn.MSELoss()
        
        inverse = False
        f_dim = -1 if self.args.features == 'MS' else 0
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            tl_f = []
            tl_s = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)  #[bsz, in_len, 1]
                batch_y = batch_y.float().to(self.device)

                pred = self.model(batch_x.permute(2,1,0), if_update=True)  #[1, out_len, bsz]
                pred = pred.permute(2,1,0)  #[bsz, out_len, 1]
                pred = pred[:, -self.args.pred_len:, f_dim:]
                
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                # if inverse:
                #     outputs = dataset_object.inverse_transform(outputs)
                #     batch_y = dataset_object.inverse_transform(batch_y)
                loss_f = criterion(batch_x, self.args.frequency_map, pred, batch_y, batch_y_mark)
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
            # vali_mse, vali_loss, vali_loss_f, vali_loss_s, vali_mae = self.vali(vali_data, vali_loader)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("prob", self.model.cluster_prob)

            print("Epoch: {0}, Steps: {1}, Cost time: {2:.3f} | Train Loss: {3:.7f}  Test Loss: {4:.7f} ".format(
                epoch + 1, train_steps, time.time()-epoch_time, train_loss, test_loss))
            
            
            wandb.log({"Train_loss":train_loss, "Train_forecast_loss":train_loss_f ,"Train_similarity_loss": train_loss_s,
                "Vali_loss": vali_loss})
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
        
        


    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape 
            id_list = np.arange(0, B, self.args.batch_size)
            outputs = torch.zeros((int(id_list[-1]), self.args.pred_len, C)).float().to(self.device)
            # id_list = np.append(id_list, B)
            f_dim = -1 if self.args.features == 'MS' else 0
            
            for i in range(len(id_list) - 1):
                input_x = x[id_list[i]:id_list[i + 1],:,:].permute(2,1,0)  #[1, in_len, bsz]
                output = self.model(input_x, if_update=False).permute(2,1,0)
                output = output[:, -self.args.pred_len:, f_dim:]  #[bsz, out_len, 1]
                outputs[id_list[i]:id_list[i + 1], :, :] = output
                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs.detach().cpu()
            true = torch.from_numpy(np.array(y)).detach().cpu()
            batch_y_mark = torch.ones(true.shape)
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)
        self.model.train()
        return loss
    
    
    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape 
            id_list = np.arange(0, B, self.args.batch_size)
            outputs = torch.zeros((int(id_list[-1]), self.args.pred_len, C)).float().to(self.device)
            # id_list = np.append(id_list, B)
            f_dim = -1 if self.args.features == 'MS' else 0
            
            for i in range(len(id_list) - 1):
                input_x = x[id_list[i]:id_list[i + 1]].permute(2,1,0)  #[1, in_len, B]
                output = self.model(input_x, if_update=False).permute(2,1,0)
                output = output[:, -self.args.pred_len:, f_dim:]  #[bsz, out_len, 1]
                outputs[id_list[i]:id_list[i + 1], :, :] = output
                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            preds = outputs.detach().cpu().numpy()
            trues = y
            
            # x = x.detach().cpu().numpy()

            # for i in range(0, preds.shape[0], preds.shape[0] // 10):
            #     gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
            #     pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return

 