import numpy as np
import torch
import torch.nn as nn
from module.lstm import LSTM
from module.user_branch import Melt_User_Branch


class MELT_LSTM(nn.Module):
    def __init__(self, kc_num, emb_size, hidden_size, u_L_min, u_L_max, user_threshold, device, epochs, is_mlstm, n_clusters=5):
        super(MELT_LSTM, self).__init__()
        self.kc_num = kc_num
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.u_L_max = u_L_max
        self.u_L_min = u_L_min
        self.user_threshold = user_threshold
        self.device = device
        self.epochs = epochs
        self.is_mlstm =is_mlstm
        self.lamb_u = 1

        self.LSTM = LSTM(kc_num, emb_size, hidden_size, is_mlstm, u_L_max)

        # 创新点一：支持配置学习风格聚类数量
        self.user_branch = Melt_User_Branch(emb_size, self.u_L_min, self.u_L_max, device, self.epochs, n_clusters)


    def forward(self, batch_data, h_u_batch, epoch):

        user, concepts, responses, mask, _ = batch_data
        user_emb = self.LSTM.sequence_encoding(concepts, mask)

        h_user, h_u_concepts, h_u_responses, h_u_mask, h_u_batch_len = h_u_batch
        h_u_emb = self.LSTM.sequence_encoding(h_u_concepts, h_u_mask)

        h_u_loss = self.user_branch(self.LSTM, h_user, h_u_emb, h_u_concepts, h_u_mask, self.user_threshold, epoch)

        lstm_pred = self.LSTM(concepts[:, :-1], responses[:, :-1], user_emb)
        lstm_loss = self.LSTM.caculate_loss(lstm_pred, responses[:, 1:], concepts[:, 1:], self.kc_num, mask[:, 1:],)

        loss = h_u_loss * self.lamb_u + lstm_loss


        return lstm_loss

    def predict(self, batch_data, beta=1):
        user, concepts, responses, mask, batch_user_len = batch_data
        user_emb = self.LSTM.sequence_encoding(concepts, mask)

        t_u_idx = torch.nonzero(batch_user_len < self.user_threshold).squeeze()
        
        if t_u_idx.numel() != 0:
            if t_u_idx.dim() == 0:
                t_u_idx = t_u_idx.view(-1)
            t_u_emb = user_emb[t_u_idx]
            enhance_t_u_emb = self.user_branch.W_U(t_u_emb) + beta * t_u_emb
            user_emb.scatter_(0, t_u_idx.unsqueeze(1).expand_as(enhance_t_u_emb), enhance_t_u_emb)

        lstm_pred = self.LSTM(concepts[:, :-1], responses[:, :-1], user_emb)


        return lstm_pred

