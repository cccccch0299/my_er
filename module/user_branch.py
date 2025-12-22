import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from module.lstm import LSTM


class Melt_User_Branch(nn.Module):
    def __init__(self, hidden_units, u_L_min, u_L_max, device, e_max):
        """
        User branch: Enhance the tail user representation
        """
        super(Melt_User_Branch, self).__init__()
        self.W_U = torch.nn.Linear(hidden_units, hidden_units)
        torch.nn.init.xavier_normal_(self.W_U.weight.data)
        self.criterion = torch.nn.MSELoss()

        self.u_L_max = u_L_max
        self.u_L_min = u_L_min
        self.e_max = e_max
        self.pi = np.pi

        self.device = device

    def forward(self, seq_encoder, h_user, h_u_emb, h_u_concepts, h_u_mask, user_thres, epoch):
        full_seq_repre = h_u_emb
        h_u_num = h_user.numel()
        w_u_list = []
        for i in range(h_u_num):
            # Calculate the loss coefficient
            u_seq_length = len(torch.masked_select(h_u_concepts[i], h_u_mask[i].to(torch.bool)))
            w_u = (self.pi / 2) * (epoch / self.e_max) + \
                  (self.pi / (2 * (self.u_L_max - user_thres - 1))) * (u_seq_length - user_thres - 1)
            w_u = np.abs(np.sin(w_u))
            w_u_list.append(w_u)

        few_seq = torch.zeros(h_u_num, self.u_L_max, dtype=h_u_concepts.dtype).to(self.device)
        few_seq_mask = []
        fixed_T = 100
        h_u_sub_seq = np.full(h_u_num, fixed_T)
        for i, l in enumerate(h_u_sub_seq):
            few_seq[i, :l] = h_u_concepts[i, :l]
            few_seq_mask.append(torch.tensor([1] * l + [0] * (self.u_L_max - l)))

        few_seq_mask = torch.stack(few_seq_mask)
        few_seq_repre = seq_encoder.sequence_encoding(few_seq, few_seq_mask)
        w_u_list = torch.FloatTensor(w_u_list).view(-1, 1).to(self.device)

        # Curriculum Learning by user
        loss = torch.mean(w_u_list * ((self.W_U(few_seq_repre) - full_seq_repre) ** 2))
        return loss
