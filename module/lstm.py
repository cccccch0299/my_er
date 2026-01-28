import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig


class LSTM(nn.Module):
    def __init__(self, kc_num, embedding_size, hidden_size, is_mlstm, u_L_max, dropout=0.2):
        super(LSTM, self).__init__()

        self.kc_num = kc_num

        self.kc_emb = nn.Embedding(kc_num, embedding_size)

        self.is_mlstm = is_mlstm

        if self.is_mlstm:
            cfg_encoder = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                    )
                ),
                context_length=u_L_max,
                num_blocks=3,
                embedding_dim=embedding_size,
                slstm_at=[],

            )
            self.lstm_encoder = xLSTMBlockStack(cfg_encoder)

            cfg_kt = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                    )
                ),
                context_length=u_L_max,
                num_blocks=3,
                embedding_dim=embedding_size * 2,
                slstm_at=[],

            )
            self.lstm_kt = xLSTMBlockStack(cfg_kt)
            self.mlstm_fc = nn.Linear(embedding_size * 2, hidden_size)

        else:
            self.lstm_encoder = nn.LSTM(embedding_size, embedding_size)
            self.lstm_kt = nn.LSTM(embedding_size * 2, hidden_size)

        self.interaction_emb = nn.Embedding(kc_num * 2, embedding_size)

        self.fc = nn.Linear(hidden_size, kc_num)
        self.user_mlp = torch.nn.Linear(embedding_size * 2, embedding_size * 2)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, kc_kt, responses, user_emb):
        kc_kt = kc_kt + self.kc_num * responses
        kc_kt_emb = self.interaction_emb(kc_kt)

        user_emb = user_emb.unsqueeze(1)
        user_emb = user_emb.repeat(1, kc_kt_emb.size(1), 1)

        kc_user_emb = torch.cat((kc_kt_emb, user_emb), dim=2)
        if self.is_mlstm:
            lstm_kt= self.lstm_kt(kc_user_emb)

            lstm_kt = self.mlstm_fc(lstm_kt)
        else:
            lstm_kt, _ = self.lstm_kt(kc_user_emb)

        lstm_kt = self.dropout(lstm_kt)
        out = self.fc(lstm_kt)
        out = self.sigmoid(out)
        return out

    def caculate_loss(self, y_pred, y_true, concepts, kc_num, mask):
        y_pred = (y_pred * nn.functional.one_hot(concepts.to(torch.long), kc_num)).sum(-1)
        y_pred = torch.masked_select(y_pred, mask.to(torch.bool))
        y_true = torch.masked_select(y_true, mask.to(torch.bool))
        return nn.functional.binary_cross_entropy(y_pred, y_true.to(torch.float32))

    def sequence_encoding(self, seq, mask):
        kc_emb = self.kc_emb(seq)
        if self.is_mlstm:
            lstm_encoder = self.lstm_encoder(kc_emb)
        else:
            lstm_encoder, _ = self.lstm_encoder(kc_emb)
        lstm_encoder = self.dropout(lstm_encoder)
        user_emb = []
        for i in range(mask.size(0)):
            user_mask = mask[i]
            last_true_idx = user_mask.nonzero(as_tuple=True)[0].max().item()
            user_emb.append(lstm_encoder[i, last_true_idx])
        user_emb = torch.stack(user_emb)
        return user_emb



