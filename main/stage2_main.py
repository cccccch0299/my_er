import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
from module.rapid_self import RAPID
from module.data_load import Rapid_Dataset
import psutil
import time
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_num', type=int, default=1855, help='the number of users')
    parser.add_argument('--ex_num', type=int, default=948, help='the number of exercises')
    parser.add_argument('--kc_num', type=int, default=57, help='the number of concepts')

    parser.add_argument('--user_hidden_size',type=int, default=64, help='the size of user embedding')
    parser.add_argument('--ex_hidden_size', type=int, default=64, help='the size of exercise embedding')
    parser.add_argument('--LSTM_hidden_size', type=int, default=64, help='the size of LSTM hidden state')
    parser.add_argument('--dropout', type=float, default=0.01, help='the dropout rate')
    parser.add_argument('--batch_size', type=int, default=12, help='the batch size')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--reg_lambda', type=float, default=1e-5, help='the regularization parameter')
    parser.add_argument('--init_rank_len', type=int, default=150, help='the length of initial ranking list')
    parser.add_argument('--eval_step', type=int, default=1, help='the step of evaluation')
    parser.add_argument('--output_type', type=str, default='det', choices=['det', 'pro'])
    parser.add_argument('--Q_matrix_file', type=str, default='../dataset/nips34/Q.npy')
    parser.add_argument('--test_data_path', type=str, default='../dataset/nips34/test_sequences.csv')
    parser.add_argument('--EB_save_path', type=str, default='../dataset/nips34/EB_mlstm3_del_0.7.txt')
    parser.add_argument('--Reranking_result_save_path', type=str, default='../dataset/nips34/Reranking_result_melt.txt')
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()
    return args

def main(args):

    dataset = Rapid_Dataset(args)
    test_data = pd.read_csv(args.test_data_path)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = RAPID(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg_lambda)
    Q_matrix = torch.tensor(np.load(args.Q_matrix_file)).to(device)
    Q_matrix = Q_matrix.cpu().numpy()

    for epoch in range(args.epochs):
        
        model.train()
        model.istrain = True
        train_loss = 0
        for batch_data in tqdm(train_dataloader):
            batch_data = [data.to(device) for data in batch_data]
            loss, _ = model.forward(batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % args.eval_step == 0:
            model.eval()
            model.istrain = False
            u_rerank_list = []
            for batch_data in tqdm(test_dataloader):
                batch_data = [data.to(device) for data in batch_data]
                _, u_rerank = model.forward(batch_data)
                u_rerank_list.append(u_rerank.cpu().numpy())

            u_rerank_list = np.concatenate(u_rerank_list, axis=0)
            metrics = model.evaluate(u_rerank_list, test_data, Q_matrix)

            uid = test_data['uid'].tolist()

            with open(args.Reranking_result_save_path, 'w') as f:
                for i in range(len(u_rerank_list)):
                    rel = u_rerank_list[i]
                    filter_q = ','.join([str(q) for q in rel])
                    f.write(f"{uid[i]}\t{filter_q}\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)





