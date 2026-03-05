import os
import logging
from datetime import datetime
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def setup_logger(dataset_name):
    log_dir = os.path.join('log', dataset_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    # 1. 添加 --dataset 选项
    parser.add_argument('--dataset', type=str, default='assist2009', choices=['assist2009', 'assist2012', 'nips34', 'assist2017','slepemapy','XES3G5M'], help='choose dataset')
    
    # 通用超参数
    parser.add_argument('--user_hidden_size', type=int, default=64, help='the size of user embedding')
    parser.add_argument('--ex_hidden_size', type=int, default=64, help='the size of exercise embedding')
    parser.add_argument('--LSTM_hidden_size', type=int, default=64, help='the size of LSTM hidden state')
    parser.add_argument('--dropout', type=float, default=0.01, help='the dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--epochs', type=int, default=51, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--reg_lambda', type=float, default=1e-5, help='the regularization parameter')
    parser.add_argument('--init_rank_len', type=int, default=150, help='the length of initial ranking list')
    parser.add_argument('--eval_step', type=int, default=10, help='the step of evaluation')
    parser.add_argument('--output_type', type=str, default='det', choices=['det', 'pro'])
    parser.add_argument('--device', type=str, default='cuda:0')
    # SC-MOO: 熵正则化系数
    parser.add_argument('--lambda_ent', type=float, default=0.01, help='entropy regularization coefficient for Pareto weights')

    # 先解析一次获取 dataset 名称
    temp_args = parser.parse_known_args()[0]
    dataset = temp_args.dataset

    # 2. 定义数据集相关的配置字典 (根据论文数据)
    dataset_configs = {
        'nips34': {
            'user_num': 1855, 
            'ex_num': 948, 
            'kc_num': 57,
            'base_path': './dataset/nips34'
        },
        'assist2009': {
            'user_num': 3884, 
            'ex_num': 17737, 
            'kc_num': 123,
            'base_path': './dataset/assist2009'
        },
        'assist2012': {
            'user_num': 27485, 
            'ex_num': 53070, 
            'kc_num': 265,
            'base_path': './dataset/assist2012'
        },

        'assist2017': {
            'user_num': 1709, 
            'ex_num': 3162, 
            'kc_num': 102,
            'base_path': './dataset/assist2017'
        },

        'slepemapy': {
            'user_num': 85600, 
            'ex_num': 2913, 
            'kc_num': 1458,
            'base_path': './dataset/slepemapy'
        },

        'XES3G5M': {
            'user_num': 18066, 
            'ex_num': 7652, 
            'kc_num': 865,
            'base_path': './dataset/XES3G5M'
        }

    }

    config = dataset_configs[dataset]

    # 3. 动态设置默认值
    parser.add_argument('--user_num', type=int, default=config['user_num'])
    parser.add_argument('--ex_num', type=int, default=config['ex_num'])
    parser.add_argument('--kc_num', type=int, default=config['kc_num'])
    parser.add_argument('--Q_matrix_file', type=str, default=f"{config['base_path']}/Q.npy")
    parser.add_argument('--test_data_path', type=str, default=f"{config['base_path']}/test_sequences.csv")
    parser.add_argument('--EB_save_path', type=str, default=f"{config['base_path']}/EB.txt")
    parser.add_argument('--Reranking_result_save_path', type=str, default=f"{config['base_path']}/Reranking_result_melt.txt")

    args = parser.parse_args()
    return args

def main(args):
    # 初始化日志，自动使用 dataset 名称
    logger = setup_logger(args.dataset)
    logger.info(f"Starting Stage 2 Training for dataset: {args.dataset}")
    logger.info(f"Arguments: {vars(args)}")

    dataset = Rapid_Dataset(args)
    test_data = pd.read_csv(args.test_data_path)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = RAPID(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg_lambda)
    
    Q_matrix = torch.tensor(np.load(args.Q_matrix_file)).to(device)
    Q_matrix_cpu = Q_matrix.cpu().numpy()

    for epoch in range(args.epochs):
        model.train()
        model.istrain = True
        train_loss = 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        for batch_data in pbar:
            batch_data = [data.to(device) for data in batch_data]
            loss, _ = model.forward(batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        if epoch % args.eval_step == 0:
            model.eval()
            model.istrain = False
            u_rerank_list = []
            for batch_data in tqdm(test_dataloader, desc="Evaluating"):
                batch_data = [data.to(device) for data in batch_data]
                _, u_rerank = model.forward(batch_data)
                u_rerank_list.append(u_rerank.cpu().numpy())

            u_rerank_list = np.concatenate(u_rerank_list, axis=0)
            metrics = model.evaluate(u_rerank_list, test_data, Q_matrix_cpu)
            
            # 记录指标到日志
            logger.info(f"[Epoch {epoch+1} Metrics]")
            for k in [1, 3, 5, 10]:
                msg = (f"K={k:2d} | NDCG: {metrics['ndcg'][f'@{k}']:.4f} | "
                       f"Hit: {metrics['hit'][f'@{k}']:.4f} | "
                       f"Recall: {metrics['recall'][f'@{k}']:.4f} | "
                       f"F1: {metrics['f1'][f'@{k}']:.4f} | " 
                       f"Div: {metrics['div'][f'@{k}']:.4f}")
                logger.info(msg)
            logger.info("-" * 60)

            # 保存推荐结果
            uid = test_data['uid'].tolist()
            with open(args.Reranking_result_save_path, 'w') as f:
                for i in range(len(u_rerank_list)):
                    rel = u_rerank_list[i]
                    filter_q = ','.join([str(q) for q in rel])
                    f.write(f"{uid[i]}\t{filter_q}\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)