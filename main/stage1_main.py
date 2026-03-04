import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from module.data_load import KTDataset, Head_User_Dataset
from module.melt_lstm import MELT_LSTM
import module.EB_filter as EB_filter
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import argparse
#from module import kt_base_eval as kt_base_eval

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    # 1. 添加 --dataset 选项
    parser.add_argument('--dataset', type=str, default='assist2009', choices=['assist2009', 'assist2012', 'nips34', 'assist2017','slepemapy','XES3G5M'], help='choose dataset')
    
    # 其他通用超参数保持不变
    parser.add_argument('--emb_size', type=int, default=200, help='the size of embedding')
    parser.add_argument('--hidden_size', type=int, default=200, help='the size of hidden layer')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--eval_step', type=int, default=10, help='the step of evaluation')
    parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--train_batch_size', type=int, default=64, help='the batch size of train data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='the batch size of test data')
    parser.add_argument('--is_mlstm', type=bool, default=True, help='use lstm(False) or mlstm(True)')
    parser.add_argument('--is_pkc', type=bool, default=False, help='use pkc(True) or not(False)')
    # 创新点一：学习风格聚类数量
    parser.add_argument('--n_clusters', type=int, default=5, help='number of learning style clusters')

    # 先解析一次获取 dataset 名称
    temp_args = parser.parse_known_args()[0]
    dataset = temp_args.dataset

    # 2. 定义数据集相关的配置字典 
    # 根据论文 Table 2 的数据进行配置
    dataset_configs = {
        'nips34': {
            'kc_num': 57,
            'base_path': './dataset/nips34'
        },
        'assist2009': {
            'kc_num': 123,
            'base_path': './dataset/assist2009'
        },
        'assist2012': {
            'kc_num': 265,
            'base_path': './dataset/assist2012'
        },
        'assist2017': {
            'kc_num': 102,
            'base_path': './dataset/assist2017'
        },
        'slepemapy': {
            'kc_num': 1458,
            'base_path': './dataset/slepemapy'
        },
        'XES3G5M': {
            'kc_num': 865,
            'base_path': './dataset/XES3G5M'
        }
    }

    config = dataset_configs[dataset]

    # 3. 动态设置数据集相关的默认值
    parser.add_argument('--kc_num', type=int, default=config['kc_num'], help='the number of concepts')
    parser.add_argument('--train_data_file', type=str, 
                        default=f"{config['base_path']}/train_valid_sequences.csv")
    parser.add_argument('--test_data_file', type=str, 
                        default=f"{config['base_path']}/test_sequences.csv")
    parser.add_argument('--Q_file', type=str, 
                        default=f"{config['base_path']}/Q.npy")
    parser.add_argument('--stu_ks_save_file', type=str, 
                        default=f"./stu_ks_save/{dataset}/")

    return parser.parse_args()


def calculate_metrics(predictions, targets, mask, concepts, kc_num, is_pkc):
    #predictions = (predictions * nn.functional.one_hot(concepts.to(torch.long), kc_num)).sum(-1)
    predictions = predictions.gather(-1, concepts.to(torch.long).unsqueeze(-1)).squeeze(-1)
    predictions = torch.masked_select(predictions, mask.to(torch.bool))

    targets = torch.masked_select(targets, mask.to(torch.bool))

    if is_pkc == False:
        auc = roc_auc_score(targets.detach().numpy(), predictions.detach().numpy())
    else:
        auc = 0

    predict_label = (predictions > 0.5).long().numpy()
    acc = accuracy_score(targets.numpy(), predict_label)
    return (auc, acc)


def plot_results(indicator_result):
    plt.figure(figsize=(10, 5))
    plt.plot(indicator_result['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(10, 5))

    if len(indicator_result['test_auc']) != 0:
        plt.plot(indicator_result['test_auc'], label='Test AUC')

    plt.plot(indicator_result['test_acc'], label='Test Accuracy')
    plt.title('Test Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def main(args):
    kc_num = args.kc_num
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    emb_size = args.emb_size
    hidden_size = args.hidden_size
    epochs = args.epochs
    eval_step = args.eval_step
    lr = args.lr
    weight_decay = args.weight_decay
    train_data_file = args.train_data_file
    test_data_file = args.test_data_file
    is_pkc = args.is_pkc
    is_mlstm = args.is_mlstm

    train_dataset = KTDataset(train_data_file)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    user_threshold = train_dataset.user_threshold
    user_max_seq_len = train_dataset.max_seq_len
    user_min_seq_len = train_dataset.min_seq_len

    test_dataset = KTDataset(test_data_file)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    h_u_len_ = train_dataset.h_u_len
    while h_u_len_ % (len(train_dataloader)) != 0:
        h_u_len_ += 1
    repeat_h_u_data_len = h_u_len_ - train_dataset.h_u_len
    if (repeat_h_u_data_len != 0):
        train_dataset.h_u_df = pd.concat([train_dataset.h_u_df, train_dataset.h_u_df.sample(repeat_h_u_data_len)],
                                         ignore_index=True)
    head_user_dataset = Head_User_Dataset(train_dataset.h_u_df, user_max_seq_len)
    h_u_batch_size = head_user_dataset.h_u_len // (len(train_dataloader))
    h_u_dataloader = DataLoader(head_user_dataset, batch_size=h_u_batch_size, shuffle=True)

    model = MELT_LSTM(kc_num, emb_size, hidden_size, user_min_seq_len, user_max_seq_len, user_threshold, device,
                      epochs, is_mlstm, args.n_clusters).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_auc = 0.0

    indicator_result = {'train_loss': [], 'test_auc': [], 'test_acc': []}

    test_data = pd.read_csv(args.test_data_file)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        for batch_idx, (batch, h_u_batch) in enumerate(
                tqdm(zip(train_dataloader, h_u_dataloader), desc=f'Train Epoch {epoch}', total=len(train_dataloader))):
            batch = [data.to(device) for data in batch]

            if is_pkc:
                responses = batch[2]
                responses_all_one = torch.ones_like(responses)
                batch[2] = responses_all_one

            h_u_batch = [data.to(device) for data in h_u_batch]
            optimizer.zero_grad()
            loss = model(batch, h_u_batch, epoch)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss / len(train_dataloader)))
        indicator_result['train_loss'].append(train_loss / len(train_dataloader))

        if epoch % eval_step == 9:

            model.eval()
            test_preds = []
            test_targets = []
            test_masks = []
            test_kcs = []
            stu_ks_list = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_dataloader, desc=f'Test Epoch {epoch}')):
                    batch = [data.to(device) for data in batch]

                    if is_pkc:
                        responses = batch[2]
                        responses_all_one = torch.ones_like(responses)
                        batch[2] = responses_all_one

                    user, concepts, responses, mask, len_concepts = batch
                    lstm_pred = model.predict(batch)

                    stu_ks = lstm_pred
                    for i in range(mask.size(0)):
                        user_mask = mask[i][1:]
                        last_true_idx = user_mask.nonzero(as_tuple=True)[0].max().item()
                        stu_ks_list.append(stu_ks[i, last_true_idx])

                    test_preds.append(lstm_pred.cpu())
                    test_targets.append(responses[:, 1:].cpu())
                    test_masks.append(mask[:, 1:].cpu())
                    test_kcs.append(concepts[:, 1:].cpu())


            test_preds = torch.cat(test_preds, dim=0)
            test_targets = torch.cat(test_targets, dim=0)
            test_masks = torch.cat(test_masks, dim=0)
            test_kcs = torch.cat(test_kcs, dim=0)
            stu_ks_tensor = torch.stack(stu_ks_list, dim=0)

            metrics = calculate_metrics(test_preds, test_targets, test_masks, test_kcs, kc_num, is_pkc)
            current_auc = metrics[0]
            print('Test Epoch: {}, AUC: {:.4f}, ACC: {:.4f}'.format(epoch, metrics[0], metrics[1]))
            # --- 修改：增加保存最优逻辑 ---
            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(stu_ks_tensor, f'{args.stu_ks_save_file}/pkm.pt') 
            indicator_result['test_auc'].append(current_auc)
            indicator_result['test_acc'].append(metrics[1])
            # stu_done_ks = {}
            # stu_ks_tensor = stu_ks_tensor.to('cpu').numpy()
            # for i in range(len(stu_ks_tensor)):
            #     pkm_i = stu_ks_tensor[i]
            #     kcs = [int(kc) for kc in set(test_data.iloc[i]['concepts'].split(',')) if kc != '-1']
            #     kc_last_pre = {}
            #     for kc in kcs:
            #         kc_last_pre[kc] = pkm_i[kc]
            #         stu_done_ks[i] = kc_last_pre
            #stu_true_response = kt_base_eval.preprocess_test_data(test_data)

    plot_results(indicator_result)


if __name__ == '__main__':
    args = parse_args()
    main(args)