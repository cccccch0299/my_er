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
from module import kt_base_eval as kt_base_eval
import time
import psutil
import os

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kc_num', type=int, default=265, help='the number of concepts')
    parser.add_argument('--emb_size', type=int, default=200, help='the size of embedding')
    parser.add_argument('--hidden_size', type=int, default=200, help='the size of hidden layer')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--eval_step', type=int, default=10, help='the step of evaluation')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--train_batch_size', type=int, default=64, help='the batch size of train data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='the batch size of test data')
    parser.add_argument('--is_pkc', type=bool, default=False, help='get pkc(True) or get pkm(False)')
    parser.add_argument('--is_mlstm', type=bool, default=True, help='use lstm(False) or mlstm(True)')
    parser.add_argument('--train_data_file', type=str, default='../dataset/data_200/assist2012/train_valid_sequences.csv',
                        help='the path of train data')
    parser.add_argument('--test_data_file', type=str, default='../dataset/data_200/assist2012/test_sequences.csv',
                        help='the path of test data')
    parser.add_argument('--Q_file', type=str, default='../dataset/data_200/assist2012/Q.npy',
                        help='the path of test data')
    parser.add_argument('--stu_ks_save_file', type=str, default='../stu_ks_save/assist2012_200', help='student ks save file')
    return parser.parse_args()


def calculate_metrics(predictions, targets, mask, concepts, kc_num, is_pkc):
    predictions = (predictions * nn.functional.one_hot(concepts.to(torch.long), kc_num)).sum(-1)
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

    print(f"the number of concepts:{kc_num}")
    print(f"is_mlstm:{is_mlstm}")
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
                      epochs, is_mlstm).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    indicator_result = {'train_loss': [], 'test_auc': [], 'test_acc': []}

    test_data = pd.read_csv(args.test_data_file)
    start_time = time.time()
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    torch.cuda.empty_cache()
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
        end_time = time.time()
        memory_after = process.memory_info().rss
        print(f'memory used: {(memory_after - memory_before) / (1000 * 1000)}MB')
        print(f'train memory allocated: {torch.cuda.memory_allocated() / (1000 * 1000)}MB')
        print(f'Epoch time: {end_time - start_time}')
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss / len(train_dataloader)))
        indicator_result['train_loss'].append(train_loss / len(train_dataloader))

        memory_before1 = process.memory_info().rss
        start_time = time.time()
        torch.cuda.empty_cache()
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

            if is_pkc:
                metrics = calculate_metrics(test_preds, test_targets, test_masks, test_kcs, kc_num, is_pkc)
                print('Test Epoch: {}, ACC: {:.4f}'.format(epoch, metrics[1]))
                indicator_result['test_acc'].append(metrics[1])
                torch.save(stu_ks_tensor, f'{args.stu_ks_save_file}/pkc_5000.pth')
            else:
                metrics = calculate_metrics(test_preds, test_targets, test_masks, test_kcs, kc_num, is_pkc)
                print('Test Epoch: {}, AUC: {:.4f}, ACC: {:.4f}'.format(epoch, metrics[0], metrics[1]))
                indicator_result['test_auc'].append(metrics[0])
                indicator_result['test_acc'].append(metrics[1])
                torch.save(stu_ks_tensor, f'{args.stu_ks_save_file}/pkm_nomelt.pt')
                stu_done_ks = {}
                stu_ks_tensor = stu_ks_tensor.to('cpu').numpy()
                for i in range(len(stu_ks_tensor)):
                    pkm_i = stu_ks_tensor[i]
                    kcs = [int(kc) for kc in set(test_data.iloc[i]['concepts'].split(',')) if kc != '-1']
                    kc_last_pre = {}
                    for kc in kcs:
                        kc_last_pre[kc] = pkm_i[kc]
                    stu_done_ks[i] = kc_last_pre
                stu_true_response = kt_base_eval.preprocess_test_data(test_data)



                for k in [1, 3, 5, 10]:
                    hit, f1, ndcg = kt_base_eval.calculate_metrics(stu_true_response, stu_done_ks, k)
                    print(f'k = {k}, ndcg: {ndcg:.4f}, f1: {f1:.4f}, hit: {hit:.4f}')
                    end_time = time.time()
                    memory_after1 = process.memory_info().rss
                    print(f'test memory used: {(memory_after1 - memory_before1) / (1000 * 1000)}MB')
                    print(f'test memory allocated: {torch.cuda.memory_allocated()/ (1000 * 1000)}MB')
                    print(f'Test time: {end_time - start_time}')
    plot_results(indicator_result)


if __name__ == '__main__':
    args = parse_args()
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f'Total time: {end_time - start_time}')
