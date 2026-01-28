import random

import numpy
import pandas as pd
import numpy as np
import torch
import ast
from sklearn.metrics import f1_score

'''
    The evaluation metrics for exercise recommendation refer to the calculation process described in the KDD'2023 MMER paper.
'''


def preprocess_test_data(test_data):
    stu_true_response = {}
    test_data['uid'] = [i for i in range(test_data.shape[0])]
    for i in range(test_data.shape[0]):
        uid = test_data.iloc[i]['uid']
        kcs = [int(kc) for kc in test_data.iloc[i]['concepts'].split(',') if kc != '-1']
        responses = [int(r) for r in test_data.iloc[i]['responses'].split(',') if r != '-1']
        kc_last_response = {}
        for kc, r in zip(kcs, responses):
            kc_last_response[kc] = r
        stu_true_response[uid] = kc_last_response

    return stu_true_response


def preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response):
    stu_rec_weak_kc = {}
    for uid in range(stu_rec_ex.shape[0]):
        had_done_kc = stu_true_response[uid]
        rec_ex = stu_rec_ex[uid]
        rec_kc, rec_weak_kc = [], []
        for qid in rec_ex:
            kc = np.where(Q_matrix[qid, :] == 1)[0].tolist()
            rec_kc.extend(kc)
        for kc in rec_kc:
            if kc not in rec_weak_kc and kc in had_done_kc:
                rec_weak_kc.append(kc)
        stu_rec_weak_kc[uid] = rec_weak_kc

    return stu_rec_weak_kc

def ndcg_at_k(hits, k):
    if len(hits) < k:
        return -1

    dcg = np.sum((2 ** np.array(hits[:k]) - 1) / np.log2(np.arange(2, k + 2)))

    sorted_hits = sorted(hits, reverse=True)
    idcg = np.sum((2 ** np.array(sorted_hits[:k]) - 1) / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg

def Coverage_Fun(stu_rec_ex, Q_matrix):
    tao_qid = 1 - Q_matrix[stu_rec_ex]
    tao_qid_tensor = torch.tensor(tao_qid)
    c_topic_R = torch.prod(tao_qid_tensor, dim=0)
    c_topic_R = c_topic_R.numpy()

    d_R = 1 - c_topic_R

    return d_R

def diversity(stu_rec_ex, Q_matrix, k):
    diversity = []
    for i in range(len(stu_rec_ex)):
        rec_ex = stu_rec_ex[i][ : k]
        c_s = Coverage_Fun(rec_ex, Q_matrix)
        diversity.append(np.sum(c_s))
    return np.mean(diversity)

def calculate_metrics(stu_true_response, stu_rec_weak_kc, stu_rec_ex, Q_matrix, k=1):
    hit, ndcg_list, f1 = 0, [], 0
    hits = []
    valid_test_stu_num = 0
    div = diversity(stu_rec_ex, Q_matrix, k)

    for i in range(len(stu_rec_ex)):
        kc_true_score = {kc: 1 - stu_true_response[i][kc] for kc in stu_true_response[i]}
        rank_kc = stu_rec_weak_kc[i][:k]

        if len(rank_kc) == 0:
            continue
        valid_test_stu_num += 1

        hit_list = [kc_true_score[kc] for kc in rank_kc]
        hits.append(hit_list)

        hit += sum(hit_list) / len(hit_list)

        temp_ndcg = ndcg_at_k(hit_list, k)
        if temp_ndcg != -1:
            ndcg_list.append(temp_ndcg)

        t = [1] * len(hit_list)
        f1 += f1_score(t, hit_list)

    hit = hit / valid_test_stu_num
    ndcg = np.mean(ndcg_list)
    f1 = f1 / valid_test_stu_num

    return hit, f1, ndcg, div, valid_test_stu_num


if __name__ == '__main__':
    dataset = 'assist2012'
    #'nips34'
    #test_file = f'./dataset/{dataset}/train_valid_sequences.csv'
    test_file = f'./dataset/{dataset}/test_sequences.csv'
    #REL_generator_file = f'./dataset/{dataset}/EB.txt'
    REL_generator_file = f'./dataset/{dataset}/Reranking_result_melt.txt'
    Q_matrix = np.load(f'./dataset/{dataset}/Q.npy')

    test_data = pd.read_csv(test_file)
    stu_true_response = preprocess_test_data(test_data)

    stu_rec_ex = []
    with open(REL_generator_file, 'r') as f:
        for line in f.readlines():
            stu_id, rec_ex = line.strip().split('\t')
            rec_ex = [int(qid) for qid in rec_ex.split(',')]
            stu_rec_ex.append(rec_ex)
    stu_rec_ex = np.array(stu_rec_ex)

    stu_rec_weak_kc = preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response)


    for k in [1, 3, 5, 10]:
        hit, ndcg, f1, div, valid_test_stu_num = calculate_metrics(stu_true_response, stu_rec_weak_kc, stu_rec_ex, Q_matrix, k=k)
        print(f'k = {k}, ndcg: {ndcg:.4f}, f1: {f1:.4f}, hit: {hit:.4f}, div: {div:.4f}, valid_test_stu_num: {valid_test_stu_num}')
