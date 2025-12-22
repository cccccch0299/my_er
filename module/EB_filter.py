import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm
import module.evaluate4ndcg_etc as evaluate4ndcg

def get_stu_act_level(test_data):
    qid_lens = test_data['questions'].apply(lambda x: len(x.split(',')))
    max_qid_len = max(qid_lens)
    act_level = qid_lens / max_qid_len
    return list(act_level), test_data['uid'].tolist()

def eb_filter(Q, pkm, act_level, delta, N):
    pkm = pkm.cpu().numpy()

    EB = []
    stu_num = pkm.shape[0]
    ex_num = Q.shape[0]

    for i in tqdm(range(stu_num)):
        qid_score = []
        random_select_ex = random.sample(range(0, ex_num), N)
        stu_activate = act_level[i]
        for qid in random_select_ex:

            index_1 = np.where(Q[qid] == 1)[0]
            dis = delta - np.prod(pkm[i][index_1])

            stu_qid_score = np.sqrt(dis ** 2)
            qid_score.append((qid, stu_qid_score))

        qid_score_sort = sorted(qid_score, key=lambda x: x[1])
        qid_sort = [i[0] for i in qid_score_sort]
        EB.append(qid_sort)

    return EB


def EB_save(EB, uids, EB_save_path):
    with open(EB_save_path, 'w') as f:
        for i in range(len(EB)):
            f.write(str(uids[i]) + '\t' + ','.join([str(j) for j in EB[i]]) + '\n')



if __name__ == '__main__':
    dataset = 'nips34'
    EB_file = 'EB'
    test_file= f'test_sequences.csv'
    Q_matrix = np.load(f'Q.npy')
    EB_save_path = f'{dataset}/{EB_file}.txt'
    delta = 0.7
    N = 150
    random.seed(2024)

    pkm = torch.load(f'pkm.pt')
    test_data = pd.read_csv(test_file)
    act_level, uids = get_stu_act_level(test_data)
    test_data = pd.read_csv(test_file)

    EB = eb_filter(Q_matrix, pkm, act_level, delta, N)
    stu_true_response = evaluate4ndcg.preprocess_test_data(test_data)
    stu_rec_ex = np.array(EB)

    EB_save(EB, uids, EB_save_path)
    stu_rec_weak_kc = evaluate4ndcg.preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response)

    for k in [1, 3, 5, 10]:
        hit, ndcg, f1, div, valid_test_stu_num = evaluate4ndcg.calculate_metrics(stu_true_response, stu_rec_weak_kc, stu_rec_ex, Q_matrix, k=k)
        print(f'k = {k}, ndcg: {ndcg:.4f}, f1: {f1:.4f}, hit: {hit:.4f}, div: {div:.4f}, valid_test_stu_num: {valid_test_stu_num}')



