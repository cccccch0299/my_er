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


def calculate_recent_accuracy(responses, recent_n=20):
    """
    创新点二：计算学生最近 N 道题的平均正确率

    Args:
        responses: 学生的答题记录列表 (0/1)
        recent_n: 考虑最近的题目数量

    Returns:
        recent_acc: 最近 N 道题的平均正确率
    """
    if len(responses) == 0:
        return 0.5  # 默认值

    # 取最近 N 道题
    recent_responses = responses[-recent_n:] if len(responses) >= recent_n else responses
    recent_acc = np.mean(recent_responses)
    return recent_acc


def calculate_dynamic_delta(recent_acc, base_delta=0.7, alpha=0.3):
    """
    创新点二：根据学生最近正确率计算个性化动态阈值

    公式: user_delta = base_delta + alpha * (0.5 - recent_acc)
    - 如果 recent_acc 很低 (如 0.2)，user_delta 变大，推荐更简单的题
    - 如果 recent_acc 很高 (如 0.8)，user_delta 变小，推荐更难的题

    Args:
        recent_acc: 最近的正确率
        base_delta: 基准难度阈值
        alpha: 调节系数

    Returns:
        user_delta: 个性化的难度阈值
    """
    user_delta = base_delta + alpha * (0.5 - recent_acc)
    # 限制在合理范围内 [0.3, 0.9]
    user_delta = np.clip(user_delta, 0.3, 0.9)
    return user_delta


def eb_filter(Q, pkm, act_level, delta, N, responses_list=None, use_dynamic_delta=True,
              base_delta=0.7, alpha=0.3, recent_n=20):
    """
    习题过滤函数 - 创新点二：支持动态难度阈值

    Args:
        Q: Q矩阵
        pkm: 学生知识点掌握度
        act_level: 学生活跃度
        delta: 固定难度阈值（当 use_dynamic_delta=False 时使用）
        N: 候选题目数量
        responses_list: 学生答题记录列表（用于计算动态阈值）
        use_dynamic_delta: 是否使用动态难度阈值
        base_delta: 基准难度阈值
        alpha: 动态调节系数
        recent_n: 计算正确率时考虑的最近题目数量

    Returns:
        EB: 过滤后的习题列表
    """
    pkm = pkm.cpu().numpy()

    EB = []
    stu_num = pkm.shape[0]
    ex_num = Q.shape[0]

    for i in tqdm(range(stu_num)):
        qid_score = []
        random_select_ex = random.sample(range(0, ex_num), N)
        stu_activate = act_level[i]

        # ========== 创新点二：计算个性化动态阈值 ==========
        if use_dynamic_delta and responses_list is not None:
            recent_acc = calculate_recent_accuracy(responses_list[i], recent_n)
            user_delta = calculate_dynamic_delta(recent_acc, base_delta, alpha)
        else:
            user_delta = delta
        # ========== 创新点二结束 ==========

        for qid in random_select_ex:

            index_1 = np.where(Q[qid] == 1)[0]
            dis = user_delta - np.prod(pkm[i][index_1])

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


def get_student_responses(test_data):
    """
    从测试数据中提取学生的答题记录

    Args:
        test_data: 测试数据 DataFrame

    Returns:
        responses_list: 学生答题记录列表
    """
    responses_list = []
    for _, row in test_data.iterrows():
        responses = [int(r) for r in row['responses'].split(',')]
        responses_list.append(responses)
    return responses_list


if __name__ == '__main__':
    dataset = 'assist2017'
    EB_file = 'EB'
    test_file= f'./dataset/{dataset}/test_sequences.csv'
    #Q_matrix = np.load(f'../dataset/{dataset}/Q.npy')
    Q_matrix = np.load(f'./dataset/{dataset}/Q.npy')
    EB_save_path = f'./dataset/{dataset}/{EB_file}.txt'
    delta = 0.7
    N = 150
    random.seed(2024)

    pkm = torch.load(f'./stu_ks_save/{dataset}/pkm.pt')
    test_data = pd.read_csv(test_file)
    act_level, uids = get_stu_act_level(test_data)

    # ========== 创新点二：获取学生答题记录用于动态阈值 ==========
    responses_list = get_student_responses(test_data)

    # 使用动态难度阈值的 EB 过滤
    EB = eb_filter(Q_matrix, pkm, act_level, delta, N,
                   responses_list=responses_list,
                   use_dynamic_delta=True,
                   base_delta=0.7,
                   alpha=0.3,
                   recent_n=20)
    # ========== 创新点二结束 ==========

    stu_true_response = evaluate4ndcg.preprocess_test_data(test_data)
    stu_rec_ex = np.array(EB)

    EB_save(EB, uids, EB_save_path)
    stu_rec_weak_kc = evaluate4ndcg.preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response)

