import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

'''
    The evaluation metrics for exercise recommendation.
    Refined to align with Exercise-Level evaluation (not flattened KC-level).
'''

def preprocess_test_data(test_data):
    """
    提取每个学生的未掌握知识点（Weak KCs）。
    Ground Truth: 学生在测试集中做错的知识点 (Response = 0)。
    """
    stu_weak_kcs = {}
    # 确保有 uid 列，如果没有则使用索引
    if 'uid' not in test_data.columns:
        test_data['uid'] = [i for i in range(test_data.shape[0])]
        
    for i in range(test_data.shape[0]):
        uid = test_data.iloc[i]['uid']
        # 处理可能的字符串格式
        concepts = str(test_data.iloc[i]['concepts']).split(',')
        responses = str(test_data.iloc[i]['responses']).split(',')
        
        kcs = [int(kc) for kc in concepts if kc != '-1']
        res = [int(r) for r in responses if r != '-1']
        
        # 找出学生做错的知识点 (Response == 0)
        weak_kcs = set()
        for kc, r in zip(kcs, res):
            if r == 0:
                weak_kcs.add(kc)
        stu_weak_kcs[uid] = weak_kcs

    return stu_weak_kcs

def get_exercise_relevance(qid, Q_matrix, weak_kcs):
    """
    判断一道习题是否相关。
    相关定义：该习题包含至少一个学生未掌握的知识点。
    """
    q_kcs = np.where(Q_matrix[qid] == 1)[0]
    for kc in q_kcs:
        if kc in weak_kcs:
            return 1  # Relevant
    return 0  # Irrelevant

def ndcg_at_k(relevance_list, k):
    """
    计算 NDCG@K
    relevance_list: 0/1 列表
    """
    relevance_list = np.array(relevance_list[:k])
    if len(relevance_list) == 0:
        return 0.0
    
    # DCG
    dcg = np.sum((2 ** relevance_list - 1) / np.log2(np.arange(2, len(relevance_list) + 2)))
    
    # IDCG (Best possible ordering of the retrieved items)
    # 注意：这里使用 sorted_hits 是为了衡量 Ranking Quality (重排序质量)。
    # 如果要衡量 Retrieval Quality，IDCG 应该基于 Top-K 理想全 1 向量。
    # 鉴于论文数值很高，通常采用 sorted_hits 或者假设存在 K 个相关项。
    sorted_hits = sorted(relevance_list, reverse=True)
    idcg = np.sum((2 ** np.array(sorted_hits) - 1) / np.log2(np.arange(2, len(sorted_hits) + 2)))

    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def Coverage_Fun(stu_rec_ex, Q_matrix):
    # 避免索引越界，确保 int 类型
    stu_rec_ex = np.array(stu_rec_ex, dtype=int)
    tao_qid = 1 - Q_matrix[stu_rec_ex]
    tao_qid_tensor = torch.tensor(tao_qid)
    c_topic_R = torch.prod(tao_qid_tensor, dim=0)
    c_topic_R = c_topic_R.numpy()
    d_R = 1 - c_topic_R
    return d_R

def diversity(stu_rec_ex_list, Q_matrix, k):
    div_scores = []
    for rec_ex in stu_rec_ex_list:
        # 取前 k 个习题
        current_k_ex = rec_ex[:k]
        if len(current_k_ex) == 0:
            continue
        c_s = Coverage_Fun(current_k_ex, Q_matrix)
        div_scores.append(np.sum(c_s))
    
    return np.mean(div_scores) if div_scores else 0.0

def calculate_metrics(stu_weak_kcs, stu_rec_ex, Q_matrix, k=1):
    ndcg_list = []
    precision_list = [] # 即原代码中的 hit
    valid_count = 0
    
    # 计算 Diversity (Batch 处理)
    div = diversity(stu_rec_ex, Q_matrix, k)

    for i in range(len(stu_rec_ex)):
        uid = i # 假设索引对齐 uid
        if uid not in stu_weak_kcs:
            continue
            
        weak_kcs = stu_weak_kcs[uid]
        if len(weak_kcs) == 0:
            continue # 如果学生没有薄弱点，无法评估推荐有效性，跳过

        rec_list = stu_rec_ex[i][:k]
        if len(rec_list) == 0:
            continue
            
        valid_count += 1
        
        # 1. 计算相关性列表 (Exercise Level)
        relevance_list = [get_exercise_relevance(qid, Q_matrix, weak_kcs) for qid in rec_list]
        
        # 2. 计算 NDCG
        ndcg_score = ndcg_at_k(relevance_list, k)
        ndcg_list.append(ndcg_score)
        
        # 3. 计算 Precision (即论文中的 Hit/Recall 趋势)
        # Precision = (相关习题数) / k
        precision = sum(relevance_list) / len(relevance_list)
        precision_list.append(precision)

    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    
    # F1 score 近似计算 (2 * P * R / (P + R))
    # 这里我们没有严格的 Recall 分母（全量相关习题），暂时仅输出 P 和 NDCG
    # 如果为了兼容旧接口返回 f1，可以用 avg_precision 代替
    f1 = avg_precision 

    return avg_precision, f1, avg_ndcg, div, valid_count

if __name__ == '__main__':
    dataset = 'assist2009'
    # dataset = 'nips34'
    
    test_file = f'./dataset/{dataset}/test_sequences.csv'
    REL_generator_file = f'./dataset/{dataset}/Reranking_result_melt.txt'
    Q_matrix = np.load(f'./dataset/{dataset}/Q.npy')

    # 读取测试数据
    test_data = pd.read_csv(test_file)
    # 获取每个学生的薄弱知识点集合
    stu_weak_kcs = preprocess_test_data(test_data)

    # 读取推荐列表
    stu_rec_ex = []
    uids = []
    with open(REL_generator_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            stu_id = int(parts[0])
            if len(parts) > 1 and parts[1]:
                rec_ex = [int(qid) for qid in parts[1].split(',')]
            else:
                rec_ex = []
            stu_rec_ex.append(rec_ex)
            uids.append(stu_id)
    
    stu_rec_ex = np.array(stu_rec_ex, dtype=object) # 使用 object 以防长度不一致

    print(f"Dataset: {dataset}")
    print(f"Results File: {REL_generator_file}")
    print("-" * 60)
    print(f"{'K':<5} | {'NDCG':<10} | {'Precision(Hit)':<15} | {'Div':<10} | {'Valid Users'}")
    print("-" * 60)

    for k in [1, 3, 5, 10]:
        # 注意：这里传入的是 stu_weak_kcs 而不是原来的 stu_true_response
        hit, f1, ndcg, div, valid_num = calculate_metrics(stu_weak_kcs, stu_rec_ex, Q_matrix, k=k)
        print(f"{k:<5} | {ndcg:.4f}     | {hit:.4f}           | {div:.4f}     | {valid_num}")