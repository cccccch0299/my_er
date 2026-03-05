import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from module import evaluate4ndcg_etc as eval4ndcg

class RAPID(nn.Module):
    def __init__(self, args):
        super(RAPID, self).__init__()
        self.device = args.device
        self.user_num = args.user_num
        self.ex_num = args.ex_num
        self.kc_num = args.kc_num
        self.init_rank_len = args.init_rank_len
        self.Q_matrix =torch.tensor(np.load(args.Q_matrix_file), device=self.device)

        self.user_hidden_size = args.user_hidden_size
        self.ex_hidden_size = args.ex_hidden_size

        self.output_type = args.output_type

        self.istrain = True

        self.user_emb = nn.Embedding(self.user_num, self.user_hidden_size)
        self.ex_emb = nn.Embedding(self.ex_num, self.ex_hidden_size)

        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.LSTM_input_size = self.user_hidden_size + self.ex_hidden_size
        self.BiLSTM_input_size = self.user_hidden_size + self.ex_hidden_size + self.kc_num

        self.LSTM = nn.LSTM(self.LSTM_input_size, self.LSTM_hidden_size, batch_first=True)
        self.BiLSTM = nn.LSTM(self.BiLSTM_input_size, self.LSTM_hidden_size, batch_first=True, bidirectional=True)

        self.MLP_det = nn.Sequential(
            nn.Linear(self.LSTM_hidden_size * 2 + self.kc_num, 1),
            nn.Sigmoid()
        )  # 先用简单一层

        self.MLP_pro1 = nn.Sequential(
            nn.Linear(self.LSTM_hidden_size * 2 + self.kc_num, 1),
            nn.Sigmoid()
        )

        self.MLP_pro2 = nn.Sequential(
            nn.Linear(self.LSTM_hidden_size * 2 + self.kc_num, 1),
            nn.Sigmoid()
        )

        self.MPL_theta = nn.Sequential(
            nn.Linear(self.LSTM_hidden_size * self.kc_num, self.kc_num),
            # nn.Sigmoid()
        )

        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.LSTM_hidden_size, num_heads=8, batch_first=True)
        self.attended_cat_bn = nn.BatchNorm1d(self.kc_num * self.LSTM_hidden_size)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.loss_func = nn.BCELoss()

        # ========== SC-MOO: 状态条件多目标优化框架 ==========
        # 1. 学生状态编码器
        self.state_dim = 64
        self.state_encoder = nn.Sequential(
            nn.Linear(self.kc_num + self.LSTM_hidden_size + 1, 128),  # +1 for active_level
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.state_dim)
        )

        # 2. 三大独立目标打分器
        # 2.1 相关性打分器 (Relevance Scorer)
        self.MLP_relevance = nn.Sequential(
            nn.Linear(self.LSTM_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 2.2 多样性打分器 (Diversity Scorer)
        self.MLP_diversity = nn.Sequential(
            nn.Linear(self.kc_num, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 2.3 难度打分器 (Difficulty/ZPD Scorer)
        self.MLP_difficulty = nn.Sequential(
            nn.Linear(self.kc_num + 1, 64),  # kc_num(题目知识点) + 1(学生掌握度匹配)
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. 状态条件帕累托门控网络 (State-Conditioned Pareto Gating)
        self.pareto_gating = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)  # 输出 [α, β, γ]，和为1
        )

        # 4. 熵正则化系数
        self.lambda_ent = args.lambda_ent if hasattr(args, 'lambda_ent') else 0.01
        # ========== SC-MOO 结束 ==========


    def Listwise_Relevance(self, user, u_init_rank_list):
        x_user, tao_qid = user, u_init_rank_list
        x_u_emb = self.user_emb(x_user).unsqueeze(1).repeat(1, tao_qid.size(1), 1)
        x_q_emb = self.ex_emb(tao_qid)
        x_tao_q_emb = self.Q_matrix[tao_qid]

        item_combine = torch.cat([x_u_emb, x_q_emb, x_tao_q_emb], dim=-1).float()

        relevance_output, _ = self.BiLSTM(item_combine)
        relevance_output = self.dropout(relevance_output)

        return relevance_output


    def Personalized_Diversity(self, user, questions, concepts, u_init_rank_list, real_seq_len, max_b_seq_len=150):
        x_user, x_qid, x_qid_topic, x_real_len, D = user, questions, concepts, real_seq_len, max_b_seq_len
        x_u_t_seq_dict = {i:[] for i in range(self.kc_num)}
        x_u_t_mask_ts = torch.zeros((self.kc_num, x_user.size(0)), dtype=torch.long).to(self.device)
        for i in range(x_user.size(0)):
            i_real_len = x_real_len[i]
            i_qid = x_qid[i][:i_real_len]
            i_qid_topic = x_qid_topic[i][:i_real_len]

            i_u_t_seq_dict = {i: set() for i in range(self.kc_num)}
            for qid, topic in zip(i_qid, i_qid_topic):
                i_u_t_seq_dict[topic.item()].add(qid.item())

            for kc in i_u_t_seq_dict:
                topic_seq = list(i_u_t_seq_dict[kc])
                # --- 新增：如果长度超过 D (150)，进行截断 ---
                if len(topic_seq) > D:
                    topic_seq = topic_seq[:D]  # 截取前150个（由于它是set转list无序的，直接截断即可）
                # ----------------------------------------
                x_u_t_mask_ts[kc][i] = len(topic_seq) - 1
                if len(topic_seq) < D: 
                    topic_seq.extend([0] * (D - len(topic_seq)))
                x_u_t_seq_dict[kc].append(topic_seq)

        x_u_t_seq_ts = (self.kc_num, x_user.size(0), max_b_seq_len)
        x_u_t_seq_t = torch.zeros(x_u_t_seq_ts, dtype=torch.long).to(self.device)
        for kc, seq_list in x_u_t_seq_dict.items():
            x_u_t_seq_t[kc] = torch.tensor(seq_list)

        x_q_emb = self.ex_emb(x_u_t_seq_t)
        x_u_emb = self.user_emb(x_user)
        x_u_emb = x_u_emb.unsqueeze(0).repeat(self.kc_num, 1, 1)
        x_u_emb = x_u_emb.unsqueeze(2).repeat(1, 1, max_b_seq_len, 1)

        item_combine = torch.cat([x_u_emb, x_q_emb], dim=-1).float()
        item_combine_reshape = item_combine.view(-1, item_combine.size(2), item_combine.size(3))

        u_to_topic_vec, _ = self.LSTM(item_combine_reshape)  # [kc_num * batch_size, max_b_seq_len, hidden_size]
        u_to_topic_vec = u_to_topic_vec.view(item_combine.size(0), item_combine.size(1),
                                             item_combine.size(2), u_to_topic_vec.size(-1))

        u_to_topic_vec_final = torch.zeros((self.kc_num, x_user.size(0), self.LSTM_hidden_size),
                                           dtype=u_to_topic_vec.dtype, device=self.device)

        for i in range(self.kc_num):
            for j in range(x_user.size(0)):
                if x_u_t_mask_ts[i][j] > 0:
                    u_to_topic_vec_final[i, j] = u_to_topic_vec[i, j, x_u_t_mask_ts[i][j]]

        u_to_topic_vec_final = u_to_topic_vec_final.permute(1, 0, 2)  # [batch_size, kc_num, hidden_size], 论文中的V矩阵

        atten_output = self.multihead_attention(u_to_topic_vec_final, u_to_topic_vec_final, u_to_topic_vec_final)[0]
        atten_bn = self.attended_cat_bn(atten_output.flatten(start_dim=1))  # [batch_size, kc_num * hidden_size]

        theta = self.MPL_theta(atten_bn)

        d_R = self.Delta_Gain(u_init_rank_list).to(torch.float32)  # [batch_size, init_rank_list_len, kc_num]

        diversity_output = torch.mul(theta.unsqueeze(1), d_R)

        return diversity_output

    def build_student_state(self, kc_ans_situation, questions, concepts, responses, real_seq_len):
        """
        SC-MOO: 构建学生实时认知状态向量

        Args:
            kc_ans_situation: [batch_size, kc_num] 学生知识点掌握度 {-1, 0, 1}
            questions: [batch_size, seq_len] 学生历史交互题目
            concepts: [batch_size, seq_len] 学生历史交互知识点
            responses: [batch_size, seq_len] 学生历史答题结果
            real_seq_len: [batch_size] 学生真实序列长度

        Returns:
            S_u: [batch_size, state_dim] 学生状态向量
        """
        batch_size = kc_ans_situation.size(0)

        # 1. 知识点掌握度特征：将 {-1, 0, 1} 映射为 [0, 0.5, 1]
        kc_state = (kc_ans_situation.float() + 1) / 2.0  # [batch_size, kc_num]

        # 2. 获取 LSTM 最后时刻的 hidden state
        x_user = torch.arange(batch_size, device=self.device)
        x_u_emb = self.user_emb(x_user).unsqueeze(1).repeat(1, questions.size(1), 1)
        x_q_emb = self.ex_emb(questions)
        item_combine = torch.cat([x_u_emb, x_q_emb], dim=-1).float()

        lstm_output, (h_n, c_n) = self.LSTM(item_combine)  # h_n: [1, batch_size, hidden_size]
        lstm_hidden = h_n.squeeze(0)  # [batch_size, LSTM_hidden_size]

        # 3. 活跃度特征：real_seq_len / max_seq_len
        max_seq_len = questions.size(1)
        active_level = (real_seq_len.float() / max_seq_len).unsqueeze(-1)  # [batch_size, 1]

        # 4. 拼接所有特征
        state_input = torch.cat([kc_state, lstm_hidden, active_level], dim=-1)

        # 5. 通过状态编码器
        S_u = self.state_encoder(state_input)  # [batch_size, state_dim]

        return S_u

    def compute_difficulty_features(self, u_init_rank_list, kc_ans_situation):
        """
        SC-MOO: 为每道候选题计算难度特征（ZPD 建模）

        Args:
            u_init_rank_list: [batch_size, rank_len] 候选题目列表
            kc_ans_situation: [batch_size, kc_num] 学生知识点掌握度

        Returns:
            difficulty_features: [batch_size, rank_len, kc_num+1]
        """
        batch_size, rank_len = u_init_rank_list.shape

        # 将 kc_ans_situation 映射为连续值 [0, 0.5, 1]
        kc_mastery = (kc_ans_situation.float() + 1) / 2.0  # [batch_size, kc_num]

        difficulty_features = []

        for i in range(batch_size):
            student_mastery = kc_mastery[i]  # [kc_num]
            student_features = []

            for j in range(rank_len):
                ex_id = u_init_rank_list[i, j]
                ex_kcs = self.Q_matrix[ex_id].float()  # [kc_num] 题目涉及的知识点

                # 计算学生对该题涉及知识点的平均掌握度
                relevant_mastery = student_mastery * ex_kcs  # 只保留题目涉及的知识点
                kc_count = ex_kcs.sum().clamp(min=1)  # 避免除零
                avg_mastery = relevant_mastery.sum() / kc_count  # 标量

                # 拼接特征：[题目知识点向量, 学生平均掌握度]
                feature = torch.cat([ex_kcs, avg_mastery.unsqueeze(0)])  # [kc_num + 1]
                student_features.append(feature)

            difficulty_features.append(torch.stack(student_features))

        return torch.stack(difficulty_features)  # [batch_size, rank_len, kc_num+1]

    def compute_pareto_entropy(self, pareto_weights):
        """
        SC-MOO: 计算帕累托权重的熵（用于正则化）

        Args:
            pareto_weights: [batch_size, 3] 帕累托权重 [α, β, γ]

        Returns:
            entropy: 标量，平均熵
        """
        # 避免 log(0)
        eps = 1e-8
        entropy = -(pareto_weights * torch.log(pareto_weights + eps)).sum(dim=-1).mean()
        return entropy

    def re_ranking(self, relevance_output, diversity_output, difficulty_output,
                   pareto_weights, u_init_rank_list):
        """
        SC-MOO: 状态条件多目标优化重排序

        Args:
            relevance_output: [batch_size, rank_len, LSTM_hidden_size*2] 相关性特征
            diversity_output: [batch_size, rank_len, kc_num] 多样性特征
            difficulty_output: [batch_size, rank_len, kc_num+1] 难度特征
            pareto_weights: [batch_size, 3] 帕累托权重 [α, β, γ]
            u_init_rank_list: [batch_size, rank_len] 初始排序列表

        Returns:
            u_rerank_list: [batch_size, rank_len] 重排序后的列表
            re_ranking_score_sort: [batch_size, rank_len] 排序后的分数
        """
        # 1. 三个独立目标打分器
        S_rel = self.MLP_relevance(relevance_output)    # [batch_size, rank_len, 1]
        S_div = self.MLP_diversity(diversity_output)    # [batch_size, rank_len, 1]
        S_diff = self.MLP_difficulty(difficulty_output) # [batch_size, rank_len, 1]

        # 2. 提取帕累托权重
        alpha = pareto_weights[:, 0:1].unsqueeze(1)   # [batch_size, 1, 1]
        beta = pareto_weights[:, 1:2].unsqueeze(1)    # [batch_size, 1, 1]
        gamma = pareto_weights[:, 2:3].unsqueeze(1)   # [batch_size, 1, 1]

        # 3. 动态标量化融合
        # Final_Score = α * S_rel + β * S_div + γ * S_diff
        final_score = alpha * S_rel + beta * S_div + gamma * S_diff
        final_score = final_score.squeeze(-1)  # [batch_size, rank_len]

        # 4. 排序
        sort_idx = torch.argsort(final_score, dim=-1, descending=True)
        u_rerank_list = u_init_rank_list.gather(dim=1, index=sort_idx)
        re_ranking_score_sort, _ = torch.sort(final_score, dim=1, descending=True)

        return u_rerank_list, re_ranking_score_sort


    def Delta_Gain(self, u_init_rank_list):
        batch_size, seq_len = u_init_rank_list.size()
        d_R_all = []

        c_R_all = [self.Coverage_Fun(u_init_rank) for u_init_rank in u_init_rank_list]

        for u in range(batch_size):
            u_init_rank = u_init_rank_list[u]
            c_R = c_R_all[u]
            d_R = []

            for i in range(seq_len):

                c_R_del_i = torch.cat((u_init_rank[:i], u_init_rank[i + 1:]))
                d_R_ri = c_R - self.Coverage_Fun(c_R_del_i)
                d_R.append(d_R_ri)

            d_R = torch.stack(d_R)
            d_R_all.append(d_R)

        d_R_all = torch.stack(d_R_all)
        return d_R_all

    def Coverage_Fun(self, u_rank):
        tao_qid = 1 - self.Q_matrix[u_rank]
        c_topic_R = torch.prod(tao_qid, dim=0)
        d_R = 1 - c_topic_R

        return d_R

    def get_truth_labels(self, u_init_rank_list, kc_ans_situation, re_ranking_score_sort):
        student_num, rank_len = u_init_rank_list.shape
        ex_num, kc_num = self.Q_matrix.shape

        valid_exercises_score = []
        labels = []

        for student_idx in range(student_num):
            for ex_rank_idx in range(rank_len):
                exercise_id = u_init_rank_list[student_idx, ex_rank_idx].item()
                knowledge_points = self.Q_matrix[exercise_id]  

                ans_situation = kc_ans_situation[student_idx]
                relevant_ans = ans_situation[knowledge_points == 1]

                if len(relevant_ans) == 0:
                    continue  

                if (relevant_ans == -1).any():
                    continue 

                if (relevant_ans == 1).all():
                    label = 0 
                else:
                    label = 1 

                valid_exercises_score.append(re_ranking_score_sort[student_idx, ex_rank_idx])
                labels.append(label)

        valid_exercises_score = torch.stack(valid_exercises_score)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

        return valid_exercises_score, labels

    def forward(self, batch_data):
        """
        SC-MOO: 前向传播

        Args:
            batch_data: (user, questions, concepts, responses, u_init_rank_list,
                        mask, real_seq_len, kc_ans_situation)

        Returns:
            loss: 总损失（BCE Loss + 熵正则化）
            u_rerank_list: 重排序后的题目列表
        """
        user, questions, concepts, responses, u_init_rank_list, mask, real_seq_len, kc_ans_situation = batch_data

        # 1. 相关性特征提取
        relevance_output = self.Listwise_Relevance(user, u_init_rank_list)

        # 2. 多样性特征提取
        diversity_output = self.Personalized_Diversity(user, questions, concepts,
                                                        u_init_rank_list, real_seq_len)

        # 3. 构建学生状态向量
        S_u = self.build_student_state(kc_ans_situation, questions, concepts,
                                        responses, real_seq_len)

        # 4. 帕累托门控网络：动态输出三目标权重
        pareto_weights = self.pareto_gating(S_u)  # [batch_size, 3]

        # 5. 难度特征提取
        difficulty_output = self.compute_difficulty_features(u_init_rank_list, kc_ans_situation)

        # 6. SC-MOO 重排序
        u_rerank_list, re_ranking_score_sort = self.re_ranking(
            relevance_output, diversity_output, difficulty_output,
            pareto_weights, u_init_rank_list
        )

        # 7. 计算 BCE 损失
        valid_exercises_score, labels = self.get_truth_labels(u_rerank_list,
                                                               kc_ans_situation,
                                                               re_ranking_score_sort)
        bce_loss = self.loss_func(valid_exercises_score, labels)

        # 8. 计算帕累托权重的熵正则化
        entropy = self.compute_pareto_entropy(pareto_weights)

        # 9. 总损失：BCE Loss - λ_ent * Entropy
        # 负号是因为我们希望最大化熵（鼓励权重分布更均匀，避免退化）
        loss = bce_loss - self.lambda_ent * entropy

        return loss, u_rerank_list

    def evaluate(self, u_rerank_list, test_data, Q_matrix):
        stu_true_response = eval4ndcg.preprocess_test_data(test_data)
        stu_rec_weak_kc = eval4ndcg.preprocess_stu_rec_ex(u_rerank_list, self.Q_matrix.cpu().numpy(), stu_true_response)

        hits, f1s, ndcgs, divs, recalls = {}, {}, {}, {}, {}
        for k in [1, 3, 5, 10]:
            hit, f1, ndcg, div, recall, _ = eval4ndcg.calculate_metrics(stu_true_response, stu_rec_weak_kc, u_rerank_list, Q_matrix, k)
            hits[f'@{k}'] = round(hit, 4)
            f1s[f'@{k}'] = round(f1, 4)
            ndcgs[f'@{k}'] = round(ndcg, 4)
            divs[f'@{k}'] = round(div, 4)
            recalls[f'@{k}'] = round(recall, 4)

        return {'hit': hits, 'f1': f1s, 'ndcg': ndcgs, 'div': divs, 'recall': recalls}