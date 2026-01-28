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

    def re_ranking(self, relevance_output, diversity_output, u_init_rank_list, output_type='det'):
        rel_div_combine = torch.cat([relevance_output, diversity_output], dim=-1)

        if output_type == 'det':
            re_ranking_score = self.MLP_det(rel_div_combine)  # [batch_size, init_rank_list_len, 1]
            re_ranking_score = re_ranking_score.squeeze(-1)  # [batch_size, init_rank_list_len]
        else:
            re_ranking_score_pro1 = self.MLP_pro1(rel_div_combine)
            re_ranking_score_pro2 = self.MLP_pro2(rel_div_combine)
            if self.istrain:
                random_tensor = torch.randn(re_ranking_score_pro2.size()).to(self.device)
            else:
                random_tensor = 1
            re_ranking_score = re_ranking_score_pro1 + re_ranking_score_pro2 * random_tensor

        sort_idx = torch.argsort(re_ranking_score, dim=-1, descending=True)
        u_rerank_list = u_init_rank_list.gather(dim=1, index=sort_idx)
        re_ranking_score_sort, _ = torch.sort(re_ranking_score, dim=1, descending=True)

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

    def forward(self, batch_data, output_type='det'):
        user, questions, concepts, responses, u_init_rank_list, mask, real_seq_len, kc_ans_situation = batch_data

        relevance_output = self.Listwise_Relevance(user, u_init_rank_list)  

        diversity_output = self.Personalized_Diversity(user, questions, concepts, u_init_rank_list, real_seq_len)

        u_rerank_list, re_ranking_score_sort = self.re_ranking(relevance_output, diversity_output, u_init_rank_list,
                                                               output_type)

        valid_exercises_score, labels = self.get_truth_labels(u_rerank_list, kc_ans_situation, re_ranking_score_sort)

        loss = self.loss_func(valid_exercises_score, labels)

        return loss, u_rerank_list

    def evaluate(self, u_rerank_list, test_data, Q_matrix):
        stu_true_response = eval4ndcg.preprocess_test_data(test_data)
        stu_rec_weak_kc = eval4ndcg.preprocess_stu_rec_ex(u_rerank_list, self.Q_matrix.cpu().numpy(), stu_true_response)

        hits, f1s, ndcgs, divs = {}, {}, {}, {}
        for k in [1, 3, 5, 10]:
            hit, f1, ndcg, div, _ = eval4ndcg.calculate_metrics(stu_true_response, stu_rec_weak_kc, u_rerank_list, Q_matrix, k)
            hits[f'@{k}'], f1s[f'@{k}'], ndcgs[f'@{k}'], divs[f'@{k}'] = round(hit, 4), round(f1, 4), round(ndcg, 4), round(div, 4)

        return {'hit':hits, 'f1':f1s, 'ndcg':ndcgs, 'div':divs}