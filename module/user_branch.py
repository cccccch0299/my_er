import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from module.lstm import LSTM


class Melt_User_Branch(nn.Module):
    def __init__(self, hidden_units, u_L_min, u_L_max, device, e_max, n_clusters=5):
        """
        User branch: Enhance the tail user representation
        创新点一：基于"学习风格聚类"的差异化增强 (Style-Aware Student Enhancer)

        Args:
            n_clusters: K-Means 聚类的簇数量，代表不同的学习风格
        """
        super(Melt_User_Branch, self).__init__()
        self.W_U = torch.nn.Linear(hidden_units, hidden_units)
        torch.nn.init.xavier_normal_(self.W_U.weight.data)
        self.criterion = torch.nn.MSELoss()

        self.u_L_max = u_L_max
        self.u_L_min = u_L_min
        self.e_max = e_max
        self.pi = np.pi
        self.n_clusters = n_clusters
        self.hidden_units = hidden_units

        self.device = device

    def kmeans_clustering(self, embeddings, n_clusters, n_iters=10):
        """
        PyTorch 实现的 K-Means 聚类算法

        Args:
            embeddings: [N, D] 活跃学生的嵌入表示
            n_clusters: 聚类数量
            n_iters: 迭代次数

        Returns:
            centroids: [K, D] 聚类中心
            labels: [N] 每个样本的簇标签
        """
        N, D = embeddings.shape

        # 如果样本数少于聚类数，调整聚类数
        actual_clusters = min(n_clusters, N)

        # 随机初始化聚类中心（从样本中选择）
        indices = torch.randperm(N)[:actual_clusters]
        centroids = embeddings[indices].clone()

        for _ in range(n_iters):
            # 计算每个样本到各聚类中心的距离
            distances = torch.cdist(embeddings, centroids)  # [N, K]

            # 分配每个样本到最近的聚类中心
            labels = torch.argmin(distances, dim=1)  # [N]

            # 更新聚类中心
            new_centroids = torch.zeros_like(centroids)
            for k in range(actual_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centroids[k] = embeddings[mask].mean(dim=0)
                else:
                    # 如果某个簇为空，保持原来的中心
                    new_centroids[k] = centroids[k]

            centroids = new_centroids

        return centroids, labels

    def assign_cluster(self, few_seq_repre, centroids):
        """
        将冷启动学生分配到最相似的风格簇

        Args:
            few_seq_repre: [M, D] 冷启动学生的嵌入表示
            centroids: [K, D] 聚类中心

        Returns:
            cluster_assignments: [M] 每个冷启动学生的簇分配
        """
        distances = torch.cdist(few_seq_repre, centroids)  # [M, K]
        cluster_assignments = torch.argmin(distances, dim=1)  # [M]
        return cluster_assignments

    def forward(self, seq_encoder, h_user, h_u_emb, h_u_concepts, h_u_mask, user_thres, epoch):
        full_seq_repre = h_u_emb
        h_u_num = h_user.numel()
        w_u_list = []
        for i in range(h_u_num):
            # Calculate the loss coefficient
            u_seq_length = len(torch.masked_select(h_u_concepts[i], h_u_mask[i].to(torch.bool)))
            w_u = (self.pi / 2) * (epoch / self.e_max) + \
                  (self.pi / (2 * (self.u_L_max - user_thres - 1))) * (u_seq_length - user_thres - 1)
            w_u = np.abs(np.sin(w_u))
            w_u_list.append(w_u)

        few_seq = torch.zeros(h_u_num, self.u_L_max, dtype=h_u_concepts.dtype).to(self.device)
        few_seq_mask = []
        fixed_T = 100
        h_u_sub_seq = np.full(h_u_num, fixed_T)
        for i, l in enumerate(h_u_sub_seq):
            few_seq[i, :l] = h_u_concepts[i, :l]
            few_seq_mask.append(torch.tensor([1] * l + [0] * (self.u_L_max - l)))

        few_seq_mask = torch.stack(few_seq_mask)
        few_seq_repre = seq_encoder.sequence_encoding(few_seq, few_seq_mask)
        w_u_list = torch.FloatTensor(w_u_list).view(-1, 1).to(self.device)

        # ========== 创新点一：基于学习风格聚类的差异化增强 ==========
        # Step 1: 对活跃学生的表示进行 K-Means 聚类
        centroids, active_labels = self.kmeans_clustering(
            full_seq_repre.detach(),
            self.n_clusters
        )

        # Step 2: 将冷启动学生分配到最相似的风格簇
        transformed_few = self.W_U(few_seq_repre)
        cluster_assignments = self.assign_cluster(transformed_few.detach(), centroids)

        # Step 3: 计算基于聚类的定向增强 Loss
        # 对于每个冷启动学生，只与同簇的活跃学生计算 MSE Loss
        loss = torch.tensor(0.0, device=self.device)
        for i in range(h_u_num):
            cluster_id = cluster_assignments[i]
            # 找到同簇的活跃学生
            same_cluster_mask = (active_labels == cluster_id)

            if same_cluster_mask.sum() > 0:
                # 只与同簇的活跃学生计算 MSE
                same_cluster_full = full_seq_repre[same_cluster_mask]
                # 计算冷启动学生与同簇活跃学生的平均表示之间的 MSE
                cluster_mean = same_cluster_full.mean(dim=0)
                sample_loss = w_u_list[i] * ((transformed_few[i] - cluster_mean) ** 2).mean()
            else:
                # 如果没有同簇的活跃学生，退化为原始方法
                sample_loss = w_u_list[i] * ((transformed_few[i] - full_seq_repre.mean(dim=0)) ** 2).mean()

            loss = loss + sample_loss

        loss = loss / h_u_num
        # ========== 创新点一结束 ==========

        return loss
