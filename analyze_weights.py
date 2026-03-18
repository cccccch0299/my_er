"""
SC-MOO Pareto Gating 权重动态性与多样性分析实验

实验一：权重分布可视化 (Diversity Analysis)
  - 三元图 (Ternary Plot)：展示所有学生的 (α, β, γ) 权重分布
  - 核密度估计图 (KDE)：展示 α, β, γ 三个维度的分布曲线

实验二：权重随时间/交互的变化 (Temporal Dynamics)
  - 选取交互序列较长的典型学生，展示权重随学习进程的演变趋势

用法:
  python analyze_weights.py --dataset assist2009 [--checkpoint path/to/model.pt]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from module.rapid_self import RAPID
from module.data_load import Rapid_Dataset


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SC-MOO 权重分析实验')
    parser.add_argument('--dataset', type=str, default='assist2009',
                        choices=['assist2009', 'assist2012', 'assist2017',
                                 'slepemapy', 'XES3G5M', 'nips34'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='已训练模型的 checkpoint 路径 (若不提供则使用随机初始化)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./analysis_output',
                        help='图片输出目录')
    parser.add_argument('--num_students', type=int, default=5,
                        help='实验二选取的学生数量')

    # 模型超参数 (需与训练时一致)
    parser.add_argument('--user_hidden_size', type=int, default=64)
    parser.add_argument('--ex_hidden_size', type=int, default=64)
    parser.add_argument('--LSTM_hidden_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--init_rank_len', type=int, default=150)
    parser.add_argument('--output_type', type=str, default='det')
    parser.add_argument('--lambda_ent', type=float, default=0.01)

    temp_args = parser.parse_known_args()[0]
    dataset = temp_args.dataset

    dataset_configs = {
        'nips34':     {'user_num': 1855,  'ex_num': 948,   'kc_num': 57,   'base_path': './dataset/nips34'},
        'assist2009': {'user_num': 3884,  'ex_num': 17737, 'kc_num': 123,  'base_path': './dataset/assist2009'},
        'assist2012': {'user_num': 27485, 'ex_num': 53070, 'kc_num': 265,  'base_path': './dataset/assist2012'},
        'assist2017': {'user_num': 1709,  'ex_num': 3162,  'kc_num': 102,  'base_path': './dataset/assist2017'},
        'slepemapy':  {'user_num': 85600, 'ex_num': 2913,  'kc_num': 1458, 'base_path': './dataset/slepemapy'},
        'XES3G5M':    {'user_num': 18066, 'ex_num': 7652,  'kc_num': 865,  'base_path': './dataset/XES3G5M'},
    }

    config = dataset_configs[dataset]
    parser.add_argument('--user_num', type=int, default=config['user_num'])
    parser.add_argument('--ex_num', type=int, default=config['ex_num'])
    parser.add_argument('--kc_num', type=int, default=config['kc_num'])
    parser.add_argument('--Q_matrix_file', type=str, default=f"{config['base_path']}/Q.npy")
    parser.add_argument('--test_data_path', type=str, default=f"{config['base_path']}/test_sequences.csv")
    parser.add_argument('--EB_save_path', type=str, default=f"{config['base_path']}/EB.txt")

    args = parser.parse_args()
    return args


# ============================================================
# 工具函数
# ============================================================
def load_model(args):
    """加载 RAPID 模型"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    model = RAPID(args).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        print(type(state_dict))
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"[INFO] 已加载 checkpoint: {args.checkpoint}")
    else:
        if args.checkpoint:
            print(f"[WARN] checkpoint 文件不存在: {args.checkpoint}，使用随机初始化模型")
        else:
            print("[WARN] 未提供 checkpoint，使用随机初始化模型（结果仅供架构验证）")

    model.eval()
    return model, device


def extract_all_weights(model, dataloader, device):
    """
    在全量数据上提取所有学生的 Pareto 权重 (α, β, γ)

    Returns:
        weights_np: [N, 3] numpy array，每行为 [α, β, γ]
        seq_lens: [N] 每个学生的真实序列长度
        user_ids: [N] 学生 ID
    """
    all_weights = []
    all_seq_lens = []
    all_user_ids = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="提取 Pareto 权重"):
            batch_data = [data.to(device) for data in batch_data]
            user, questions, concepts, responses, u_init_rank_list, mask, real_seq_len, kc_ans_situation = batch_data

            # 构建学生状态向量
            S_u = model.build_student_state(kc_ans_situation, questions, concepts,
                                            responses, real_seq_len)
            # 通过 pareto_gating 获得权重
            pareto_weights = model.pareto_gating(S_u)  # [batch_size, 3]

            all_weights.append(pareto_weights.cpu().numpy())
            all_seq_lens.append(real_seq_len.cpu().numpy())
            all_user_ids.append(user.cpu().numpy())

    weights_np = np.concatenate(all_weights, axis=0)
    seq_lens = np.concatenate(all_seq_lens, axis=0)
    user_ids = np.concatenate(all_user_ids, axis=0)

    return weights_np, seq_lens, user_ids


# ============================================================
# 实验一：权重分布可视化
# ============================================================
def plot_ternary(weights_np, output_dir, dataset_name):
    """
    三元图 (Ternary Plot)：使用 plotly 绘制
    三角形三个顶点分别代表 α(Relevance), β(Diversity), γ(Difficulty)
    """
    alpha = weights_np[:, 0]
    beta = weights_np[:, 1]
    gamma = weights_np[:, 2]

    fig = go.Figure(go.Scatterternary(
        a=alpha,
        b=beta,
        c=gamma,
        mode='markers',
        marker=dict(
            size=4,
            color=alpha,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='α (Relevance)'),
            opacity=0.6,
            line=dict(width=0.3, color='white')
        ),
        text=[f'α={a:.3f}, β={b:.3f}, γ={g:.3f}'
              for a, b, g in zip(alpha, beta, gamma)],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=dict(
            text=f'Pareto Weight Distribution ({dataset_name})',
            x=0.5,
            font=dict(size=18)
        ),
        ternary=dict(
            aaxis=dict(title='α (Relevance)', min=0, linewidth=2, ticks='outside'),
            baxis=dict(title='β (Diversity)', min=0, linewidth=2, ticks='outside'),
            caxis=dict(title='γ (Difficulty)', min=0, linewidth=2, ticks='outside'),
        ),
        width=800,
        height=700,
    )

    path_html = os.path.join(output_dir, f'ternary_{dataset_name}.html')
    path_png = os.path.join(output_dir, f'ternary_{dataset_name}.png')
    fig.write_html(path_html)
    fig.write_image(path_png, scale=2)
    print(f"[实验一] 三元图已保存: {path_html}, {path_png}")

    return fig


def plot_kde(weights_np, output_dir, dataset_name):
    """
    核密度估计图 (KDE)：展示 α, β, γ 的分布曲线
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    labels = [r'$\alpha$ (Relevance)', r'$\beta$ (Diversity)', r'$\gamma$ (Difficulty)']
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    for i, (label, color) in enumerate(zip(labels, colors)):
        sns.kdeplot(weights_np[:, i], ax=ax, label=label, color=color,
                    linewidth=2.5, fill=True, alpha=0.2)

    ax.set_xlabel('Weight Value', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Pareto Weight KDE Distribution ({dataset_name})', fontsize=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'kde_{dataset_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[实验一] KDE 图已保存: {path}")


def plot_weight_statistics(weights_np, output_dir, dataset_name):
    """
    补充：权重统计箱线图
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    data_for_box = [weights_np[:, 0], weights_np[:, 1], weights_np[:, 2]]
    labels = [r'$\alpha$ (Rel.)', r'$\beta$ (Div.)', r'$\gamma$ (Diff.)']
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    bp = ax.boxplot(data_for_box, tick_labels=labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='gold', markersize=7))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('Weight Value', fontsize=13)
    ax.set_title(f'Pareto Weight Box Plot ({dataset_name})', fontsize=15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, f'boxplot_{dataset_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[实验一] 箱线图已保存: {path}")


def print_weight_statistics(weights_np):
    """打印权重统计信息"""
    labels = ['α (Relevance)', 'β (Diversity)', 'γ (Difficulty)']
    print("\n" + "=" * 60)
    print("Pareto 权重统计摘要")
    print("=" * 60)
    print(f"{'指标':<18} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8} {'中位数':>8}")
    print("-" * 60)
    for i, label in enumerate(labels):
        col = weights_np[:, i]
        print(f"{label:<18} {col.mean():>8.4f} {col.std():>8.4f} "
              f"{col.min():>8.4f} {col.max():>8.4f} {np.median(col):>8.4f}")
    print(f"\n学生总数: {weights_np.shape[0]}")
    print(f"权重范围跨度 (max - min):")
    for i, label in enumerate(labels):
        col = weights_np[:, i]
        print(f"  {label}: {col.max() - col.min():.4f}")
    print("=" * 60)


# ============================================================
# 实验二：权重随时间/交互的变化
# ============================================================
def simulate_temporal_weights(model, dataset, student_indices, device):
    """
    模拟学生从交互步数 1 到最大长度的权重演变过程。

    对于每个选定的学生，逐步增长其交互序列长度，
    每步重新构建学生状态并计算 Pareto 权重。

    Args:
        model: RAPID 模型
        dataset: Rapid_Dataset
        student_indices: 选取的学生在 dataset 中的索引列表
        device: 设备

    Returns:
        results: dict {student_idx: {'weights': [T, 3], 'seq_len': int, 'user_id': int}}
    """
    results = {}

    for idx in student_indices:
        # 获取该学生的原始数据
        raw_row = dataset.df.iloc[idx]
        user_id = raw_row['uid']

        questions_full = [int(i) for i in raw_row['questions'].split(',')]
        concepts_full = [int(i) for i in raw_row['concepts'].split(',')]
        responses_full = [int(i) for i in raw_row['responses'].split(',')]

        # 截取到有效长度
        try:
            end_idx = concepts_full.index(-1)
        except ValueError:
            end_idx = len(concepts_full)
        questions_full = questions_full[:end_idx]
        concepts_full = concepts_full[:end_idx]
        responses_full = responses_full[:end_idx]

        total_len = len(concepts_full)
        if total_len < 3:
            continue

        max_seq_len = dataset.max_seq_len
        # 获取该学生的候选题目列表
        u_init_rank_list = np.array(dataset.EB[idx])

        temporal_weights = []

        with torch.no_grad():
            for t in range(1, total_len + 1):
                # 截取前 t 步的交互
                q_t = questions_full[:t]
                c_t = concepts_full[:t]
                r_t = responses_full[:t]

                # 构建 kc_ans_situation
                kc_ans = np.full(dataset.kc_num, -1)
                for kc, res in zip(c_t, r_t):
                    kc_ans[kc] = res

                # 填充到 max_seq_len
                q_padded = np.array(q_t + [0] * (max_seq_len - t))
                c_padded = np.array(c_t + [0] * (max_seq_len - t))
                r_padded = np.array(r_t + [0] * (max_seq_len - t))

                # 转 tensor 并加 batch 维度
                questions_ts = torch.tensor(q_padded, dtype=torch.long).unsqueeze(0).to(device)
                concepts_ts = torch.tensor(c_padded, dtype=torch.long).unsqueeze(0).to(device)
                responses_ts = torch.tensor(r_padded, dtype=torch.long).unsqueeze(0).to(device)
                real_seq_len_ts = torch.tensor([t], dtype=torch.long).to(device)
                kc_ans_ts = torch.tensor(kc_ans, dtype=torch.long).unsqueeze(0).to(device)

                # 构建学生状态
                S_u = model.build_student_state(kc_ans_ts, questions_ts, concepts_ts,
                                                responses_ts, real_seq_len_ts)
                # Pareto 门控
                pw = model.pareto_gating(S_u)  # [1, 3]
                temporal_weights.append(pw.cpu().numpy().squeeze())

        temporal_weights = np.array(temporal_weights)  # [T, 3]
        results[idx] = {
            'weights': temporal_weights,
            'seq_len': total_len,
            'user_id': user_id,
        }
        print(f"  学生 idx={idx} (uid={user_id}), 序列长度={total_len}, 权重已提取")

    return results


def select_long_sequence_students(dataset, num_students=5):
    """
    选取具有较长交互序列的典型学生

    策略：按序列长度排序，从不同的长度分位取样，确保多样性
    """
    seq_lens = np.array(dataset.seq_len)
    total = len(seq_lens)

    # 只考虑序列长度 >= 10 的学生
    valid_mask = seq_lens >= 10
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < num_students:
        # 降低门槛
        valid_mask = seq_lens >= 3
        valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < num_students:
        valid_indices = np.arange(min(num_students, total))

    # 按序列长度排序
    sorted_by_len = valid_indices[np.argsort(seq_lens[valid_indices])[::-1]]

    # 从不同分位取样以体现多样性
    n = len(sorted_by_len)
    selected = []
    for i in range(num_students):
        pos = int(i * n / num_students)
        selected.append(sorted_by_len[pos])

    return selected


def plot_temporal_dynamics(temporal_results, output_dir, dataset_name):
    """
    多子图折线图：展示选取学生的权重随时间演变趋势
    """
    num_students = len(temporal_results)
    if num_students == 0:
        print("[WARN] 无有效学生数据用于绘制时序图")
        return

    fig, axes = plt.subplots(1, num_students, figsize=(5 * num_students, 4.5),
                             squeeze=False, sharey=True)
    axes = axes[0]

    colors = {'α': '#e74c3c', 'β': '#2ecc71', 'γ': '#3498db'}

    for ax_idx, (student_idx, data) in enumerate(temporal_results.items()):
        ax = axes[ax_idx]
        weights = data['weights']  # [T, 3]
        T = weights.shape[0]
        steps = np.arange(1, T + 1)

        ax.plot(steps, weights[:, 0], color=colors['α'], linewidth=2,
                label=r'$\alpha$ (Relevance)', marker='o', markersize=2)
        ax.plot(steps, weights[:, 1], color=colors['β'], linewidth=2,
                label=r'$\beta$ (Diversity)', marker='s', markersize=2)
        ax.plot(steps, weights[:, 2], color=colors['γ'], linewidth=2,
                label=r'$\gamma$ (Difficulty)', marker='^', markersize=2)

        ax.set_xlabel('Time Step (Interaction)', fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel('Weight Value', fontsize=11)
        ax.set_title(f'Student {data["user_id"]} (len={T})', fontsize=12)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle(f'Pareto Weight Temporal Dynamics ({dataset_name})', fontsize=16, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, f'temporal_dynamics_{dataset_name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[实验二] 时序演变图已保存: {path}")


def plot_temporal_heatmap(temporal_results, output_dir, dataset_name):
    """
    补充：热力图展示权重随时间的变化梯度
    """
    for student_idx, data in temporal_results.items():
        weights = data['weights']  # [T, 3]
        uid = data['user_id']

        fig, ax = plt.subplots(1, 1, figsize=(max(8, weights.shape[0] * 0.15), 3))

        im = ax.imshow(weights.T, aspect='auto', cmap='YlOrRd',
                       interpolation='bilinear', vmin=0, vmax=1)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([r'$\alpha$ (Rel.)', r'$\beta$ (Div.)', r'$\gamma$ (Diff.)'])
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_title(f'Weight Heatmap - Student {uid}', fontsize=13)

        plt.colorbar(im, ax=ax, label='Weight Value')
        plt.tight_layout()

        path = os.path.join(output_dir, f'heatmap_student_{uid}_{dataset_name}.png')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"[实验二] 热力图已保存到 {output_dir}")


# ============================================================
# 主函数
# ============================================================
def main():
    args = parse_args()

    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"SC-MOO 权重动态性分析 - 数据集: {args.dataset}")
    print(f"{'=' * 60}")

    # 加载模型
    model, device = load_model(args)

    # 加载数据集
    print("[INFO] 加载数据集...")
    dataset = Rapid_Dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[INFO] 数据集大小: {len(dataset)} 个学生")

    # ==================== 实验一 ====================
    print(f"\n{'=' * 60}")
    print("实验一：权重分布可视化 (Diversity Analysis)")
    print(f"{'=' * 60}")

    weights_np, seq_lens, user_ids = extract_all_weights(model, dataloader, device)

    # 打印统计信息
    print_weight_statistics(weights_np)

    # 绘制三元图
    plot_ternary(weights_np, output_dir, args.dataset)

    # 绘制 KDE 图
    plot_kde(weights_np, output_dir, args.dataset)

    # 绘制箱线图
    plot_weight_statistics(weights_np, output_dir, args.dataset)

    # 保存权重数据
    np.save(os.path.join(output_dir, f'pareto_weights_{args.dataset}.npy'), weights_np)
    print(f"[INFO] 权重数据已保存: pareto_weights_{args.dataset}.npy")

    # ==================== 实验二 ====================
    print(f"\n{'=' * 60}")
    print("实验二：权重随时间/交互的变化 (Temporal Dynamics)")
    print(f"{'=' * 60}")

    # 选取长序列学生
    selected_students = select_long_sequence_students(dataset, num_students=args.num_students)
    print(f"[INFO] 选取的学生索引: {selected_students}")
    print(f"[INFO] 对应序列长度: {[dataset.seq_len[i] for i in selected_students]}")

    # 模拟时序权重
    temporal_results = simulate_temporal_weights(model, dataset, selected_students, device)

    # 绘制时序折线图
    plot_temporal_dynamics(temporal_results, output_dir, args.dataset)

    # 绘制热力图
    plot_temporal_heatmap(temporal_results, output_dir, args.dataset)

    # 保存时序数据
    temporal_data = {}
    for idx, data in temporal_results.items():
        temporal_data[int(data['user_id'])] = {
            'weights': data['weights'].tolist(),
            'seq_len': data['seq_len'],
        }
    import json
    with open(os.path.join(output_dir, f'temporal_weights_{args.dataset}.json'), 'w') as f:
        json.dump(temporal_data, f, indent=2)
    print(f"[INFO] 时序权重数据已保存: temporal_weights_{args.dataset}.json")

    print(f"\n{'=' * 60}")
    print(f"分析完成！所有输出已保存至: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
