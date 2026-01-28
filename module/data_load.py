import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class KTDataset(Dataset):
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.df = self.df.sort_values(by='uid').reset_index(drop=True)
        self.seq_len = []
        for i in range(len(self.df)):
            kcs = self.df.iloc[i]['concepts'].split(',')
            try:
                index = kcs.index('-1')
            except:
                index = len(kcs)
            kcs = kcs[:index]
            self.seq_len.append(len(kcs))
        self.max_seq_len = np.max(self.seq_len)
        self.min_seq_len = np.min(self.seq_len)
        self.sort_seq_len = sorted(self.seq_len, reverse=False)
        self.user_threshold = np.quantile(self.sort_seq_len, 0.95)

        self.user_seq_np = np.array(self.seq_len)
        self.h_u_indx = np.where(self.user_seq_np >= self.user_threshold)[0]
        self.h_u_df = self.df.iloc[self.h_u_indx]
        self.h_u_df = self.h_u_df.sort_values(by='uid').reset_index(drop=True)
        self.h_u_len = len(self.h_u_df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index_data = self.df.iloc[idx]

        user = index_data['uid']
        concepts = [int(i) for i in index_data['concepts'].split(',')]
        responses = [int(i) for i in index_data['responses'].split(',')]

        try:
            index = concepts.index(-1)
        except:
            index = len(concepts)
        concepts = concepts[:index]
        responses = responses[:index]

        len_concepts = len(concepts)

        concepts = np.array(concepts + [0] * (self.max_seq_len - len(concepts)))
        responses = np.array(responses + [-0] * (self.max_seq_len - len(responses)))
        mask = np.array([1] * len_concepts + [0] * (self.max_seq_len - len_concepts))

        return user, concepts, responses, mask, len_concepts


class Head_User_Dataset(Dataset):
    def __init__(self, h_u_df, max_seq_len):
        self.h_u_df = h_u_df
        self.h_u_len = len(self.h_u_df)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.h_u_df)

    def __getitem__(self, idx):
        index_data = self.h_u_df.iloc[idx]

        h_user = index_data['uid']
        h_u_concepts = [int(i) for i in index_data['concepts'].split(',')]
        h_u_responses = [int(i) for i in index_data['responses'].split(',')]

        try:
            index = h_u_concepts.index(-1)
        except:
            index = len(h_u_concepts)
        h_u_concepts = h_u_concepts[:index]
        h_u_responses = h_u_responses[:index]

        h_u_len_concepts = len(h_u_concepts)

        h_u_concepts = np.array(h_u_concepts + [0] * (self.max_seq_len - len(h_u_concepts)))
        h_u_responses = np.array(h_u_responses + [0] * (self.max_seq_len - len(h_u_responses)))
        h_u_mask = np.array([1] * h_u_len_concepts + [0] * (self.max_seq_len - h_u_len_concepts))

        return h_user, h_u_concepts, h_u_responses, h_u_mask, h_u_len_concepts

class Rapid_Dataset(Dataset):
    def __init__(self, args):
        self.original_data_file, self.EB_file, self.Q_matrix_file, self.kc_num = (args.test_data_path, args.EB_save_path,
                                                                                  args.Q_matrix_file, args.kc_num)
        self.df = pd.read_csv(self.original_data_file)
        self.df['uid'] = [i for i in range(len(self.df))]
        self.seq_len = []
        for i in range(len(self.df)):

            kcs = self.df.iloc[i]['concepts'].split(',')
            try:
                index = kcs.index('-1')
            except:
                index = len(kcs)
            kcs = kcs[:index]
            self.seq_len.append(len(kcs))

        self.max_seq_len = np.max(self.seq_len)
        self.min_seq_len = np.min(self.seq_len)

        self.EB = []
        with open(self.EB_file, "r") as f:
            for line in f:
                uid, qid = line.strip().split('\t')
                self.EB.append([int(q) for q in qid.split(',')])
        self.EB_len = len(self.EB[0])

        self.Q_matrix = np.load(self.Q_matrix_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index_df = self.df.iloc[idx]
        index_eb = self.EB[idx]

        user = index_df['uid']
        questions = [int(i) for i in index_df['questions'].split(',')]
        concepts = [int(i) for i in index_df['concepts'].split(',')]
        responses = [int(i) for i in index_df['responses'].split(',')]

        try:
            index = concepts.index(-1)
        except:
            index = len(concepts)
        concepts = concepts[:index]
        responses = responses[:index]
        questions = questions[:index]

        kc_ans_situation = np.full(self.kc_num, -1)
        for kc, res in zip(concepts, responses):
            kc_ans_situation[kc] = res

        real_seq_len = len(concepts)

        questions = np.array(questions + [0] * (self.max_seq_len - real_seq_len))
        concepts = np.array(concepts + [0] * (self.max_seq_len - real_seq_len))
        responses = np.array(responses + [0] * (self.max_seq_len - real_seq_len))
        u_init_rank_list = np.array(index_eb)
        mask = np.array([1] * real_seq_len + [0] * (self.max_seq_len - real_seq_len))

        return user, questions, concepts, responses, u_init_rank_list, mask, real_seq_len, kc_ans_situation