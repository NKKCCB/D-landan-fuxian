import sys
sys.path.append('/data1/hhc/code/D-LADAN-main/')
import json, random
import os
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers.tokenization_utils_base import BatchEncoding
import torch
import pickle as pkl
from Formatters import SentenceFormatter_D
from tqdm import tqdm
import ipdb
from law_processing import get_law_graph_adj, get_law_graph_large_adj


class DladanCailLoader(Dataset):
    """
    用于加载基于 BERT 的 D-LADAN 模型的数据集（D-LADAN_BERT）。
    [cite_start]论文提到 D-LADAN 的升级版本 D-LADAN_BERT 使用 Transformer 架构直接处理 token 序列 [cite: 541, 542]。
    """
    def __init__(self, data_path, group_indexes, mode, *args, **kwargs) -> None:
        self.mode = mode
        # group_indexes 用于将具体的法条映射到其所属的“混淆社区”（Community）。
        # [cite_start]论文中，D-LADAN 基于先验知识将法条划分为若干社区，社区内的法条在语义上高度相似且易混淆 [cite: 54, 197]。
        self.group_indexes = group_indexes
        self.data = []
        # 加载预处理好的 pickle 数据
        data_dict = pkl.load(open(data_path, 'rb'))
        data_list = list(zip(*[list(v) for v in data_dict.values()]))
        for data in tqdm(data_list, ncols=80):
            # 获取 BERT 需要的输入：input_ids, token_type_ids, attention_mask
            input_ids = data[0]
            token_type_ids = data[1]
            attention_mask = data[2]
            # [cite_start]获取三个子任务的标签：法条(Law Article)、罪名(Charge)、刑期(Term of Penalty) [cite: 42]
            law_label = data[3]
            accu_label = data[4]
            time_label = data[5]
            # 根据法条标签，查找其所属的先验混淆社区 ID (group_label)。
            # [cite_start]模型需要预测该 group_label 以便选择对应的先验区分向量（Prior Distinction Vector） [cite: 200, 335]。
            group_label = self.group_indexes[law_label]
            self.data.append({
                "inputx": input_ids, 
                'mask': attention_mask,
                'segment': token_type_ids,
                'law_label': law_label,
                'accu_label': accu_label,
                'time_label': time_label,
                'group_label': group_label # 将社区标签加入数据中，用于辅助训练
                })
            
    def __getitem__(self, item):
        data = self.data[item % len(self.data)]
        return data
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.data)
        
        
class DladanCailLoader_W2V(Dataset):
    """
    用于加载基于 Word2Vec/RNN 的标准 D-LADAN 模型的数据集。
    [cite_start]对应论文中提到的使用 Bi-GRU 作为基本编码器的情况 [cite: 333, 632]。
    """
    def __init__(self, data_path, group_indexes, mode, *args, **kwargs) -> None:
        self.mode = mode
        self.group_indexes = group_indexes
        self.data = []
        data_dict = pkl.load(open(data_path, 'rb'))
        data_list = list(zip(*[list(v) for v in data_dict.values()]))
        for data in tqdm(data_list, ncols=80):
            # 对于非 BERT 模型，输入通常只是 input_ids (词索引序列)
            input_ids = data[0]
            law_label = data[1]
            accu_label = data[2]
            time_label = data[3]
            # 同样需要获取法条所属的混淆社区标签
            group_label = self.group_indexes[law_label]
            self.data.append({
                "inputx": input_ids, 
                'law_label': law_label,
                'accu_label': accu_label,
                'time_label': time_label,
                'group_label': group_label
                })
            
    def __getitem__(self, item):
        data = self.data[item % len(self.data)]
        return data
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.data)
            
        

if __name__ == "__main__":
    # 构建法条图 (Graph Construction Layer, GCL)。
    # [cite_start]threshold=0.35 对应论文实验设置中的阈值 tau [cite: 631]。
    # [cite_start]该步骤将法条划分为不相连的子图（即社区），权重小于 0.35 的边被移除 [cite: 386-387]。
    # group_indexes 用于后续数据加载器中确定每个法条属于哪个社区。
    law_index_matrix, graph_list_1, graph_membership, adj_matrix = get_law_graph_adj(threshold=0.35)
    group_indexes = list(zip(*graph_membership))[1]
    
    test_path = '/home/nxu/Ladan_tnnls/Bert_Ladan/processed_dataset/CAIL_new/full_doc/small/test_bert_chinese.pkl'    
    test_Dataset = DladanCailLoader(test_path, group_indexes, mode='test')
    
    # 格式化器，负责将 batch 数据处理成模型所需的 tensor 格式
    Formatter = SentenceFormatter_D(mode='train')
    
    def collate_fn(data):
        return Formatter.process(data, "train")

    test_dataset = DataLoader(dataset=test_Dataset, batch_size=2,
                               num_workers=1, drop_last=True,
                               shuffle=True, collate_fn=collate_fn,
                               # sampler=RandomSampler(Dataset)
                               )
    
    print(len(test_dataset))
    
    for step, data in tqdm(enumerate(test_dataset), total=len(test_dataset), position=0, leave=True):
        print(step)
        print(data)
        print(data['inputx'])
        inputx = data['inputx']
        ipdb.set_trace()
        break