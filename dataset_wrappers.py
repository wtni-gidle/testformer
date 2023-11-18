import torch
from torch.utils.data import Dataset
import numpy as np
import random
from anndata import AnnData
from typing import Union

class Tokenizer:
    def __init__(self, num_bins) -> None:
        self.num_bins = num_bins
        self.MASK_ID = 0
        # 词汇表中没有pad，这里的PAD_ID只会用于nn.crossentropyloss中的ignore_index
        self.PAD_ID = -100

    def __call__(self, cell: np.ndarray, mask: bool):
        """
        对基因表达数据(预处理成panglao.h5ad)tokenize, 得到int型数据
        若mask为False, 则只tokenize,
        若mask为True, 则tokenize后做mask处理
        """
        if mask:
            token_ids, cand_mask_index = self.tokenize_cell(cell, mask)
            mlm_input_token_ids, mlm_label = self.replace_masked_tokens(token_ids, cand_mask_index)
            
            return mlm_input_token_ids, mlm_label
        else:
            token_ids = self.tokenize_cell(cell, mask)

            return token_ids

    def tokenize_cell(self, cell: np.ndarray, mask: bool):
        """
        若mask为True, 额外返回非零值的index, 非零的表达值才有可能被mask
        tokenize的规则和performer_pytorch中的TokenEmbedding模块对应
        MASK: 0
        bins: 1,2,3,... 
        """
        if mask:
            nonzero_index = np.nonzero(cell)[0]
        # 若num_bins为5, 则表达值被划为[0, 1), [1, 2), [2, 3), [3, 4), [4, +infinity)
        cell[cell > (self.num_bins - 1)] = self.num_bins - 1
        # [0, 1, …, num_bins - 1]
        token_ids = cell.astype(np.int64)
        # [1, 2, …, num_bins]
        token_ids += 1

        if mask:
            return token_ids, nonzero_index
        else:
            return token_ids
        
    def replace_masked_tokens(
        self, 
        token_ids: np.ndarray, 
        cand_mask_index: Union[list, np.ndarray],
        mask_rate = 0.15, 
        masked_token_rate = 0.8, 
        masked_token_unchanged_rate = 0.1
    ):
        """
        对token_ids进行mask得到mlm_input_token_ids, 以及相应的mlm_label
        """
        # 需要被mask的token数量
        num_masks = max(1, int(len(cand_mask_index) * mask_rate))
        # 需要被mask的token的位置
        cand_mask_index = np.random.choice(cand_mask_index, num_masks)
        # 用于mlm任务的输入序列
        mlm_input_token_ids = token_ids.copy()

        for i in cand_mask_index:
            rand = random.random()
            # 被mask之后的token id
            masked_token_id = None

            if rand <= masked_token_rate:
                masked_token_id = self.MASK_ID
            elif rand < (masked_token_rate + masked_token_unchanged_rate):
                masked_token_id = token_ids[i]
            else:
                # 从[1, ……, num_bins]中随机选择
                masked_token_id = random.choice(range(1, self.num_bins + 1))

            mlm_input_token_ids[i] = masked_token_id
        # 用于mlm任务的label：[pad, true_id, pad]
        mlm_label = np.array([self.PAD_ID] * len(token_ids))
        mlm_label[cand_mask_index] = token_ids[cand_mask_index]

        return mlm_input_token_ids, mlm_label



class PretrainingDataset(Dataset):
    def __init__(self, data: AnnData, num_bins) -> None:
        self.data = data
        self.tokenizer = Tokenizer(num_bins = num_bins)

    def __getitem__(self, index):
        """
        由于mask具有随机性, 为了保证每次输入模型的同一个细胞被mask后的input和label是一样的, 使用index固定随机种子
        """
        np.random.seed(index)
        random.seed(index)
        cell = self.data[index].X.toarray()[0]
        mlm_input_token_ids, mlm_label = self.tokenizer(cell, mask = True)
        x = torch.as_tensor(mlm_input_token_ids, dtype = torch.long)
        y = torch.as_tensor(mlm_label, dtype = torch.long)

        return x, y
    
    def __len__(self):
        return len(self.data)