import os
import json
import numpy as np
import torch
from itertools import chain

from torch.utils.data import Dataset



# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
SPECIAL_TOKENS = ["<bos>", "<eos>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>"}
                

def tokenize(obj, tokenizer):
    if isinstance(obj, str):  # 来判断obj是否str类别
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))  # 如果是str类别，返回其嵌入值
    if isinstance(obj, dict):  # 判断obj是否是dict类别
        return dict((n, tokenize(o, tokenizer)) for n, o in obj.items())  # items() 函数以列表返回可遍历的(键, 值) 元组数组，同时将o进行标志化
    return list(tokenize(o, tokenizer) for o in obj)  # 将obj每一个元素提取出，进行标志化后，转换为列表，方便修改

class LAWDataset(Dataset):
    def __init__(self, dialogs, tokenizer, pad_token, mask_set = False, p_mask = 0):
        self.dialogs = dialogs
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.mask_set = mask_set
        self.p_mask = p_mask

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):

        que = self.dialogs[index]['question']
        ans = self.dialogs[index]['answer']

        instance, _ = build_input_from_segments(que,ans, self.tokenizer)

        input_ids = torch.Tensor(instance["input_ids"]).long()
        lm_labels = torch.Tensor(instance["lm_labels"]).long()

        

        return input_ids,lm_labels
        

def collate_fn(batch,pad_token,mask_token):
    # pad_token id 20003
    def padding(seq, pad_token,mask_token=None):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:  # 如果seq[0]——caption 仅有一个维度
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(1))).float() * pad_token  # size(-1)，返回倒数第一个维度的大小
        for i in range(len(seq)):
            if mask_token is not None:
                mask_pos = seq[i].eq(mask_token).nonzero()[-1]
                result[i, :mask_pos] = seq[i][:mask_pos]  # 将result中各个位置与seq对应，其余拓展部份为1
                result[i, mask_pos - seq[i].size(0):] = seq[i][mask_pos:]
            else:
                result[i, -seq[i].size(0):] = seq[i]
                
        return result



    input_ids_list,lm_labels_list = [], []
    for i in batch:
        input_ids_list.append(i[0])
        lm_labels_list.append(i[1])


    #text pad
    input_ids = padding(input_ids_list,pad_token,mask_token)
    lm_labels = padding(lm_labels_list,-1)


    return input_ids,lm_labels

    
def build_input_from_segments(que, reply, tokenizer):
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """
    instance = {}
    sequence = [que + [tokenizer.mask_token_id]] + [ [tokenizer.bos_token_id] ] + [reply + [tokenizer.eop_token_id]]

    instance["input_ids"] = list(chain(*sequence))
    instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + sequence[-1]
    

    return instance, sequence


