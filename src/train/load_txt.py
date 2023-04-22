from fileinput import filename
import json
import os
import numpy as np
from tqdm import tqdm


def tokenize(obj, tokenizer):
    if isinstance(obj, str):  # 来判断obj是否str类别
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))  # 如果是str类别，返回其嵌入值
    if isinstance(obj, dict):  # 判断obj是否是dict类别
        return dict((n, tokenize(o)) for n, o in obj.items())  # items() 函数以列表返回可遍历的(键, 值) 元组数组，同时将o进行标志化
    return list(tokenize(o) for o in obj)  # 将obj每一个元素提取出，进行标志化后，转换为列表，方便修改

def load_feature(tokenizer, text_file, test=False, val= False):  #最终输出dialog[cap,history,que,ans]
    #load_text_feature
    dialog_list = []
    with open(text_file, 'rb') as f:
        str = f.read()
        train_data = json.loads(str)

    pbar = tqdm(total=len(train_data))

    for dialogue_index, dialogue in enumerate(train_data):

        question = tokenize("[Round 0]\n问：{}\n答：".format(dialogue['input']), tokenizer)
        answer = tokenize(dialogue['output'], tokenizer)
        if test is False:
            item = { 'question': question, 'answer': answer}
        else:
            item = { 'question': dialogue['input'], 'answer': dialogue['output']}
        dialog_list.append(item)
        pbar.update(1)
    pbar.close()


    return [dialog_list]

