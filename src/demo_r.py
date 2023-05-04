from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration 
from model.configuration_chatglm import ChatGLMConfig
import sys
import pdb
import logging
import os
import math
import json
import torch
from argparse import ArgumentParser
from retriver.retrieve_law import retriver

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    BottleneckConfig,
)
import faiss                   
import pickle
import argparse
import json
from text2vec import SentenceModel


def retriver(query,t2v_model,index,raw_law_data,args_retriver):
    input_q = query
    while input_q != 'kill':
        q_emb = t2v_model.encode([input_q])
        D, I = index.search(q_emb, args_retriver.top_k)
        output = [raw_law_data[i] for i in I[0]]
        return output

def main(): 
    parser_retriver = argparse.ArgumentParser()
    parser_retriver .add_argument('--embedding_path', default='./retriver/law_embs.pkl', type=str, help='')
    parser_retriver .add_argument('--rawdata_path', default='./retriver/fatiao.json', type=str, help='核心法条文件')
    parser_retriver .add_argument('--top_k', type=int, default=3, help='dst root to faiss database')
    args_retriver  = parser_retriver .parse_args()

    law_embeds = pickle.load(open(args_retriver.embedding_path, 'rb'))
    raw_law_data = json.load(open(args_retriver.rawdata_path, 'rb'))
    
    print('load retriver model')  
    index = faiss.IndexFlatIP(law_embeds.shape[-1])   
    print(index.is_trained)
    index.add(law_embeds)  
    print(index.ntotal)   

    t2v_model = SentenceModel("./text2vec-base-chinese")

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default = "./model")
    parser.add_argument('--peft_path', type=str, default = './peft_r_model/1.p')
    parser.add_argument('--adapter_path', type=str, default = '')
    parser.add_argument('--lora_use', type=bool, default = True)
    parser.add_argument('--adapter_use', type=bool, default = False)
    args = parser.parse_args()
    def read_json(path):
        with open(path, "r") as f:
            return json.load(f)
        
    logger = logging.getLogger(__file__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    model_class = ChatGLMForConditionalGeneration 

    logger.info("Setup Model")
    num_layers = read_json(os.path.join(args.model_path , "config.json"))["num_layers"]
    device_ids = list(range(torch.cuda.device_count()))

    device_map = {}
    device_map["transformer.word_embeddings"] = device_ids[0]
    device_map["transformer.final_layernorm"] = device_ids[-1]
    device_map["lm_head"] = device_ids[0]

    allocations = [
        device_ids[i] for i in
        sorted(list(range(len(device_ids))) * math.ceil(num_layers / len(device_ids)))
    ]
    allocations = allocations[len(allocations)-num_layers:]
    for layer_i, device_id in enumerate(allocations):
        device_map[f"transformer.layers.{layer_i}.input_layernorm"] = device_id
        device_map[f"transformer.layers.{layer_i}.attention.rotary_emb"] = device_id
        device_map[f"transformer.layers.{layer_i}.attention.query_key_value"] = device_id
        device_map[f"transformer.layers.{layer_i}.attention.dense"] = device_id
        device_map[f"transformer.layers.{layer_i}.post_attention_layernorm"] = device_id
        device_map[f"transformer.layers.{layer_i}.mlp.dense_h_to_4h"] = device_id
        device_map[f"transformer.layers.{layer_i}.mlp.dense_4h_to_h"] = device_id

    if args.lora_use:
        model_class = ChatGLMForConditionalGeneration 
        model = model_class.from_pretrained(args.model_path, device_map = device_map).half()
        model.config.use_cache = True # silence the warnings. Please re-enable for inference!
        logger.info("Setup PEFT")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['query_key_value'],
        )
        model = get_peft_model(model, peft_config)

        for layer_i in range(len(model.base_model.model.transformer.layers)):
            device = model.base_model.model.transformer.layers[layer_i].attention.query_key_value.weight.device
            model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_B.half().to(device)
            model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_A.half().to(device)

        if os.path.exists(args.peft_path ):
                #start = read_json(peft_arg.peft_path + '/latest.json')["latest_step"]
                model.load_state_dict(torch.load(args.peft_path), strict=False)
    elif args.adapter_use:
        model_class = ChatGLMForConditionalGeneration 
        model = model_class.from_pretrained(args.model_path, device_map = device_map).half()
        model.config.use_cache = True # silence the warnings. Please re-enable for inference!
        logger.info("Setup PEFT")
        peft_config = BottleneckConfig(
        bottleneck_size=512,
        non_linearity='tanh',
        adapter_dropout=0.1,
        use_parallel_adapter=True,
        use_adapterp=False,
        target_modules={"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
        scaling=1.0,
        bias="none",
        task_type="CAUSAL_LM",
        )#['query_key_value']
        model = get_peft_model(model, peft_config)

        for layer_i in range(len(model.base_model.model.transformer.layers)):
            device = model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.weight.device
            model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.adapter_down.half().to(device)
            model.base_model.model.transformer.layers[layer_i].mlp.dense_h_to_4h.adapter_up.half().to(device)
            model.base_model.model.transformer.layers[layer_i].mlp.dense_4h_to_h.adapter_down.half().to(device)
            model.base_model.model.transformer.layers[layer_i].mlp.dense_4h_to_h.adapter_up.half().to(device)

        if os.path.exists(args.adapter_path ):
                #start = read_json(peft_arg.peft_path + '/latest.json')["latest_step"]
                model.load_state_dict(torch.load(args.adapter_path), strict=False)
    else:
        model_class = ChatGLMForConditionalGeneration 
        model = model_class.from_pretrained(args.model_path, device_map = device_map).half()
        model.config.use_cache = True # silence the warnings. Please re-enable for inference!


    model.eval()
    history=[]
    print("Human:")
    while True:
        line = input()
        query = line.strip()
        if 'new chat' in query:
            history = []
        else:
            str1='-'
            response, his= model.chat(tokenizer, query + '请给出法律依据', history=history)
            law = retriver(query + response,t2v_model,index,raw_law_data,args_retriver)
            prompt = '1、' + str1.join(law[0]) + '2、' + str1.join(law[1]) + '3、'+ str1.join(law[2]) + '请根据以上法律，选择最合适的法律生成问题的合理答复，问题是：' + query
            response1, history= model.chat(tokenizer, prompt, history=history)
            print("\n------------------------------------------------\nAnswer:")
            print(response1)
            print("\n------------------------------------------------\nHuman:")


if __name__ == '__main__':
    main()