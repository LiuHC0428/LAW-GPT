from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration 
from model.configuration_chatglm import ChatGLMConfig
import sys
import logging
import os
import math
import json
import torch
from argparse import ArgumentParser
import numpy as np
import random

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    BottleneckConfig,
)

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default = "model")
    parser.add_argument('--model_checkpoint', type=str, default ='model_checkpoint')
    parser.add_argument('--peft_path', type=str, default = 'peft_model/lora.p')
    parser.add_argument('--lora_use', type=bool, default = True)
    parser.add_argument('--adapter_use', type=bool, default = False)
    parser.add_argument('--gpu_id', type=str, default = "0")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    # set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    def read_json(path):
        with open(path, "r") as f:
            return json.load(f)
        
    logger = logging.getLogger(__file__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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

        if os.path.exists(args.peft_path ):
                #start = read_json(peft_arg.peft_path + '/latest.json')["latest_step"]
                model.load_state_dict(torch.load(args.peft_path), strict=False)

    else:
        model_class = ChatGLMForConditionalGeneration 
        model = model_class.from_pretrained(args.model_checkpoint, device_map = device_map).half()
        model.config.use_cache = True # silence the warnings. Please re-enable for inference!


    model.eval()
    print("Human:")
    history=[]
    while True:
        query = input()
        if 'new chat' in query:
            history=[]
        else:
            response, history = model.chat(tokenizer, query, history=history, max_length=500)
        print("\n------------------------------------------------\nAnswer:")
        print(response)
        print("\n------------------------------------------------\nHuman:")


if __name__ == '__main__':
    main()