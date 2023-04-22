import logging
import logging.handlers
import math
from operator import truediv
import os
import random
import numpy as np
import json
import datetime
from pickle import FALSE, TRUE
from pprint import pformat
from argparse import ArgumentParser


os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from lion_pytorch import Lion

from dataset import LAWDataset,collate_fn
from load_txt import load_feature
from model.modeling_chatglm import ChatGLMForConditionalGeneration 
from model.tokenization_chatglm import ChatGLMTokenizer
from transformers import CONFIG_NAME,WEIGHTS_NAME, AdamW
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar,create_lr_scheduler_with_warmup,CosineAnnealingScheduler
from ignite.contrib.handlers.tensorboard_logger import (OptimizerParamsHandler,
                                                        OutputHandler,
                                                        TensorboardLogger)
from ignite.handlers import ModelCheckpoint,global_step_from_engine,Checkpoint

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

import random
import numpy as np
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def model_save(engine,model,save_dir,title):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    if not os.path.exists(os.path.join(save_dir, title)):
        os.makedirs(os.path.join(save_dir, title))
    path = os.path.join(save_dir, title, str(engine.state.epoch)+'.p')
    torch.save(saved_params, path)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def get_data_loaders_new(args, tokenizer):
    train_data = load_feature(tokenizer, args.train_path)

    train_dataset = LAWDataset(train_data[0], tokenizer, tokenizer.pad_token_id)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,  batch_size=args.train_batch_size, num_workers=4, shuffle=(not args.distributed), sampler=train_sampler,drop_last=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id,tokenizer.mask_token_id))
    else:
        train_loader = DataLoader(train_dataset,  batch_size=args.train_batch_size, num_workers=4, shuffle=(not args.distributed),drop_last=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id,tokenizer.mask_token_id))
    return train_loader

def train():
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default= ", help="Path of the trainset")
    parser.add_argument("--model_checkpoint", type=str, default="/model", help="Path, url or short name of the model")
    parser.add_argument("--peft_path", type=str, default=None, help="Model type (gpt or gpt2)")
    parser.add_argument("--save_dir", type=str,default= './Fine_Tuning_Results')
    parser.add_argument("--title", type=str,default= ' ')
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr1", type=float, default=5e-6, help="model1 learning rate")

    parser.add_argument("--lora_rank", type=int, default=8, help="lora_rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora_dropout")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--log_path", type=str, default="log/", help="Log path")

    args = parser.parse_args()

    args.log_path = os.path.join(args.log_path,args.title)

    if args.local_rank in [-1, 0]:
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logger = logging.getLogger("logger")

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join(args.log_path,"test.log"))

    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.WARNING)
    handler2.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    args.distributed = (args.local_rank != -1)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    tokenizer_class = ChatGLMTokenizer         
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)

    logger.info("Setup Model")
    num_layers = read_json(os.path.join(args.model_checkpoint, "config.json"))["num_layers"]
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

    model_class = ChatGLMForConditionalGeneration 
    model = model_class.from_pretrained(args.model_checkpoint, device_map = device_map).half()

    model.model_parallel = True

    for param in model.named_parameters():
        param[1].requires_grad = False

    model.config.use_cache = False # silence the warnings. Please re-enable for inference!

    logger.info("Setup PEFT")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=['query_key_value'],
    )
    model = get_peft_model(model, peft_config)

    for layer_i in range(len(model.base_model.model.transformer.layers)):
        device = model.base_model.model.transformer.layers[layer_i].attention.query_key_value.weight.device
        model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_B.half().to(device)
        model.base_model.model.transformer.layers[layer_i].attention.query_key_value.lora_A.half().to(device)

    if args.peft_path is not None:
        #start = read_json(peft_arg.peft_path + '/latest.json')["latest_step"]
        model.load_state_dict(torch.load(args.peft_path), strict=False)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False,find_unused_parameters=False)
        model = model.module

    optimizer1 = Lion(model.parameters(), lr=args.lr1)


    logger.info("Prepare datasets")
    train_loader = get_data_loaders_new(args, tokenizer)
    #训练
    def train_step(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids,lm_labels = batch

        loss = model(input_ids=input_ids,labels=lm_labels).loss
        loss = (loss / args.gradient_accumulation_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer1.step()
            optimizer1.zero_grad()
        if engine.state.iteration % 500 == 0:
            logger.info("Epoch [%d], Iter [%d] Loss: %.4f" % (engine.state.epoch,  engine.state.iteration,  loss.item()))
        return loss.item()

    trainer = Engine(train_step)

    p_scheduler1 = PiecewiseLinear(optimizer1, "lr", [(0, args.lr1), ((args.n_epochs) * len(train_loader)-5000, 0.0)])


    scheduler1 = create_lr_scheduler_with_warmup(   p_scheduler1,
                                            warmup_start_value=0.0,
                                            warmup_end_value=args.lr1,
                                            warmup_duration=5000)


    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1)
    #trainer.add_event_handler(Events.ITERATION_STARTED, Cosscheduler1)

    
    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)#进度条
        pbar.attach(trainer, metric_names=["loss"])

        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))


        tb_logger = TensorboardLogger(log_dir="./tb_logs/{i}".format(i=args.title))
        #ignite库里面的函数 TensorBoard 处理程序，用于在训练和验证期间记录指标、模型/优化器参数、梯度。

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)

        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer1 , param_name = 'lr',tag='lr1'), event_name=Events.ITERATION_STARTED)

        
        trainer.add_event_handler(Events.EPOCH_COMPLETED, model_save, model, args.save_dir, args.title)

    # Run the trainingd
    if args.local_rank != -1:
        rank = torch.distributed.get_rank()
        # 问题完美解决！
        init_seeds(3407 + rank)
    else:
        init_seeds(3407)
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        tb_logger.close()

if __name__ == "__main__":
    train()
