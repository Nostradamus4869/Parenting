
import os

import sys
from typing import List
from peft import PeftModel
import fire
import torch
import torch.nn.functional as F
import transformers
import shutil
from datasets import load_dataset
from transformers import AutoConfig
import gc
import pandas as pd

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import time

from transformers import AutoModelForCausalLM,AutoTokenizer

from utils.prompter import Prompter
from transformers import set_seed
set_seed(42)
from utils.dataset_order import get_dataset_order



def train(
    # model/data params
    base_model: str = "/llama-7b-hf",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 20,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    task_id: int = 0, # 1 - 5  
    data_id: int = 0, # 
    beta1: float = 0.85, 
    beta2: float = 0.85,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"with_replay: {with_replay}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    
    dataset_order = ["adherence_0","robustness_0"]
    

    llm_type = prompt_template_name
    #os.environ['CUDA_VISIBLE_DEVICES']='4,6,7'

    print(f"current service name: {dataset_order[data_id]}... begin fine tuning!")
    
    
    data_path = "./data/SGD_single_service_train/" + dataset_order[data_id] + "-train-LLM_"+llm_type+".json"
    output_dir = os.path.join("./checkpoint_files", "importance4_task_id_"+str(task_id)+"_averaging", str(data_id)+"-"+dataset_order[data_id])+"-"+llm_type
    print(f"data path: {data_path}")
    if not os.path.exists(data_path):
        print(f"data_path {data_path} not find!")
        sys.exit(1)
    print(f"output_dir: {output_dir}")
    

    lora_weights = ""

    print(f"lora_weights: {lora_weights}\n")

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        load_in_8bit=False,
        torch_dtype=torch.float32,
        device_map = device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length",
            return_tensors="pt",
        )
        if (result["input_ids"][0, -1] != tokenizer.eos_token_id and 
            result["input_ids"].size(1) < cutoff_len and
            add_eos_token):
            
            # Create a new tensor for EOS token and append it
            eos_tensor = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long)
            result["input_ids"] = torch.cat([result["input_ids"], eos_tensor], dim=1)
            
            # Update the attention mask similarly
            attention_tensor = torch.tensor([[1]], dtype=torch.long)
            result["attention_mask"] = torch.cat([result["attention_mask"], attention_tensor], dim=1)

            # Set the labels by copying input_ids
            result["labels"] = result["input_ids"].clone()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            # tokenized_full_prompt["labels"] = [
            #     -100
            # ] * user_prompt_len + tokenized_full_prompt["labels"][
            #     user_prompt_len:
            # ]  # could be sped up, probably
            mask_value = -100
            labels = tokenized_full_prompt["input_ids"].clone()
            labels[:, :user_prompt_len] = mask_value

            tokenized_full_prompt["labels"] = labels
        return tokenized_full_prompt

    print("fine tune lora from scratch!")
    # https://github.com/tloen/alpaca-lora/issues/44
    
    print(data_path)
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, cache_dir=None)
    else:
        data = load_dataset(data_path)

    print("Train Data Example:\n")
    print(data["train"][0])
    print(generate_and_tokenize_prompt(data["train"][0]))

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None


    steps = 5
    batch_size = 4
    print(type(train_data))
    for i in range(steps):
        start_index = i*batch_size
        train_data_batch = train_data[start_index:start_index+batch_size]
        input_ids_tensor = torch.tensor(train_data_batch['input_ids'], dtype=torch.long).squeeze(1)
        assert isinstance(input_ids_tensor,torch.Tensor)
        assert input_ids_tensor.dim() == 2
        print(f"step: {i}  input size: {input_ids_tensor.shape}")
    
        output = model.generate(input_ids_tensor.to('cuda'), max_length=4096, pad_token_id=tokenizer.pad_token_id)

    output = dict(n=steps*batch_size, sum1=model.model.sum1.to('cpu'), sum2=model.model.sum2.to('cpu'), sum3=model.model.sum3.to('cpu'), sum4=model.model.sum4.to('cpu'), over_zero=model.model.over_zero.to('cpu').float())

    print(output["over_zero"].shape)
    layer_means = torch.mean(output["over_zero"], dim=1)
    print(layer_means)
    log_tensor = torch.log1p(layer_means - layer_means.min())
    normalized_exp = F.softmax(log_tensor, dim=0)
    print(normalized_exp)
    weights_path = "./ipt_file/Weights_averaging_task_id_"+ str(task_id) + "_" + str(data_id)+"-"+dataset_order[data_id]+"-"+llm_type+".pt"
    torch.save(normalized_exp, weights_path)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

      
if __name__ == "__main__":
    fire.Fire(train)
