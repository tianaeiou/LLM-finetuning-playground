from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoConfig

import torch
import time
import json
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import sys


@torch.inference_mode()
def get_response(messages):
    start = time.perf_counter()
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = model.generate(input_ids, max_new_tokens=128, eos_token_id=terminators, do_sample=False, temperature=0.6,
                             top_p=0.9)
    end = time.perf_counter()
    latency = end - start
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True), latency


model_name = "ADMET_reg_clsv1_all_five_shot_checkpoint-500"  # TODO
model_dir = "/root/llama-instruction-tuning/LLaMA-Factory-main/saves/LLaMA3-8B/lora/ADMET_reg_clsv1_all_five_shot_10epoch/checkpoint-500"  # TODO
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.to('cuda')

data_folder = "/root/llama-instruction-tuning/data_ADMET/test_instructions"
result_folder = "/root/llama-instruction-tuning/result_ADMET"
task_category = ["reg"]  # ["reg","cls"]
task_type = ["5-shot"]
SYSTEM_INSTRUCTION = "You are an AI assistant specializing in ADMET property prediction for drug discovery. The user may ask you to predict the absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties of a molecule. You should provide insightful predictions and follow instructions based on your knowledge."

# 对reg / cls任务，做0-shot,1-shot,2-shot,5-shot
for cat in task_category:
    cat_folder = f"{result_folder}/{cat}"
    for task in task_type:
        print(f"performing {cat} task: {task}")
        # 确认保存路径
        os.makedirs(f"{cat_folder}/{task}", exist_ok=True)
        save_dir = f"{cat_folder}/{task}/{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        existing_files = os.listdir(save_dir)
        # 开始读取数据
        data_directory = f"{data_folder}/{cat}/{task}"
        entries = os.listdir(data_directory)
        for file_name in entries:
            if "json" in file_name and file_name not in existing_files:
                with open(f"{data_directory}/{file_name}", "r") as json_file:
                    dataset = json.load(json_file)
                    reply_sentence_list, latency_list = [], []
                    target_outputs = []
                    for data in dataset:
                        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user",
                                                                                        "content": "Instruction: " +
                                                                                                   data[
                                                                                                       "instruction"] + "\n\n" +
                                                                                                   data["input"]}]
                        reply_sentence, latency = get_response(messages)
                        reply_sentence_list.append(reply_sentence)
                        latency_list.append(latency)
                        target_outputs.append(data["output"])
                        result_dict = {
                            "pred_values": reply_sentence_list,
                            "real_values": target_outputs
                        }
                    print("current data_size: ", len(reply_sentence_list))

                    with open(f"{save_dir}/{file_name}", "w") as f:
                        json.dump(result_dict, f)

"""
! CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /autodl-fs/data/model_weight/Meta-Llama-3-8B-Instruct \
    --dataset ADMET_regression_5-shot,ADMET_classification_5-shot \
    --dataset_dir ./data \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ./saves/LLaMA3-8B/lora/ADMET_reg_cls_all_five_shot_10epoch \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
    
"""