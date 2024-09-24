# %%
""" official benchmark code """
import os, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def multichoice(model_name):
    QnA_dir_path = 'instruction_tuning/test_instructions/ProteinLMBench.json'
    with open(QnA_dir_path, 'r') as f:
        file_data = json.load(f)

    model_path = f'/you_models_parent_path/{model_name}'
    if 'models--' in model_name:
        fs = f'/data/llm_models/huggingface/hub/{model_name}/snapshots/'
        model_path = fs + os.listdir(f'/data/llm_models/huggingface/hub/{model_name}/snapshots/')[0]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()

    answer_list = [f['answer'] for f in file_data]
    answer_list = [re.search(r'\d+', a).group() for a in answer_list]

    prompt = ("""
Answer the multiple-choice question based solely on the provided context. 
If you are still unsure about the answer, output option 7.
Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: "The correct option is Option X."
Think step by step.
    """)

    question = []
    for f in file_data[:]:
        options = ''
        for o in f['options']:
            options += o + '\n'
        sb = prompt + '\n Question: \n' + f['question'] + '\n Options: \n' + options + '\nThe correct option is:'
        question.append(sb)

    tokenizer.pad_token = tokenizer.eos_token

    chat_model = ('chat' in model_name) or ('Chat' in model_name)
    if 'Yi' in model_name:
        chat_model = False

    output_list = []
    temp = 0.1
    mnt = 20
    for q in tqdm(question[:]):
        if chat_model:
            try:
                if 'Mistral' in model_name:
                    output_list.append(
                        model.chat(tokenizer, q, do_sample=True, max_new_tokens=mnt, temperature=temp, history=[],
                                   eos_token_id=2, pad_token_id=2))
                else:
                    output_list.append(
                        model.chat(tokenizer, q, max_new_tokens=mnt, do_sample=True, temperature=temp, history=[]))
            except:
                output_list.append(model.generate(q, max_new_tokens=mnt, do_sample=True, temperature=temp))
        else:

            a = tokenizer(q, return_tensors="pt", padding=True)
            input_ids = a.input_ids.to('cuda')
            if 'Mistral' in model_name:
                output_list.append(
                    model.generate(input_ids, max_new_tokens=mnt, do_sample=True, temperature=temp, eos_token_id=2,
                                   pad_token_id=2))
            else:
                output_list.append(model.generate(input_ids, max_new_tokens=mnt, do_sample=True, temperature=temp))
    # print(output_list[1])
    try:
        lst = [tokenizer.decode(i[0], skip_special_tokens=True) for i in output_list]
    except:
        lst = [i[0] for i in output_list]

    after = []
    for i, j in zip(lst, question):
        # for i, j in zip(output_list, question):
        after.append(i.replace(j, ''))

    # print(after)
    v_ans = []
    non_number = 0
    for o in after:
        try:
            # v_ans.append(re.search(r'The correct option number is: (\d+)', o).group(1))
            v_ans.append(re.search(r'\d+', o).group())
        except:
            non_number += 1
            v_ans.append("None")

    print(f"The number of answer we could find is {non_number}.")
    psd = 0
    # wrong_list = []
    from datetime import datetime
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    if "/" in model_name:
        model_name_list = model_name.split("/")
        if model_name_list[-1] == "":
            model_name = model_name_list[-2]
        else:
            model_name = model_name_list[-1]

    if not os.path.exists("result/"):
        os.makedirs("result/")

    with open(f'result/raw_result_{formatted_time}_{model_name}.json', 'w') as jj:
        json.dump(after, jj)

    with open(f'result/rst_compar_{formatted_time}_{model_name}.txt', 'w') as results:
        for i in range(len(v_ans)):
            # print(i)
            if v_ans[i] != answer_list[i]:
                results.write(str(v_ans[i]) + "   " + str(answer_list[i]))
                results.write("\n")
                continue
            else:
                results.write("Right")
                psd += 1
                results.write("\n")
    accuracy = psd / len(v_ans)
    print('correct rate: ' + str(psd / len(v_ans)))

    del tokenizer
    del model
    return accuracy


model_name_list = [
    'chat',
    'test'
]
for model in model_name_list:
    acc = multichoice(model)
    print(f"Acc of {model} is: {acc}")

# %%
import sys
import json

sys.path.append("..")
INSTRUCTION = "Please answer the following protein-related multiple-choice question:"
template_da = """Instruction: {INSTRUCTION}

Question: {QUESTION}
{OPTIONS}

Please analyze the question and the provided options, return the most appropriate answer as a single number corresponding to the chosen option and DO NOT RETURN ANYTHING ELSE."""

INSTRUCTION_paper = "Answer the multiple-choice question based solely on the provided context."
template_cot = """Instruction: {INSTRUCTION}

Question: {QUESTION}
{OPTIONS}

If you are still unsure about the answer, output option 7.
Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: "The correct option is Option X."
Think step by step."""

instruction = INSTRUCTION_paper
template = template_cot


def get_option_text(options):
    option_text = "\n".join(options)
    return option_text


def generate_prompt_benchmark(data, instruction=INSTRUCTION):
    question = data["question"]
    options_text = get_option_text(data["options"])
    return template.format(INSTRUCTION=instruction, QUESTION=question, OPTIONS=options_text)


with open("dataset/ProteinLMBench.json", "r") as json_file:
    dataset = json.load(json_file)

benchmark_prompts = []
for data in dataset:
    benchmark_prompts.append(
        {"prompt": generate_prompt_benchmark(data), "output": data["answer"][-1], "explanation": data["explanation"]})
with open("dataset/benchmark_944_prompt_paper.json", 'w') as json_file:
    json.dump(benchmark_prompts, json_file)
# %%
""" Randomly Sample 10% instruction from each SFT dataset to form a new instruction_tuning_set"""
import os
import json
import random

folder_dir = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/ProteinLMBench"
file_list = os.listdir(folder_dir)
question_num = {"sft_uniprot_Post-translational modification.json": 41,
                "sft_uniprot_Tissue specificity.json": 45,
                "sft_uniprot_Function.json": 415,
                "sft_uniprot_Subunit structure.json": 260,
                "sft_uniprot_Involvement in disease.json": 5,
                "enzyme_CoT.json": 10, "sft_uniprot_Induction.json": 23}

tests_all = []
for file in file_list:
    if "json" in file:
        print(file)
        with open(f"{folder_dir}/{file}", "r") as f:
            cur_dataset = json.load(f)
        sample_size = int(len(cur_dataset) * 0.10)
        cur_instructions = random.sample(cur_dataset, sample_size)

        full_range = list(range(len(cur_dataset)))
        unique_integers = random.sample(full_range, 2 * question_num[file])
        cur_tests = []
        count = 0
        while len(cur_tests) < question_num[file] and count < 2 * question_num[file]:
            if cur_dataset[unique_integers[count]] not in cur_instructions:
                cur_tests.append(cur_dataset[unique_integers[count]])
            count += 1
        assert len(cur_tests) == question_num[file]
        # remaining_data = [item for item in cur_dataset if item not in cur_instructions]
        # cur_tests = random.sample(remaining_data, question_num[file])
        tests_all.extend(cur_tests)
        print(file, len(cur_dataset), len(cur_instructions), len(cur_tests))
        with open(f"{folder_dir}/miniset/{file.split('.json')[0]}-mini.json", "w") as f:
            json.dump(cur_instructions, f)
        # with open(f"{folder_dir}/miniset/{file.split('.json')[0].replace(" ","-")}-mini.json", "w") as f:
        #     json.dump(cur_instructions)
with open(f"{folder_dir}/miniset/{len(tests_all)}-benchmark.json", "w") as f:
    json.dump(tests_all, f)
# %%
""" evaluation """
import json

with open("SFT_dataset/test/ProteinLMBench/Llama3-ProteinLMBenchmark-results.json", "r") as f:
    result = json.load(f)
correct_num = 0
for pre, tar in zip(result["pred_values"], result["real_values"]):
    correct_num += 1 if int(pre) == int(tar) else 0
