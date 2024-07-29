# %%
import os
import sys
import pickle
import json
from tdc.single_pred import ADME
import numpy as np
import random
from tdc.benchmark_group import admet_group
from benchmark.utils import normalize, get_min_max_value
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from instruction_tuning.instruction_prompt import *
import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")

all_dataset = ['caco2_wang', 'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'lipophilicity_astrazeneca',
               'solubility_aqsoldb', 'bbb_martins', 'ppbr_az', 'vdss_lombardo', 'cyp2d6_veith', 'cyp3a4_veith',
               'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels',
               'cyp2c9_substrate_carbonmangels', 'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az',
               'herg', 'ames', 'dili', 'ld50_zhu']
regression_task = ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 'vdss_lombardo',
                   'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az', 'ld50_zhu']
classification_task = ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp2d6_veith',
                       'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels',
                       'cyp3a4_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'herg', 'ames', 'dili']
project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"


def calculate_similarity(data_df, ref_df, file_dir, num_of_neighbors=10):
    """

    :param data_df:
    :param ref_df: searching for neighbors here
    :param file_dir: dir to save the pkl file
    :param num_of_neighbors:
    :return:
    """
    if os.path.isfile(file_dir):
        with open(file_dir, 'rb') as file:
            try:
                neighbor_dict = pickle.load(file)
                print("neighbor data loaded!")
                return neighbor_dict
            except Exception as e:
                print("generating neighbor data")
    smiles_list_test, smiles_list_ref = data_df["Drug"].to_list(), ref_df["Drug"].to_list()
    # calculate similarity
    molecules_test = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_test]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_test]
    molecules_ref = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_ref]
    fps_ref = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_ref]
    nearest_neighbors = {i: [] for i in range(len(smiles_list_test))}
    for i, fp1 in enumerate(fps_test):
        print(i)
        for j, fp2 in enumerate(fps_ref):
            if fp1 != fp2:
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                if len(nearest_neighbors[i]) < num_of_neighbors:
                    nearest_neighbors[i].append((j, similarity))
                else:
                    nearest_neighbors[i].append((j, similarity))
                    nearest_neighbors[i].sort(key=lambda x: x[1], reverse=True)
                    nearest_neighbors[i] = nearest_neighbors[i][:num_of_neighbors]
    result = {"smiles": smiles_list_test, "neighbor_index": nearest_neighbors}
    # save to file_dir
    with open(file_dir, 'wb') as f:
        pickle.dump(result, f)
    return result


def get_example_prompt(cur_neighbor_dict, cur_k, cur_idx, min_value, max_value, is_regression=True):
    example_prompt = ""
    for j in range(cur_k):
        neighbor_idx = cur_neighbor_dict["neighbor_index"][cur_idx][j][0]
        mol = Chem.MolFromSmiles(ref_df["Drug"].to_list()[neighbor_idx])
        neighbor_smiles = Chem.MolToSmiles(mol)
        if is_regression:
            neighbor_target = normalize(ref_df["Y"].to_list()[neighbor_idx], min_value, max_value)
        else:
            neighbor_target = int(ref_df["Y"].to_list()[neighbor_idx])
        cur_example = EXAMPLE_TEMPLATE.format(CUR_ANSWER=neighbor_target).replace("CUR_DRUG_SMILES",
                                                                                  str(neighbor_smiles))
        example_prompt += cur_example
    return example_prompt


def process_regression_dataset(data_df, dataset_name, k=0, mode="train", mixed_mode=False):
    prompt_list = []
    cur_target = QUESTION[dataset_name]['target']
    cur_context = CONTEXT[dataset_name]
    min_value, max_value = get_min_max_value(data_df)
    if mixed_mode:
        with open(f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl", 'rb') as file:
            neighbor_dict = pickle.load(file)
        # 70% zero-shot + 30% 1-5shot
        zero_shot_nums = int(len(data_df) * 0.7)
        mixed_few_shot_nums = len(data_df) - zero_shot_nums
        # 随机选 mixed_few_shot_nums 个样本
        mixed_few_shot_idx = random.sample(range(len(data_df)), mixed_few_shot_nums)
        for idx, row in data_df.iterrows():
            mol = Chem.MolFromSmiles(row["Drug"])
            cur_smiles = Chem.MolToSmiles(mol)
            if idx in mixed_few_shot_idx:
                cur_k = random.randint(1, 5)
                example_prompt = get_example_prompt(neighbor_dict, cur_k, idx, min_value, max_value)
                cur_input = PROMPT_TEMPLATE_REG_WITH_EXAMPLE.replace("CUR_CONTEXT_INFO", cur_context).replace(
                    "CUR_DRUG_SMILES", cur_smiles).replace("CUR_TARGET", cur_target).replace("CUR_EXAMPLES",
                                                                                             example_prompt)
            else:
                cur_input = PROMPT_TEMPLATE_REG.replace("CUR_CONTEXT_INFO", cur_context).replace("CUR_DRUG_SMILES",
                                                                                                 cur_smiles).replace(
                    "CUR_TARGET", cur_target)
            res = {"system": SYSTEM_INSTRUCTION, "instruction": INSTRUCTION[dataset], "input": cur_input,
                   "output": str(normalize(row["Y"], min_value, max_value))}
            prompt_list.append(res)
        return prompt_list
    if k == 0:
        for idx, row in data_df.iterrows():
            # convert current smiles to canonical smiles
            mol = Chem.MolFromSmiles(row["Drug"])
            cur_smiles = Chem.MolToSmiles(mol)
            cur_input = PROMPT_TEMPLATE_REG.replace("CUR_CONTEXT_INFO", cur_context).replace("CUR_DRUG_SMILES",
                                                                                             cur_smiles).replace(
                "CUR_TARGET", cur_target)
            res = {"system": SYSTEM_INSTRUCTION, "instruction": INSTRUCTION[dataset], "input": cur_input,
                   "output": str(normalize(row["Y"], min_value, max_value))}
            prompt_list.append(res)
        return prompt_list
    else:
        with open(f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl", 'rb') as file:
            neighbor_dict = pickle.load(file)
        for idx, row in data_df.iterrows():
            example_prompt = get_example_prompt(neighbor_dict, k, idx, min_value, max_value)
            mol = Chem.MolFromSmiles(row["Drug"])
            cur_smiles = Chem.MolToSmiles(mol)
            cur_input = PROMPT_TEMPLATE_REG_WITH_EXAMPLE.replace("CUR_CONTEXT_INFO", cur_context).replace(
                "CUR_DRUG_SMILES", cur_smiles).replace("CUR_TARGET", cur_target).replace("CUR_EXAMPLES", example_prompt)
            res = {"system": SYSTEM_INSTRUCTION, "instruction": INSTRUCTION[dataset], "input": cur_input,
                   "output": str(normalize(row["Y"], min_value, max_value))}
            prompt_list.append(res)
        return prompt_list


def process_classification_dataset(data_df, dataset_name, k=0, mode="train", mixed_mode=False):
    prompt_list = []
    cur_target = QUESTION[dataset_name]['target']
    cur_context = CONTEXT[dataset_name]
    label0 = QUESTION[dataset_name]['label0']
    label1 = QUESTION[dataset_name]['label1']
    if mixed_mode:
        with open(f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl", 'rb') as file:
            neighbor_dict = pickle.load(file)
        # 70% zero-shot + 30% 1-5shot
        zero_shot_nums = int(len(data_df) * 0.7)
        mixed_few_shot_nums = len(data_df) - zero_shot_nums
        # 随机选 mixed_few_shot_nums 个样本
        mixed_few_shot_idx = random.sample(range(len(data_df)), mixed_few_shot_nums)
        for idx, row in data_df.iterrows():
            mol = Chem.MolFromSmiles(row["Drug"])
            cur_smiles = Chem.MolToSmiles(mol)
            if idx in mixed_few_shot_idx:
                cur_k = random.randint(1, 5)
                example_prompt = get_example_prompt(neighbor_dict, cur_k, idx, 0, 1, is_regression=False)
                cur_input = PROMPT_TEMPLATE_CLS_WITH_EXAMPLE.replace("CUR_CONTEXT_INFO", cur_context).replace(
                    "CUR_DRUG_SMILES",
                    cur_smiles).replace(
                    "CUR_EXAMPLES", example_prompt).replace("CUR_TARGET", cur_target).replace("LABEL0_DESCRIPTION",
                                                                                              label0).replace(
                    "LABEL1_DESCRIPTION", label1)
            else:
                cur_input = PROMPT_TEMPLATE_CLS.replace("CUR_CONTEXT_INFO", cur_context).replace("CUR_DRUG_SMILES",
                                                                                                 cur_smiles).replace(
                    "CUR_TARGET", cur_target).replace("LABEL0_DESCRIPTION", label0).replace("LABEL1_DESCRIPTION",
                                                                                            label1)
            res = {"system": SYSTEM_INSTRUCTION, "instruction": INSTRUCTION[dataset], "input": cur_input,
                   "output": str(row["Y"])}
            prompt_list.append(res)
        return prompt_list
    if k == 0:
        # zero-shot
        for idx, row in data_df.iterrows():
            # convert current smiles to canonical smiles
            mol = Chem.MolFromSmiles(row["Drug"])
            cur_smiles = Chem.MolToSmiles(mol)
            cur_input = PROMPT_TEMPLATE_CLS.replace("CUR_CONTEXT_INFO", cur_context).replace("CUR_DRUG_SMILES",
                                                                                             cur_smiles).replace(
                "CUR_TARGET", cur_target).replace("LABEL0_DESCRIPTION", label0).replace("LABEL1_DESCRIPTION", label1)
            res = {"system": SYSTEM_INSTRUCTION, "instruction": INSTRUCTION[dataset], "input": cur_input,
                   "output": str(row["Y"])}
            prompt_list.append(res)
        return prompt_list
    else:
        # few-shot
        with open(f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl", 'rb') as file:
            neighbor_dict = pickle.load(file)
        for idx, row in data_df.iterrows():
            mol = Chem.MolFromSmiles(row["Drug"])
            cur_smiles = Chem.MolToSmiles(mol)
            example_prompt = get_example_prompt(neighbor_dict, k, idx, 0, 1, is_regression=False)
            cur_input = PROMPT_TEMPLATE_CLS_WITH_EXAMPLE.replace("CUR_CONTEXT_INFO", cur_context).replace(
                "CUR_DRUG_SMILES",
                cur_smiles).replace(
                "CUR_EXAMPLES", example_prompt).replace("CUR_TARGET", cur_target).replace("LABEL0_DESCRIPTION",
                                                                                          label0).replace(
                "LABEL1_DESCRIPTION", label1)

            res = {"system": SYSTEM_INSTRUCTION, "instruction": INSTRUCTION[dataset], "input": cur_input,
                   "output": str(row["Y"])}
            prompt_list.append(res)
        return prompt_list

#%%
"""
setting
"""
k = 0
mode = "train"# mode = "train"  # only using train_val_df ["train", "test"]

mixed_mode = False

"""
# step 0: generate neighbor_info for all dataset : f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl"

group = admet_group(path='data/')
save_dir = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/neighbor_info"
import os

existing_files = os.listdir(save_dir)
for dataset in all_dataset:
    benchmark = group.get(dataset)
    train_val_df, test_df = benchmark['train_val'], benchmark['test']
    for mode in ["train", "test"]:
        file_name = f"{dataset}-neighbors-{mode}.pkl"
        if file_name not in existing_files:
            print(file_name)
            if mode == "train":
                ref_df = train_val_df
                data_df = train_val_df
            else:
                ref_df = train_val_df
                data_df = test_df
            neighbor_dict = calculate_similarity(data_df, ref_df,
                                                 f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl")
            print(f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}.pkl, Done!")
"""

regression_task_prompt_all = []
classification_task_prompt_all = []
group = admet_group(path='data/')
for dataset in classification_task:  # ["caco2_wang"]:  # all_dataset:@# regression_task:  # classification_task
    benchmark = group.get(dataset)
    train_val_df, test_df = benchmark['train_val'], benchmark['test']

    #  依据任务类型设置ref_df 和 data_df
    if mode == "train":
        ref_df = train_val_df
        data_df = train_val_df
        save_dir = f"{project_path}/instruction_tuning/instructions"
    else:
        ref_df = train_val_df
        data_df = test_df
        save_dir = f"{project_path}/instruction_tuning/test_instructions/cls/5-shot"
    print(f"Dataset: {dataset}, Mode: {mode}, Size: {len(data_df)}")

    file_name = f"{dataset}_instruction-{k}-shot-{mode}.json" if not mixed_mode else f"{dataset}_instruction-mixed-{mode}.json"

    if dataset in regression_task:
        prompt_list = process_regression_dataset(data_df, dataset, k=k, mode=mode)
        regression_task_prompt_all.extend(prompt_list)
    else:
        prompt_list = process_classification_dataset(data_df, dataset, k=k, mode=mode)
        classification_task_prompt_all.extend(prompt_list)

    # with open(f"{save_dir}/{file_name}", 'w') as json_file:
    #     json.dump(prompt_list, json_file)
    # print(f"{save_dir}/{file_name}, Done!")

# #
print(f"whole dataset length: {len(regression_task_prompt_all), len(classification_task_prompt_all)}")
with open(f"{project_path}/instruction_tuning/instructions/classification_all-0-shot.json", 'w') as json_file:
    json.dump(classification_task_prompt_all, json_file)
