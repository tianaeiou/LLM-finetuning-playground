# %%
""" 在这里算指标！"""
import sys
import json
import os
from tdc.benchmark_group import admet_group
from benchmark.utils import *

sys.path.append("..")

task_record = {"1-shot": 1, "2-shot": 2, "5-shot": 5}  # ["0-shot", "1-shot", "2-shot", "5-shot"]
regression_task = ["caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az", "vdss_lombardo",
                   "half_life_obach", "clearance_microsome_az", "clearance_hepatocyte_az", "ld50_zhu"]
classification_task = ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp2d6_veith',
                       'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels',
                       'cyp3a4_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'herg', 'ames', 'dili']
task_name_dict = {"caco2_wang": "Caco2_Wang", "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca",
                  "solubility_aqsoldb": "Solubility_AqSolDB", "ppbr_az": "PPBR_AZ", "vdss_lombardo": "VDss_Lombardo",
                  "half_life_obach": "Half_Life_Obach", "clearance_microsome_az": "Clearance_Microsome",
                  "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ", "ld50_zhu": "LD50_Zhu"}

available_models = ['ADMET_reg_cls_all_mixed_shot_checkpoint-500', 'ADMET_regression_all_five_shot_5epoch', 'Llama3',
                    'ADMET_reg_cls_all_zero_shot_checkpoint-500', 'ADMET_regression_all_zero_shot_10epoch',
                    'ADMET_regression_all_one_shot_5epoch', 'ADMET_reg_cls_all_five_shot_checkpoint-400',
                    'ADMET_regression_all_two_shot_5epoch', 'ADMET_regression_all_mixed_shot_10epoch',
                    'ADMET_reg_cls_all_one_shot_checkpoint-400', 'ADMET_reg_cls_all_two_shot_checkpoint-400']

result_folder = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/result/reg"
model_list = ['ADMET_reg_clsv1_all_five_shot_checkpoint-500']

result_dict = {}
group = admet_group(path="data/")

# 对于0，1，2，5 shot的数据：
for task_folder, k in task_record.items():
    # 遍历每个dataset
    for dataset in regression_task:
        print(dataset)
        if dataset not in result_dict:
            result_dict[dataset] = {}

        benchmark = group.get(dataset)
        predictions = {}
        name = benchmark["name"]
        train_val, test = benchmark["train_val"], benchmark["test"]
        min_value, max_value = get_min_max_value(train_val, "Y")

        for model in model_list:
            available_files = os.listdir(f"{result_folder}/{task_folder}/{model}")
            # file_name = f"{task_name_dict[dataset]}_instruction-{k}-shot-test-ruleof5.json"
            if f"{dataset}_instruction_{k}-shot.json" in available_files:
                file_name = f"{dataset}_instruction_{k}-shot.json"
            else:
                file_name = f"{dataset}_instruction-{k}-shot-test.json"
            if model not in result_dict[dataset]:
                result_dict[dataset][model] = {}
            print(model, file_name)
            with open(f"{result_folder}/{task_folder}/{model}/{file_name}", "r") as f:
                result = json.load(f)
            predictions[name] = [denormalize(float(f), min_value, max_value) for f in result["pred_values"]]
            result_dict[dataset][model][k] = list(group.evaluate(predictions)[name].values())[0]

# %%
""" 在这里算排名！"""
from benchmark_record import TDC_ADMET_BENCHMARK_REG, TDC_ADMET_BENCHMARK_CLS, Tx_LLM_TDC_ADMET_result, \
    result_best_ruleof5
from instruction_prompt import CLASSIFICATION_TASKS


def get_rank(dataset_name, new_res, benchmark=TDC_ADMET_BENCHMARK_REG, greater_better=False):
    if greater_better:
        count = sum(1 for d in benchmark[dataset_name]["value"] if d > new_res)
    else:
        count = sum(1 for d in benchmark[dataset_name]["value"] if d < new_res)
    print(f"当前新数据排名：{count + 1}")
    return count + 1


spearman_tasks = ["vdss_lombardo", "half_life_obach", "clearance_microsome_az", "clearance_hepatocyte_az"]

ranking_dict = {}
for dataset in regression_task:
    results_all = Tx_LLM_TDC_ADMET_result[dataset]
    ranking_dict[dataset] = {}
    if dataset in spearman_tasks:
        greater_better = True
    else:
        greater_better = False

    model_name = 'Tx-LLM'
    for k in [2, 5]:
        ranking_dict[dataset][k] = get_rank(dataset, results_all, greater_better)
print(ranking_dict)

# %%
""" 在这里算排名 v1"""

from benchmark_record import TDC_ADMET_BENCHMARK_REG, TDC_ADMET_BENCHMARK_CLS, Tx_LLM_TDC_ADMET_result, \
    result_best_ruleof5
from instruction_prompt import CLASSIFICATION_TASKS


def get_rank(dataset_name, new_res, benchmark=TDC_ADMET_BENCHMARK_REG, greater_better=False):
    if greater_better:
        count = sum(1 for d in benchmark[dataset_name]["value"] if d > new_res)
    else:
        count = sum(1 for d in benchmark[dataset_name]["value"] if d < new_res)
    print(f"当前新数据排名：{count + 1}")
    return count + 1


spearman_tasks = ["vdss_lombardo", "half_life_obach", "clearance_microsome_az", "clearance_hepatocyte_az"]

model_name = "ADMET_reg_clsv1_all_five_shot_checkpoint-500"
ranking_dict = {}
for dataset, results_all in result_best_ruleof5.items():
    if dataset in spearman_tasks + CLASSIFICATION_TASKS:
        greater_better = True
    else:
        greater_better = False

    ranking_dict[dataset] = {}

    result = results_all[model_name]
    for k in [5]:  # [0, 1, 2, 5]:
        if dataset in CLASSIFICATION_TASKS:
            benchmark = TDC_ADMET_BENCHMARK_CLS
        else:
            benchmark = TDC_ADMET_BENCHMARK_REG
        ranking_dict[dataset][k] = get_rank(dataset, result[k], benchmark, greater_better)
print(model_name)
print(ranking_dict)

# %%
""" 在这里可视化！ """
""" 这里是单独画每个数据集的！"""
import seaborn as sns
import matplotlib.pyplot as plt
from instruction_prompt import CLASSIFICATION_TASKS, REGRESSION_TASKS
from benchmark_record import result_dict_cls

available_models = ['ADMET_reg_cls_all_mixed_shot_checkpoint-500', 'ADMET_regression_all_five_shot_5epoch', 'Llama3',
                    'ADMET_reg_cls_all_zero_shot_checkpoint-500', 'ADMET_regression_all_zero_shot_10epoch',
                    'ADMET_regression_all_one_shot_5epoch', 'ADMET_reg_cls_all_five_shot_checkpoint-400',
                    'ADMET_regression_all_two_shot_5epoch', 'ADMET_regression_all_mixed_shot_10epoch',
                    'ADMET_reg_cls_all_one_shot_checkpoint-400', 'ADMET_reg_cls_all_two_shot_checkpoint-400']
model_list = ['Llama3', 'ADMET_reg_cls_all_five_shot_checkpoint-400', 'ADMET_reg_cls_all_two_shot_checkpoint-400']
task_list = [2, 5]

color_map = {
    "Llama3": "red",
    "ADMET_reg_cls_all_mixed_shot_checkpoint-500": "blue",
    "ADMET_reg_cls_all_zero_shot_checkpoint-500": "blue",
    "ADMET_reg_cls_all_five_shot_checkpoint-400": "blue",
    "ADMET_reg_cls_all_one_shot_checkpoint-400": "blue",
    "ADMET_reg_cls_all_two_shot_checkpoint-400": "blue",
    "ADMET_regression_all_mixed_shot_10epoch": "orange",
    "ADMET_regression_all_zero_shot_10epoch": "orange",
    "ADMET_regression_all_one_shot_5epoch": "orange",
    "ADMET_regression_all_two_shot_5epoch": "orange",
    "ADMET_regression_all_five_shot_5epoch": "orange",
}

label_map = {
    "Llama3": "Llama3",
    "ADMET_reg_cls_all_mixed_shot_checkpoint-500": "reg+cls hybrid",
    "ADMET_reg_cls_all_zero_shot_checkpoint-500": "reg+cls 0-shot",
    "ADMET_reg_cls_all_five_shot_checkpoint-400": "reg+cls 5-shot",
    "ADMET_reg_cls_all_one_shot_checkpoint-400": "reg+cls 1-shot",
    "ADMET_reg_cls_all_two_shot_checkpoint-400": "reg+cls 2-shot",
    "ADMET_regression_all_mixed_shot_10epoch": "reg hybrid",
    "ADMET_regression_all_zero_shot_10epoch": "reg 0-shot",
    "ADMET_regression_all_one_shot_5epoch": "reg 1-shot",
    "ADMET_regression_all_two_shot_5epoch": "reg 2-shot",
    "ADMET_regression_all_five_shot_5epoch": "reg 5-shot",

}

marker_map = {
    "Llama3": "s",
    "ADMET_reg_cls_all_mixed_shot_checkpoint-500": "o",
    "ADMET_reg_cls_all_zero_shot_checkpoint-500": "+",
    "ADMET_reg_cls_all_five_shot_checkpoint-400": "*",
    "ADMET_reg_cls_all_one_shot_checkpoint-400": "^",
    "ADMET_reg_cls_all_two_shot_checkpoint-400": "p",
    "ADMET_regression_all_mixed_shot_10epoch": "o",
    "ADMET_regression_all_zero_shot_10epoch": "+",
    "ADMET_regression_all_one_shot_5epoch": "^",
    "ADMET_regression_all_two_shot_5epoch": "p",
    "ADMET_regression_all_five_shot_5epoch": "*",

}

import pandas as pd

df = pd.DataFrame({
    'x': [0.0, 1.0, 2.0, 3.0],
    'y': ['0-shot', '1-shot', '2-shot', '5-shot']
})

for dataset in CLASSIFICATION_TASKS:
    cur_result = result_dict_cls[dataset]
    labels = {}
    plt.figure(figsize=(12, 3))
    for i in range(len(task_list)):
        labels[i] = f"{task_list[i]}-shot"
        for model in model_list:
            sns.scatterplot(x=[cur_result[model][task_list[i]]], y=[i], color=color_map[model],
                            marker=marker_map[model], alpha=0.5)
    plt.yticks(ticks=range(len(df['y'])), labels=df['y'])
    plt.title(dataset)
    plt.tight_layout()
    plt.show()

# %%
""" 在这里跟benchmark一起可视化"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from benchmark_record import BENCHMARK_ALL, TDC_ADMET_BENCHMARK_REG, TDC_ADMET_BENCHMARK_CLS, Llama_prompting_result, \
    Tx_LLM_TDC_ADMET_result

benchmark_record = {}
for dataset, cur_dict in BENCHMARK_ALL.items():
    if cur_dict["metric"] not in benchmark_record:
        benchmark_record[cur_dict["metric"]] = [{"value": cur_dict["value"], "dataset": dataset}]
    else:
        benchmark_record[cur_dict["metric"]].append({"value": cur_dict["value"], "dataset": dataset})

model_list = ['ADMET_reg_clsv1_all_five_shot_checkpoint-500']
task_list = [5]

color_map = {
    "Llama3": "red",
    "ADMET_cls_two_shot_checkpoint-400": "purple",
    "ADMET_cls_five_shot_checkpoint-400": "purple",
    "ADMET_reg_cls_all_mixed_shot_checkpoint-500": "blue",
    "ADMET_reg_cls_all_zero_shot_checkpoint-500": "blue",
    "ADMET_reg_cls_all_five_shot_checkpoint-400": "blue",
    "ADMET_reg_cls_all_one_shot_checkpoint-400": "blue",
    "ADMET_reg_cls_all_two_shot_checkpoint-400": "blue",
    "ADMET_regression_all_mixed_shot_10epoch": "orange",
    "ADMET_regression_all_zero_shot_10epoch": "orange",
    "ADMET_regression_all_one_shot_5epoch": "orange",
    "ADMET_regression_all_two_shot_5epoch": "orange",
    "ADMET_regression_all_five_shot_5epoch": "orange",
}
label_map = {
    "Llama3": "Llama3",
    "ADMET_cls_two_shot_checkpoint-400": "cls 2-shot",
    "ADMET_cls_five_shot_checkpoint-400": "cls 5-shot",
    "ADMET_reg_cls_all_mixed_shot_checkpoint-500": "reg+cls hybrid",
    "ADMET_reg_cls_all_zero_shot_checkpoint-500": "reg+cls 0-shot",
    "ADMET_reg_cls_all_five_shot_checkpoint-400": "reg+cls 5-shot",
    "ADMET_reg_cls_all_one_shot_checkpoint-400": "reg+cls 1-shot",
    "ADMET_reg_cls_all_two_shot_checkpoint-400": "reg+cls 2-shot",
    "ADMET_regression_all_mixed_shot_10epoch": "reg hybrid",
    "ADMET_regression_all_zero_shot_10epoch": "reg 0-shot",
    "ADMET_regression_all_one_shot_5epoch": "reg 1-shot",
    "ADMET_regression_all_two_shot_5epoch": "reg 2-shot",
    "ADMET_regression_all_five_shot_5epoch": "reg 5-shot",
}
marker_map = {
    "Llama3": "s",
    "ADMET_cls_two_shot_checkpoint-400": "p",
    "ADMET_cls_five_shot_checkpoint-400": "*",
    "ADMET_reg_cls_all_mixed_shot_checkpoint-500": "o",
    "ADMET_reg_cls_all_zero_shot_checkpoint-500": "+",
    "ADMET_reg_cls_all_five_shot_checkpoint-400": "*",
    "ADMET_reg_cls_all_one_shot_checkpoint-400": "^",
    "ADMET_reg_cls_all_two_shot_checkpoint-400": "p",
    "ADMET_regression_all_mixed_shot_10epoch": "o",
    "ADMET_regression_all_zero_shot_10epoch": "+",
    "ADMET_regression_all_one_shot_5epoch": "^",
    "ADMET_regression_all_two_shot_5epoch": "p",
    "ADMET_regression_all_five_shot_5epoch": "*",

}

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
counter = 0
for metric, results in benchmark_record.items():
    if metric not in ["MAE", "Spearman"]:
        y_value = 1
        labels = {}
        for result in results:
            if result["dataset"] != "!ppbr_az":
                labels[y_value] = result["dataset"]
                ax[counter].scatter(result["value"], [y_value] * len(result["value"]), color="g", alpha=0.3)
                # 07.28
                cur_result = result_dict_cls[result["dataset"]]

                for i in range(len(task_list)):

                    for model in model_list:
                        print(model, color_map[model])
                        ax[counter].scatter([cur_result[model][task_list[i]]], [y_value], color=color_map[model],
                                            marker=marker_map[model], alpha=0.8, label=label_map[model])
                ax[counter].scatter(Tx_LLM_TDC_ADMET_result[result["dataset"]], [y_value], color="black", alpha=1.0,
                                    label="Tx-LLM ADMET")

                # add similarity-based ruleof5 prompting result:
                our_value = Llama_prompting_result[result["dataset"]][2]
                for k, v in our_value.items():
                    ax[counter].scatter(v[metric], y_value, color="purple", marker="*",
                                        alpha=0.5, label="Llama3 2-shot-similarity-ruleof5")

                y_value += 1
        # ax[counter].legend()
        ax[counter].set_title(metric)
        ax[counter].set_yticks(list(labels.keys()))
        ax[counter].set_yticklabels(list(labels.values()))
        counter += 1
plt.tight_layout()
plt.show()

