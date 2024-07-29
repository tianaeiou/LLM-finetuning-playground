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
task_name_dict = {"caco2_wang": "Caco2_Wang", "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca",
                  "solubility_aqsoldb": "Solubility_AqSolDB", "ppbr_az": "PPBR_AZ", "vdss_lombardo": "VDss_Lombardo",
                  "half_life_obach": "Half_Life_Obach", "clearance_microsome_az": "Clearance_Microsome",
                  "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ", "ld50_zhu": "LD50_Zhu"}

# TODO: UPDATE
model_list = ['ADMET_reg_cls_all_mixed_shot_checkpoint-500', 'ADMET_regression_all_five_shot_5epoch', 'Llama3',
              'ADMET_reg_cls_all_zero_shot_checkpoint-500', 'ADMET_regression_all_zero_shot_10epoch',
              'ADMET_regression_all_one_shot_5epoch', 'ADMET_reg_cls_all_five_shot_checkpoint-400',
              'ADMET_regression_all_two_shot_5epoch', 'ADMET_regression_all_mixed_shot_10epoch',
              'ADMET_reg_cls_all_one_shot_checkpoint-400', 'ADMET_reg_cls_all_two_shot_checkpoint-400']
# result_folder = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/result/reg"
result_folder = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/result/reg"

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

        # TODO: check file_name

        for model in model_list:
            available_files = os.listdir(f"{result_folder}/{task_folder}/{model}")
            file_name = f"{task_name_dict[dataset]}_instruction-{k}-shot-test-ruleof5.json"
            # if f"{dataset}_instruction_{k}-shot.json" in available_files:
            #     file_name = f"{dataset}_instruction_{k}-shot-ruleof5.json"
            # else:
            #     file_name = f"{dataset}_instruction-{k}-shot-test-ruleof5.json"
            if model not in result_dict[dataset]:
                result_dict[dataset][model] = {}
            print(model, file_name)
            with open(f"{result_folder}/{task_folder}/{model}/{file_name}", "r") as f:
                result = json.load(f)
            predictions[name] = [denormalize(float(f), min_value, max_value) for f in result["pred_values"]]
            result_dict[dataset][model][k] = list(group.evaluate(predictions)[name].values())[0]
# %%
""" 算指标的结果在这里"""
result_dict = {'caco2_wang': {'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 0.554, 1: 0.48, 2: 0.447, 5: 0.466},
                              'ADMET_regression_all_five_shot_5epoch': {0: 0.781, 1: 0.664, 2: 0.559, 5: 0.57},
                              'Llama3': {0: 0.845, 1: 0.627, 2: 0.616, 5: 0.617},
                              'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 0.539, 1: 0.467, 2: 0.452, 5: 0.468},
                              'ADMET_regression_all_zero_shot_10epoch': {0: 0.497, 1: 0.517, 2: 0.48, 5: 0.483},
                              'ADMET_regression_all_one_shot_5epoch': {0: 0.612, 1: 0.565, 2: 0.578, 5: 0.585},
                              'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 0.75, 1: 0.576, 2: 0.966, 5: 0.918},
                              'ADMET_regression_all_two_shot_5epoch': {0: 0.777, 1: 0.587, 2: 0.576, 5: 0.567},
                              'ADMET_regression_all_mixed_shot_10epoch': {0: 0.496, 1: 0.514, 2: 0.479, 5: 0.485},
                              'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 1.062, 1: 0.687, 2: 0.903, 5: 0.55},
                              'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 0.749, 1: 0.537, 2: 0.573, 5: 0.458}},
               'lipophilicity_astrazeneca': {
                   'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 0.992, 1: 0.858, 2: 0.787, 5: 0.755},
                   'ADMET_regression_all_five_shot_5epoch': {0: 1.847, 1: 0.92, 2: 0.867, 5: 0.783},
                   'Llama3': {0: 0.959, 1: 0.898, 2: 0.89, 5: 0.861},
                   'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 0.966, 1: 0.856, 2: 0.788, 5: 0.761},
                   'ADMET_regression_all_zero_shot_10epoch': {0: 1.108, 1: 0.855, 2: 0.841, 5: 0.81},
                   'ADMET_regression_all_one_shot_5epoch': {0: 0.968, 1: 0.856, 2: 0.814, 5: 0.755},
                   'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 1.579, 1: 0.917, 2: 0.88, 5: 0.787},
                   'ADMET_regression_all_two_shot_5epoch': {0: 1.712, 1: 0.867, 2: 0.827, 5: 0.776},
                   'ADMET_regression_all_mixed_shot_10epoch': {0: 1.104, 1: 0.858, 2: 0.841, 5: 0.813},
                   'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 1.535, 1: 0.903, 2: 0.879, 5: 0.809},
                   'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 1.602, 1: 0.894, 2: 0.873, 5: 0.783}},
               'solubility_aqsoldb': {
                   'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 2.186, 1: 2.069, 2: 1.779, 5: 1.729},
                   'ADMET_regression_all_five_shot_5epoch': {0: 2.712, 1: 1.62, 2: 1.581, 5: 1.592},
                   'Llama3': {0: 2.9, 1: 1.574, 2: 1.599, 5: 1.635},
                   'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 2.139, 1: 2.022, 2: 1.79, 5: 1.741},
                   'ADMET_regression_all_zero_shot_10epoch': {0: 2.378, 1: 1.844, 2: 1.666, 5: 1.689},
                   'ADMET_regression_all_one_shot_5epoch': {0: 2.183, 1: 1.65, 2: 1.637, 5: 1.714},
                   'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 3.01, 1: 1.679, 2: 1.638, 5: 1.512},
                   'ADMET_regression_all_two_shot_5epoch': {0: 7.058, 1: 1.6, 2: 1.624, 5: 1.646},
                   'ADMET_regression_all_mixed_shot_10epoch': {0: 2.379, 1: 1.837, 2: 1.67, 5: 1.691},
                   'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 6.733, 1: 1.687, 2: 1.562, 5: 1.449},
                   'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 6.652, 1: 1.763, 2: 1.668, 5: 1.524}},
               'ppbr_az': {'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 26.786, 1: 22.965, 2: 16.861, 5: 10.743},
                           'ADMET_regression_all_five_shot_5epoch': {0: 34.228, 1: 13.027, 2: 11.947, 5: 11.904},
                           'Llama3': {0: 39.232, 1: 12.733, 2: 12.091, 5: 12.375},
                           'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 26.012, 1: 22.357, 2: 17.303, 5: 10.753},
                           'ADMET_regression_all_zero_shot_10epoch': {0: 20.52, 1: 25.558, 2: 15.621, 5: 11.791},
                           'ADMET_regression_all_one_shot_5epoch': {0: 19.818, 1: 14.362, 2: 12.207, 5: 10.746},
                           'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 14.337, 1: 17.122, 2: 12.977, 5: 11.147},
                           'ADMET_regression_all_two_shot_5epoch': {0: 38.826, 1: 13.973, 2: 12.505, 5: 10.8},
                           'ADMET_regression_all_mixed_shot_10epoch': {0: 20.547, 1: 25.422, 2: 15.445, 5: 11.758},
                           'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 22.085, 1: 17.199, 2: 15.566, 5: 14.199},
                           'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 28.398, 1: 16.805, 2: 14.586, 5: 12.208}},
               'vdss_lombardo': {
                   'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 0.265, 1: -0.021, 2: 0.164, 5: 0.474},
                   'ADMET_regression_all_five_shot_5epoch': {0: 0.171, 1: 0.511, 2: 0.53, 5: 0.568},
                   'Llama3': {0: 0.022, 1: 0.492, 2: 0.461, 5: 0.482},
                   'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 0.303, 1: -0.046, 2: 0.149, 5: 0.451},
                   'ADMET_regression_all_zero_shot_10epoch': {0: 0.153, 1: 0.154, 2: 0.458, 5: 0.526},
                   'ADMET_regression_all_one_shot_5epoch': {0: 0.038, 1: 0.506, 2: 0.502, 5: 0.56},
                   'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 0.183, 1: 0.474, 2: 0.487, 5: 0.48},
                   'ADMET_regression_all_two_shot_5epoch': {0: 0.237, 1: 0.506, 2: 0.5, 5: 0.558},
                   'ADMET_regression_all_mixed_shot_10epoch': {0: 0.153, 1: 0.143, 2: 0.456, 5: 0.521},
                   'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 0.027, 1: 0.433, 2: 0.483, 5: 0.501},
                   'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: -0.041, 1: 0.402, 2: 0.417, 5: 0.445}},
               'half_life_obach': {
                   'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 0.166, 1: -0.279, 2: -0.294, 5: -0.091},
                   'ADMET_regression_all_five_shot_5epoch': {0: 0.104, 1: 0.481, 2: 0.49, 5: 0.482},
                   'Llama3': {0: 0.0, 1: 0.295, 2: 0.494, 5: 0.471},
                   'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 0.128, 1: -0.248, 2: -0.311, 5: -0.157},
                   'ADMET_regression_all_zero_shot_10epoch': {0: -0.022, 1: -0.174, 2: -0.294, 5: -0.016},
                   'ADMET_regression_all_one_shot_5epoch': {0: 0.152, 1: 0.393, 2: 0.447, 5: 0.472},
                   'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: -0.086, 1: 0.455, 2: 0.417, 5: 0.372},
                   'ADMET_regression_all_two_shot_5epoch': {0: 0.16, 1: 0.485, 2: 0.477, 5: 0.498},
                   'ADMET_regression_all_mixed_shot_10epoch': {0: -0.054, 1: -0.162, 2: -0.306, 5: -0.028},
                   'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: -0.01, 1: 0.451, 2: 0.473, 5: 0.463},
                   'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: -0.027, 1: 0.498, 2: 0.481, 5: 0.412}},
               'clearance_microsome_az': {
                   'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 0.101, 1: -0.085, 2: -0.043, 5: 0.028},
                   'ADMET_regression_all_five_shot_5epoch': {0: 0.036, 1: 0.471, 2: 0.445, 5: 0.505},
                   'Llama3': {0: 0.035, 1: 0.442, 2: 0.434, 5: 0.395},
                   'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 0.128, 1: 0.024, 2: -0.029, 5: -0.021},
                   'ADMET_regression_all_zero_shot_10epoch': {0: 0.058, 1: -0.027, 2: 0.018, 5: 0.289},
                   'ADMET_regression_all_one_shot_5epoch': {0: -0.054, 1: 0.472, 2: 0.467, 5: 0.485},
                   'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 0.095, 1: 0.468, 2: 0.475, 5: 0.434},
                   'ADMET_regression_all_two_shot_5epoch': {0: 0.071, 1: 0.463, 2: 0.46, 5: 0.463},
                   'ADMET_regression_all_mixed_shot_10epoch': {0: 0.056, 1: -0.025, 2: 0.014, 5: 0.286},
                   'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 0.047, 1: 0.449, 2: 0.385, 5: 0.406},
                   'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 0.08, 1: 0.474, 2: 0.485, 5: 0.409}},
               'clearance_hepatocyte_az': {
                   'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: -0.046, 1: -0.139, 2: -0.058, 5: 0.009},
                   'ADMET_regression_all_five_shot_5epoch': {0: -0.057, 1: 0.282, 2: 0.281, 5: 0.325},
                   'Llama3': {0: -0.001, 1: 0.289, 2: 0.304, 5: 0.296},
                   'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 0.028, 1: -0.08, 2: -0.076, 5: 0.032},
                   'ADMET_regression_all_zero_shot_10epoch': {0: 0.051, 1: 0.096, 2: 0.141, 5: 0.19},
                   'ADMET_regression_all_one_shot_5epoch': {0: -0.039, 1: 0.273, 2: 0.271, 5: 0.283},
                   'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: -0.106, 1: 0.331, 2: 0.31, 5: 0.255},
                   'ADMET_regression_all_two_shot_5epoch': {0: -0.102, 1: 0.278, 2: 0.274, 5: 0.344},
                   'ADMET_regression_all_mixed_shot_10epoch': {0: 0.062, 1: 0.11, 2: 0.138, 5: 0.183},
                   'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: -0.049, 1: 0.283, 2: 0.291, 5: 0.281},
                   'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 0.024, 1: 0.29, 2: 0.319, 5: 0.225}},
               'ld50_zhu': {'ADMET_reg_cls_all_mixed_shot_checkpoint-500': {0: 1.203, 1: 0.91, 2: 0.915, 5: 0.937},
                            'ADMET_regression_all_five_shot_5epoch': {0: 2.896, 1: 1.215, 2: 1.212, 5: 1.314},
                            'Llama3': {0: 2.27, 1: 1.389, 2: 1.341, 5: 1.33},
                            'ADMET_reg_cls_all_zero_shot_checkpoint-500': {0: 1.293, 1: 0.912, 2: 0.915, 5: 0.937},
                            'ADMET_regression_all_zero_shot_10epoch': {0: 1.463, 1: 0.937, 2: 0.942, 5: 1.004},
                            'ADMET_regression_all_one_shot_5epoch': {0: 2.117, 1: 1.116, 2: 1.148, 5: 1.179},
                            'ADMET_reg_cls_all_five_shot_checkpoint-400': {0: 2.638, 1: 1.143, 2: 1.107, 5: 1.213},
                            'ADMET_regression_all_two_shot_5epoch': {0: 3.104, 1: 1.204, 2: 1.194, 5: 1.223},
                            'ADMET_regression_all_mixed_shot_10epoch': {0: 1.46, 1: 0.937, 2: 0.944, 5: 1.006},
                            'ADMET_reg_cls_all_one_shot_checkpoint-400': {0: 1.447, 1: 1.118, 2: 1.147, 5: 1.294},
                            'ADMET_reg_cls_all_two_shot_checkpoint-400': {0: 2.135, 1: 1.15, 2: 1.147, 5: 1.255}}}

"""
template_result_dict = {
    "caco2_wang": {"llama3": {1: 0.845}, "llama3_all_reg_5_epoch": {1: 0.573}, "llama3_all_reg_10_epoch": {1: 0.497}}
}
"""
# %%
""" 在这里算排名！"""
regression_benchmark = {
    "caco2_wang": {
        "value": [0.276, 0.285, 0.287, 0.287, 0.289, 0.297, 0.321, 0.330, 0.335, 0.341, 0.393, 0.401, 0.446, 0.502,
                  0.530, 0.546, 0.599, 0.908], "metric": "MAE"},
    "lipophilicity_astrazeneca": {
        "value": [0.467, 0.470, 0.479, 0.479, 0.515, 0.525, 0.535, 0.538, 0.539, 0.541, 0.547, 0.563, 0.572, 0.574,
                  0.617, 0.621, 0.626, 0.656, 0.701, 0.743],
        "metric": "MAE",
    },
    "solubility_aqsoldb": {
        "value": [0.761, 0.771, 0.775, 0.776, 0.789, 0.792, 0.796, 0.827, 0.828, 0.829, 0.907, 0.939, 0.947, 1.023,
                  1.026, 1.040, 1.076, 1.203],
        "metric": "MAE",
    },
    "ppbr_az": {
        "value": [7.526, 7.660, 7.788, 7.914, 7.99, 8.288, 8.582, 8.680, 9.185, 9.292, 9.373, 9.445, 9.942, 9.994,
                  10.075, 10.194, 11.106, 12.848],
        "metric": "MAE",
    }, "vdss_lombardo": {
        "value": [0.713, 0.707, 0.707, 0.628, 0.627, 0.609, 0.582, 0.561, 0.559, 0.497, 0.493, 0.491, 0.485, 0.457,
                  0.389, 0.258, 0.241, 0.226],
        "metric": "Spearman",
    },
    "half_life_obach": {
        "value": [0.576, 0.562, 0.557, 0.547, 0.544, 0.511, 0.485, 0.438, 0.392, 0.329, 0.265, 0.239, 0.239, 0.184,
                  0.177, 0.151, 0.129, 0.085, 0.038],
        "metric": "Spearman",
    }, "clearance_microsome_az": {
        "value": [0.630, 0.626, 0.625, 0.625, 0.599, 0.597, 0.586, 0.585, 0.578, 0.572, 0.555, 0.533, 0.532, 0.529,
                  0.518, 0.492, 0.365, 0.252],
        "metric": "Spearman",
    }, "clearance_hepatocyte_az": {
        "value": [0.536, 0.498, 0.466, 0.457, 0.440, 0.440, 0.439, 0.431, 0.430, 0.424, 0.413, 0.401, 0.382, 0.366,
                  0.289, 0.272, 0.235],
        "metric": "Spearman",
    }, "ld50_zhu": {
        "value": [0.552, 0.588, 0.606, 0.614, 0.621, 0.622, 0.625, 0.630, 0.631, 0.633, 0.636, 0.646, 0.649, 0.649,
                  0.667, 0.669, 0.675, 0.678, 0.678, 0.685],
        "metric": "MAE",
    }
}


def get_rank(dataset_name, new_res, greater_better=False):
    if greater_better:
        count = sum(1 for d in regression_benchmark[dataset_name]["value"] if d > new_res)
    else:
        count = sum(1 for d in regression_benchmark[dataset_name]["value"] if d < new_res)
    print(f"当前新数据排名：{count + 1}")
    return count + 1


# %%
spearman_tasks = ["vdss_lombardo", "half_life_obach", "clearance_microsome_az", "clearance_hepatocyte_az"]

ranking_dict = {}
for dataset, results_all in result_dict.items():
    ranking_dict[dataset] = {}
    if dataset in spearman_tasks:
        greater_better = True
    else:
        greater_better = False

    model_name = "ADMET_reg_cls_all_mixed_shot_checkpoint-500"#"ADMET_regression_all_mixed_shot_10epoch"#"ADMET_regression_all_two_shot_5epoch" #"ADMET_regression_all_one_shot_5epoch"  # "Llama3"
    result = results_all[model_name]
    for k in [0, 1, 2, 5]:
        ranking_dict[dataset][k] = get_rank(dataset, result[k], greater_better)
print(ranking_dict)

# %%
spearman_tasks = ["vdss_lombardo", "half_life_obach", "clearance_microsome_az", "clearance_hepatocyte_az"]

ranking_dict = {}
for dataset, results_all in result_dict.items():
    if dataset in spearman_tasks:
        greater_better = True
    else:
        greater_better = False
    ranking_dict[dataset] = {}
    for model_name, results in results_all.items():
        if model_name not in ranking_dict[dataset]:
            ranking_dict[dataset] = {}
        for k, v in results.items():
            ranking_dict[dataset][k] = get_rank(dataset, v, greater_better)

# %%
""" 在这里可视化！ """
import seaborn as sns
import matplotlib.pyplot as plt

available_markers = ['o', 's', '^', 'v', '>', '<', 'p', '*', 'h', '+', 'o']
available_colors = ["black", "green", "red", "orange", "blue", "magenta", "cyan", "purple", "yellow", "pink", "brown"]
regression_task = ["caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az", "vdss_lombardo",
                   "half_life_obach", "clearance_microsome_az", "clearance_hepatocyte_az", "ld50_zhu"]

# TODO: UPDATE
model_list = ['ADMET_reg_cls_all_mixed_shot_checkpoint-500', 'ADMET_regression_all_five_shot_5epoch', 'Llama3',
              'ADMET_reg_cls_all_zero_shot_checkpoint-500', 'ADMET_regression_all_zero_shot_10epoch',
              'ADMET_regression_all_one_shot_5epoch', 'ADMET_reg_cls_all_five_shot_checkpoint-400',
              'ADMET_regression_all_two_shot_5epoch', 'ADMET_regression_all_mixed_shot_10epoch',
              'ADMET_reg_cls_all_one_shot_checkpoint-400', 'ADMET_reg_cls_all_two_shot_checkpoint-400']

task_list = [0, 1, 2, 5]

# color_map = {model_list[i]: available_colors[i] for i in range(len(model_list))}
# marker_map = {task_list[i]: available_markers[i] for i in range(len(task_list))}
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

# %%

for dataset in regression_task:
    cur_result = result_dict[dataset]
    labels = {}
    plt.figure(figsize=(12, 3))
    for i in range(len(task_list)):
        labels[i] = f"{task_list[i]}-shot"
        for model in model_list:
            sns.scatterplot(x=[cur_result[model][task_list[i]]], y=[i], color=color_map[model],
                            marker=marker_map[model], alpha=0.5)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.yticks(ticks=range(len(df['y'])), labels=df['y'])
    plt.title(dataset)
    plt.tight_layout()
    plt.show()
