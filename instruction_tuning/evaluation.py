# %%
""" get result_dict """

import sys
import json
from tdc.benchmark_group import admet_group
from instruction_tuning.instruction_prompt import CLASSIFICATION_TASKS, REGRESSION_TASKS
from benchmark.utils import get_min_max_value, denormalize
from instruction_tuning.benchmark_record import *

sys.path.append("..")
task_record = {"5-shot": 5}  # {"ruleof5": 5}  #
result_folder = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/result/cls_v1"
models_to_evaluate = ['ADMET_reg_clsv1_all_five_shot_checkpoint-500']  # ["ADMET_cls_five_shot_checkpoint-400","ADMET_regression_all_five_shot_5epoch","ADMET_reg_cls_all_five_shot_checkpoint-400"]

task_name_dict = {"caco2_wang": "Caco2_Wang", "lipophilicity_astrazeneca": "Lipophilicity_AstraZeneca",
                  "solubility_aqsoldb": "Solubility_AqSolDB", "ppbr_az": "PPBR_AZ", "vdss_lombardo": "VDss_Lombardo",
                  "half_life_obach": "Half_Life_Obach", "clearance_microsome_az": "Clearance_Microsome_AZ",
                  "clearance_hepatocyte_az": "Clearance_Hepatocyte_AZ", "ld50_zhu": "LD50_Zhu"}

result_dict = {}
group = admet_group(path="data/")
for task_folder, k in task_record.items():
    for dataset in CLASSIFICATION_TASKS:
        print(f"processing {dataset} results")
        if dataset not in result_dict:
            result_dict[dataset] = {}
        benchmark = group.get(dataset)
        predictions = {}
        name = benchmark["name"]
        train_val, test = benchmark["train_val"], benchmark["test"]
        min_value, max_value = get_min_max_value(train_val, "Y")
        for model in models_to_evaluate:
            file_name = f"{dataset}_instruction-{k}-shot-test-stepbystep.json"
            if model not in result_dict[dataset]:
                result_dict[dataset][model] = {}
            with open(f"{result_folder}/{task_folder}/{model}/{file_name}", "r") as f:
                result = json.load(f)
            predictions[name] = [denormalize(float(f), min_value, max_value) for f in result["pred_values"]]
            result_dict[dataset][model][k] = list(group.evaluate(predictions)[name].values())[0]
# %%
def get_rank(benchmark, dataset_name, new_res, greater_better=False):
    if greater_better:
        count = sum(1 for d in benchmark[dataset_name]["value"] if d > new_res)
    else:
        count = sum(1 for d in benchmark[dataset_name]["value"] if d < new_res)
    print(f"当前新数据排名：{count + 1}")
    return count + 1


get_rank(TDC_ADMET_BENCHMARK_REG, "vdss_lombardo", 0.472, greater_better=True)
get_rank(TDC_ADMET_BENCHMARK_CLS, "dili", 0.521, greater_better=True)
