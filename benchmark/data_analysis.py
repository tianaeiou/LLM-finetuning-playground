# %%
import json
import os
import sys
import pickle
from tdc.single_pred import ADME
from tdc.benchmark_group import admet_group
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski

from benchmark.TDC_BenchmarkGroup_prompt_generation import *

project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"

group = admet_group(path='data/')
config_dict = {
    "caco2_wang": {
        "dataset": "Caco2_Wang",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CACO_TEMPLATE
    },
    "hia_hou": {
        "dataset": "HIA_Hou",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HIA_HOU_TEMPLATE
    },
    "pgp_broccatelli": {
        "dataset": "Pgp_Broccatelli",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PGP_TEMPLATE
    },
    "bioavailability_ma": {
        "dataset": "Bioavailability_Ma",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": MA_TEMPLATE
    },
    "lipophilicity_astrazeneca": {
        "dataset": "Lipophilicity_AstraZeneca",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LIPO_TEMPLATE
    },
    "solubility_aqsoldb": {
        "dataset": "Solubility_AqSolDB",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Solubility_TEMPLATE
    },
    "bbb_martins": {
        "dataset": "BBB_Martins",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": BBB_TEMPLATE
    },
    "ppbr_az": {
        "dataset": "PPBR_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PPBR_TEMPLATE
    },
    "vdss_lombardo": {
        "dataset": "VDss_Lombardo",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": VDSS_TEMPLATE
    },
    "cyp2d6_veith": {
        "dataset": "CYP2D6_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_TEMPLATE
    },
    "cyp3a4_veith": {
        "dataset": "CYP3A4_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_TEMPLATE
    },
    "cyp2c9_veith": {
        "dataset": "CYP2C9_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_Veith_TEMPLATE
    },
    "cyp2d6_substrate_carbonmangels": {
        "dataset": "CYP2D6_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_SUBSTRATE_TEMPLATE
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "CYP3A4_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_SUBSTRATE_TEMPLATE
    },
    "cyp2c9_substrate_carbonmangels": {
        "dataset": "CYP2C9_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_SUBSTRATE_SUBSTRATE_TEMPLATE
    },
    "half_life_obach": {
        "dataset": "Half_Life_Obach",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HL_TEMPLATE
    },
    "clearance_microsome_az": {
        "dataset": "Clearance_Microsome_AZ",
        "task": "regression",
        "example_strategy": "quantile",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Microsome_AZ_TEMPLATE
    },
    "clearance_hepatocyte_az": {
        "dataset": "Clearance_Hepatocyte_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Hepatocyte_AZ_TEMPLATE
    },
    "herg": {
        "dataset": "hERG",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": hERG_TEMPLATE
    },
    "ames": {
        "dataset": "AMES",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": AMES_TEMPLATE
    },
    "dili": {
        "dataset": "DILI",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": DILI_TEMPLATE
    },
    "ld50_zhu": {
        "dataset": "LD50_Zhu",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LD50_TEMPLATE
    },
}
print("HELLO  WORLD")
counter = 0
for key, config in config_dict.items():
    if config["task"] == "regression":
        for k in [2]:
            benchmark = group.get(config["dataset"])
            predictions = {}
            name = benchmark['name']
            train_val_df, test_df = benchmark['train_val'], benchmark['test']
            data_df = test_df
            ref_df = train_val_df

            min_value, max_value = get_min_max_value(ref_df, "Y")
            file_dir = f"{project_path}/data/Caco2_benchmark/q1q3_{config["dataset"]}-smiles-neighbors_{config["split"]}.pkl"

            with open(file_dir, 'rb') as file:
                neighbor_dict = pickle.load(file)
            neighbor_index_dict = neighbor_dict["neighbor_index"]

            record_dict = {}
            for idx, neighbors in neighbor_index_dict.items():
                record_dict[idx] = {"target": [normalize(test_df["Y"].to_list()[idx], min_value, max_value)],
                                    "neighbors": [normalize(ref_df["Y"].to_list()[i], min_value, max_value) for i in
                                                  [neighbors[i][0] for i in range(k)]]}

            neighbors_diff = []

            for item in record_dict.values():
                target = item['target'][0]
                neighbors = item['neighbors']
                diff = [abs(neighbor - target) for neighbor in neighbors]
                neighbors_diff.extend(diff)

            # 可视化差异的分布
            plt.figure(figsize=(10, 6))

            # 使用直方图
            # plt.hist(neighbors_diff, bins=20, color='skyblue', alpha=0.7)
            # plt.title(f"{config["dataset"]}_Histogram of Neighbors-Target Difference_{k}-shot")
            # plt.ylabel("# samples")
            # plt.xlabel("average |neighbor value - target value|")
            # # 使用箱型图
            # plt.subplot(1, 2, 2)  # 1行2列的第2个图
            sns.boxplot(x=neighbors_diff)
            plt.title(f"Boxplot of Neighbors-Target Difference {config["dataset"]} {k}-shot")
            plt.xlim(0.0, 1000.0)
            # 显示图表
            plt.tight_layout()  # 调整子图布局以适应标签和标题
            plt.show()
# %%
import json
import os
import sys
import pickle
from tdc.single_pred import ADME
from tdc.benchmark_group import admet_group
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski

from benchmark.TDC_BenchmarkGroup_prompt_generation import *

project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"

group = admet_group(path='data/')
config_dict = {
    "caco2_wang": {
        "dataset": "Caco2_Wang",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CACO_TEMPLATE
    },
    "hia_hou": {
        "dataset": "HIA_Hou",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HIA_HOU_TEMPLATE
    },
    "pgp_broccatelli": {
        "dataset": "Pgp_Broccatelli",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PGP_TEMPLATE
    },
    "bioavailability_ma": {
        "dataset": "Bioavailability_Ma",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": MA_TEMPLATE
    },
    "lipophilicity_astrazeneca": {
        "dataset": "Lipophilicity_AstraZeneca",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LIPO_TEMPLATE
    },
    "solubility_aqsoldb": {
        "dataset": "Solubility_AqSolDB",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Solubility_TEMPLATE
    },
    "bbb_martins": {
        "dataset": "BBB_Martins",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": BBB_TEMPLATE
    },
    "ppbr_az": {
        "dataset": "PPBR_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PPBR_TEMPLATE
    },
    "vdss_lombardo": {
        "dataset": "VDss_Lombardo",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": VDSS_TEMPLATE
    },
    "cyp2d6_veith": {
        "dataset": "CYP2D6_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_TEMPLATE
    },
    "cyp3a4_veith": {
        "dataset": "CYP3A4_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_TEMPLATE
    },
    "cyp2c9_veith": {
        "dataset": "CYP2C9_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_Veith_TEMPLATE
    },
    "cyp2d6_substrate_carbonmangels": {
        "dataset": "CYP2D6_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_SUBSTRATE_TEMPLATE
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "CYP3A4_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_SUBSTRATE_TEMPLATE
    },
    "cyp2c9_substrate_carbonmangels": {
        "dataset": "CYP2C9_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_SUBSTRATE_SUBSTRATE_TEMPLATE
    },
    "half_life_obach": {
        "dataset": "Half_Life_Obach",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HL_TEMPLATE
    },
    "clearance_microsome_az": {
        "dataset": "Clearance_Microsome_AZ",
        "task": "regression",
        "example_strategy": "quantile",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Microsome_AZ_TEMPLATE
    },
    "clearance_hepatocyte_az": {
        "dataset": "Clearance_Hepatocyte_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Hepatocyte_AZ_TEMPLATE
    },
    "herg": {
        "dataset": "hERG",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": hERG_TEMPLATE
    },
    "ames": {
        "dataset": "AMES",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": AMES_TEMPLATE
    },
    "dili": {
        "dataset": "DILI",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": DILI_TEMPLATE
    },
    "ld50_zhu": {
        "dataset": "LD50_Zhu",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LD50_TEMPLATE
    },
}
print("HELLO  WORLD")
counter = 0
fig, ax = plt.subplots(3, 3, figsize=(12, 8))
for key, config in config_dict.items():
    if config["task"] == "regression":
        for k in [2]:
            benchmark = group.get(config["dataset"])
            predictions = {}
            name = benchmark['name']
            train_val_df, test_df = benchmark['train_val'], benchmark['test']
            data_df = test_df
            ref_df = train_val_df

            min_value, max_value = get_min_max_value(ref_df, "Y")
            file_dir = f"{project_path}/data/Caco2_benchmark/{config["dataset"]}-smiles-neighbors_{config["split"]}.pkl"

            with open(file_dir, 'rb') as file:
                neighbor_dict = pickle.load(file)
            neighbor_index_dict = neighbor_dict["neighbor_index"]

            record_dict = {}
            for idx, neighbors in neighbor_index_dict.items():
                record_dict[idx] = {
                    "target": [normalize(test_df["Y"].to_list()[idx], min_value, max_value)],
                    "neighbors": [normalize(ref_df["Y"].to_list()[i], min_value, max_value) for i in
                                  [neighbors[i][0] for i in range(k)]]}

            neighbors_diff = []

            for item in record_dict.values():
                target = item['target'][0]
                neighbors = item['neighbors']
                diff = [abs(neighbor - target) for neighbor in neighbors]
                neighbors_diff.extend(diff)

            # 可视化差异的分

            # 使用直方图
            # plt.hist(neighbors_diff, bins=20, color='skyblue', alpha=0.7)
            # plt.title(f"{config["dataset"]}_Histogram of Neighbors-Target Difference_{k}-shot")
            # plt.ylabel("# samples")
            # plt.xlabel("average |neighbor value - target value|")
            # # 使用箱型图
            # plt.subplot(1, 2, 2)  # 1行2列的第2个图
            sns.boxplot(x=neighbors_diff, ax=ax[counter // 3, counter % 3])
            ax[counter // 3, counter % 3].set_title(f"{config["dataset"]}")
            ax[counter // 3, counter % 3].set_xlim(0.0, 1000.0)
            # 显示图表
            counter += 1
plt.tight_layout()  # 调整子图布局以适应标签和标题
plt.show()
# %%
# 看分类任务的分布
import os
import sys
import pickle
import json
from tdc.single_pred import ADME
import numpy as np
from tdc.benchmark_group import admet_group
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark.prompt_template import *
from benchmark.utils import *
import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")
fig, ax = plt.subplots(4, 4, figsize=(10, 12))
counter = 0

for dataset_name, config in config_dict.items():
    if config["task"] != "regression":
        group = admet_group(path='data/')
        benchmark = group.get(dataset_name)
        predictions = {}
        name = benchmark['name']
        train_val_df, test_df = benchmark['train_val'], benchmark['test']

        value_counts = test_df['Y'].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values / len(test_df), ax=ax[counter // 4, counter % 4])
        ax[counter // 4, counter % 4].set_title(f'{dataset_name}')
        ax[counter // 4, counter % 4].set_xlabel("")
        ax[counter // 4, counter % 4].set_ylim(0.0, 1.0)
        counter += 1
plt.tight_layout()
plt.show()
# %%
import json
import os
import sys
import pickle
from tdc.single_pred import ADME
from tdc.benchmark_group import admet_group
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski

from benchmark.TDC_BenchmarkGroup_prompt_generation import *

project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"

group = admet_group(path='data/')
config_dict = {
    "caco2_wang": {
        "dataset": "Caco2_Wang",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CACO_TEMPLATE
    },
    "hia_hou": {
        "dataset": "HIA_Hou",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HIA_HOU_TEMPLATE
    },
    "pgp_broccatelli": {
        "dataset": "Pgp_Broccatelli",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PGP_TEMPLATE
    },
    "bioavailability_ma": {
        "dataset": "Bioavailability_Ma",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": MA_TEMPLATE
    },
    "lipophilicity_astrazeneca": {
        "dataset": "Lipophilicity_AstraZeneca",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LIPO_TEMPLATE
    },
    "solubility_aqsoldb": {
        "dataset": "Solubility_AqSolDB",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Solubility_TEMPLATE
    },
    "bbb_martins": {
        "dataset": "BBB_Martins",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": BBB_TEMPLATE
    },
    "ppbr_az": {
        "dataset": "PPBR_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PPBR_TEMPLATE
    },
    "vdss_lombardo": {
        "dataset": "VDss_Lombardo",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": VDSS_TEMPLATE
    },
    "cyp2d6_veith": {
        "dataset": "CYP2D6_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_TEMPLATE
    },
    "cyp3a4_veith": {
        "dataset": "CYP3A4_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_TEMPLATE
    },
    "cyp2c9_veith": {
        "dataset": "CYP2C9_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_Veith_TEMPLATE
    },
    "cyp2d6_substrate_carbonmangels": {
        "dataset": "CYP2D6_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_SUBSTRATE_TEMPLATE
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "CYP3A4_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_SUBSTRATE_TEMPLATE
    },
    "cyp2c9_substrate_carbonmangels": {
        "dataset": "CYP2C9_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_SUBSTRATE_SUBSTRATE_TEMPLATE
    },
    "half_life_obach": {
        "dataset": "Half_Life_Obach",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HL_TEMPLATE
    },
    "clearance_microsome_az": {
        "dataset": "Clearance_Microsome_AZ",
        "task": "regression",
        "example_strategy": "quantile",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Microsome_AZ_TEMPLATE
    },
    "clearance_hepatocyte_az": {
        "dataset": "Clearance_Hepatocyte_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Hepatocyte_AZ_TEMPLATE
    },
    "herg": {
        "dataset": "hERG",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": hERG_TEMPLATE
    },
    "ames": {
        "dataset": "AMES",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": AMES_TEMPLATE
    },
    "dili": {
        "dataset": "DILI",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": DILI_TEMPLATE
    },
    "ld50_zhu": {
        "dataset": "LD50_Zhu",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LD50_TEMPLATE
    },
}
print("HELLO  WORLD")
counter = 0
fig, ax = plt.subplots(3, 5, figsize=(15, 8))
for key, config in config_dict.items():
    if config["task"] != "regression":
        for k in [2]:
            benchmark = group.get(config["dataset"])
            predictions = {}
            name = benchmark['name']
            train_val_df, test_df = benchmark['train_val'], benchmark['test']
            data_df = test_df
            ref_df = train_val_df

            min_value, max_value = get_min_max_value(ref_df, "Y")
            file_dir = f"{project_path}/data/Caco2_benchmark/cls_{config["dataset"]}-smiles-neighbors_{config["split"]}_cls.pkl"

            with open(file_dir, 'rb') as file:
                neighbor_dict = pickle.load(file)
            neighbor_index_dict = neighbor_dict["neighbor_index"]

            record_dict = {}
            for idx, neighbors in neighbor_index_dict.items():
                record_dict[idx] = {
                    "target": [test_df["Y"].to_list()[idx]],
                    "neighbors": [ref_df["Y"].to_list()[neighbors[0][0]], ref_df["Y"].to_list()[neighbors[5][0]]]}

            # 初始化两个列表来存储不同target值的记录
            record_for_0 = []
            record_for_1 = []

            # 遍历字典并根据target值分配记录
            for idx, record in record_dict.items():
                if record["target"][0] == 0:
                    record_for_0.append(sum(record["neighbors"]))
                elif record["target"][0] == 1:
                    record_for_1.append(sum(record["neighbors"]))


            # 计算每个列表中neighbors的取值占比
            def calculate_proportions(records):
                from collections import Counter
                # 将所有neighbors列表展平成一个列表
                all_neighbors = [item for sublist in records for item in sublist]
                # 使用Counter进行计数
                neighbor_counts = Counter(all_neighbors)
                # 计算占比
                proportions = {k: (v / len(all_neighbors)) * 100 for k, v in neighbor_counts.items()}
                return proportions


            # 获取0和1的neighbors占比
            proportions_for_0 = calculate_proportions(record_for_0)
            proportions_for_1 = calculate_proportions(record_for_1)

            # 将字典转换为用于绘图的列表
            categories = [0, 1]  # categories = list(proportions_for_0.keys())
            proportions_0 = list(proportions_for_0.values())
            proportions_1 = list(proportions_for_1.values())

            # 设置颜色
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            # 创建图表和轴对象
            fig, ax = plt.subplots()

            # 绘制分组柱状图
            bar_width = 0.35
            index = range(len(categories))

            bar1 = plt.bar(index, proportions_0, bar_width, label='Target=0', color=colors)
            bar2 = plt.bar([i + bar_width for i in index], proportions_1, bar_width, label='Target=1',
                           color=[colors[i] * 0.7 for i in range(len(colors))])

            # 添加图例和标题
            plt.legend()
            plt.title('Proportions of Neighbors for Different Target Values')

            # 设置坐标轴标签和刻度
            plt.xlabel('Neighbor Categories')
            plt.xticks([r + bar_width / 2 for r in range(len(categories))], categories)
            plt.ylabel('Proportion (%)')

            # 显示图表
            plt.tight_layout()
            plt.show()
# plt.tight_layout()  # 调整子图布局以适应标签和标题
# plt.show()
# %%
import json
import os
import sys
import pickle
from tdc.single_pred import ADME
from tdc.benchmark_group import admet_group
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski

from benchmark.TDC_BenchmarkGroup_prompt_generation import *

project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"

group = admet_group(path='data/')
config_dict = {
    "caco2_wang": {
        "dataset": "Caco2_Wang",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CACO_TEMPLATE
    },
    "hia_hou": {
        "dataset": "HIA_Hou",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HIA_HOU_TEMPLATE
    },
    "pgp_broccatelli": {
        "dataset": "Pgp_Broccatelli",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PGP_TEMPLATE
    },
    "bioavailability_ma": {
        "dataset": "Bioavailability_Ma",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": MA_TEMPLATE
    },
    "lipophilicity_astrazeneca": {
        "dataset": "Lipophilicity_AstraZeneca",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LIPO_TEMPLATE
    },
    "solubility_aqsoldb": {
        "dataset": "Solubility_AqSolDB",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Solubility_TEMPLATE
    },
    "bbb_martins": {
        "dataset": "BBB_Martins",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": BBB_TEMPLATE
    },
    "ppbr_az": {
        "dataset": "PPBR_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PPBR_TEMPLATE
    },
    "vdss_lombardo": {
        "dataset": "VDss_Lombardo",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": VDSS_TEMPLATE
    },
    "cyp2d6_veith": {
        "dataset": "CYP2D6_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_TEMPLATE
    },
    "cyp3a4_veith": {
        "dataset": "CYP3A4_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_TEMPLATE
    },
    "cyp2c9_veith": {
        "dataset": "CYP2C9_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_Veith_TEMPLATE
    },
    "cyp2d6_substrate_carbonmangels": {
        "dataset": "CYP2D6_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_SUBSTRATE_TEMPLATE
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "CYP3A4_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_SUBSTRATE_TEMPLATE
    },
    "cyp2c9_substrate_carbonmangels": {
        "dataset": "CYP2C9_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_SUBSTRATE_SUBSTRATE_TEMPLATE
    },
    "half_life_obach": {
        "dataset": "Half_Life_Obach",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HL_TEMPLATE
    },
    "clearance_microsome_az": {
        "dataset": "Clearance_Microsome_AZ",
        "task": "regression",
        "example_strategy": "quantile",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Microsome_AZ_TEMPLATE
    },
    "clearance_hepatocyte_az": {
        "dataset": "Clearance_Hepatocyte_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Hepatocyte_AZ_TEMPLATE
    },
    "herg": {
        "dataset": "hERG",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": hERG_TEMPLATE
    },
    "ames": {
        "dataset": "AMES",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": AMES_TEMPLATE
    },
    "dili": {
        "dataset": "DILI",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": DILI_TEMPLATE
    },
    "ld50_zhu": {
        "dataset": "LD50_Zhu",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [2, 5],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LD50_TEMPLATE
    },
}
print("HELLO  WORLD")
counter = 0
fig, ax = plt.subplots(3, 5, figsize=(15, 8))
for key, config in config_dict.items():
    if config["task"] != "regression":
        for k in [2]:
            benchmark = group.get(config["dataset"])
            predictions = {}
            name = benchmark['name']
            train_val_df, test_df = benchmark['train_val'], benchmark['test']
            data_df = test_df
            ref_df = train_val_df

            min_value, max_value = get_min_max_value(ref_df, "Y")
            file_dir = f"{project_path}/data/Caco2_benchmark/cls_{config["dataset"]}-smiles-neighbors_{config["split"]}_cls.pkl"

            with open(file_dir, 'rb') as file:
                neighbor_dict = pickle.load(file)
            neighbor_index_dict = neighbor_dict["neighbor_index"]

            record_dict = {}
            for idx, neighbors in neighbor_index_dict.items():
                record_dict[idx] = {
                    "target": [test_df["Y"].to_list()[idx]],
                    "neighbors": [ref_df["Y"].to_list()[0], ref_df["Y"].to_list()[5]]}

            neighbors_diff = []

            for item in record_dict.values():
                target = item['target'][0]
                neighbors = item['neighbors']
                diff = [abs(neighbor - target) for neighbor in neighbors]
                neighbors_diff.extend(diff)

            # 可视化差异的分

            # 使用直方图
            # plt.hist(neighbors_diff, bins=20, color='skyblue', alpha=0.7)
            # plt.title(f"{config["dataset"]}_Histogram of Neighbors-Target Difference_{k}-shot")
            # plt.ylabel("# samples")
            # plt.xlabel("average |neighbor value - target value|")
            # # 使用箱型图
            # plt.subplot(1, 2, 2)  # 1行2列的第2个图
            sns.boxplot(x=neighbors_diff, ax=ax[counter // 5, counter % 5])
            ax[counter // 5, counter % 5].set_title(f"{config["dataset"]}")

            # 显示图表
            counter += 1
plt.tight_layout()  # 调整子图布局以适应标签和标题
plt.show()
# %%
# 似乎VDSS 的分布不太对 -> 看所有regression task上的分布！

import os
import sys
import pickle
import json
from tdc.single_pred import ADME
import numpy as np
from tdc.benchmark_group import admet_group
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark.prompt_template import *
from benchmark.utils import *
import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")
fig, ax = plt.subplots(3, 3, figsize=(12, 8))
counter = 0

for dataset_name, config in config_dict.items():
    if config["task"] == "regression":
        group = admet_group(path='data/')
        benchmark = group.get(dataset_name)
        predictions = {}
        name = benchmark['name']
        train_val_df, test_df = benchmark['train_val'], benchmark['test']

        sns.boxplot(x=test_df['Y'], ax=ax[counter // 3, counter % 3])
        ax[counter // 3, counter % 3].set_title(f'Boxplot of {dataset_name} Values (Test)')
        ax[counter // 3, counter % 3].set_xlabel("")
        counter += 1
plt.tight_layout()
plt.show()
# %%
import os
import sys
import pickle
import json
from tdc.single_pred import ADME
import numpy as np
from tdc.benchmark_group import admet_group
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark.prompt_template import *
from benchmark.utils import *
from benchmark.TDC_BenchmarkGroup_prompt_generation import config_dict
import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")
fig, ax = plt.subplots(3, 3, figsize=(12, 8))
counter = 0

for dataset_name, config in config_dict.items():
    if config["task"] == "regression":
        group = admet_group(path='data/')
        benchmark = group.get(dataset_name)
        predictions = {}
        name = benchmark['name']
        ref_df, test_df = benchmark['train_val'], benchmark['test']
        Q1 = ref_df['Y'].quantile(0.25)
        Q3 = ref_df['Y'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        ref_df = ref_df[(ref_df['Y'] >= lower_bound) & (ref_df['Y'] <= upper_bound)]

        sns.boxplot(x=ref_df['Y'], ax=ax[counter // 3, counter % 3])
        ax[counter // 3, counter % 3].set_title(f'Processed {dataset_name} Values')
        ax[counter // 3, counter % 3].set_xlabel("")
        counter += 1
plt.tight_layout()
plt.show()
