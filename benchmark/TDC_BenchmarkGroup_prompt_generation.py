# %%
import os
import sys
import pickle
import json
from tdc.single_pred import ADME
import numpy as np
from tdc.benchmark_group import admet_group
from rdkit import Chem
from benchmark.prompt_template import *
from benchmark.utils import *
import warnings

warnings.filterwarnings("ignore")

sys.path.append("..")

group = admet_group(path='data/')
print(group.dataset_names)
CACO_TEMPLATE = {
    "instruction": CACO_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CACO_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "few_shot_template": CACO_FEW_SHOT_TEMPLATE,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": CACO_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
BBB_TEMPLATE = {
    "instruction": BBB_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": BBB_FEW_SHOT_TEMPLATE_WITH_PROPERTY_v2,
    "few_shot_template": BBB_FEW_SHOT_TEMPLATE_v2,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
CYP2C9_Veith_TEMPLATE = {
    "instruction": CYP2C9_Veith_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CYP2C9_Veith_FEW_SHOT_TEMPLATE_WITH_PROPERTY_v1,
    "few_shot_template": CYP2C9_Veith_FEW_SHOT_TEMPLATE_v1,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}

HL_TEMPLATE = {
    "instruction": HR_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": HR_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "few_shot_template": HR_FEW_SHOT_TEMPLATE,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": HR_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
LD50_TEMPLATE = {
    "instruction": LD50_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": LD50_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "few_shot_template": LD50_FEW_SHOT_TEMPLATE,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": LD50_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
HIA_HOU_TEMPLATE = {
    "instruction": HIA_HOU_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": HIA_HOU_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
PGP_TEMPLATE = {
    "instruction": PGP_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": PGP_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
MA_TEMPLATE = {
    "instruction": MA_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": MA_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
LIPO_TEMPLATE = {
    "instruction": LIPO_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": LIPO_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": LIPO_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
Solubility_TEMPLATE = {
    "instruction": Solubility_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": Solubility_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": Solubility_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
PPBR_TEMPLATE = {
    "instruction": PPBR_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": PPBR_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": PPBR_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
VDSS_TEMPLATE = {
    "instruction": VDSS_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": VDSS_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "few_shot_template": VDSS_TEMPLATE_FEW_SHOT_TEMPLATE,
    "raw_few_shot_template_with_property": VDSS_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW
}
CYP2D6_TEMPLATE = {
    "instruction": CYP2D6_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CYP2D6_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
CYP3A4_TEMPLATE = {
    "instruction": CYP3A4_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CYP3A4_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
CYP2D6_SUBSTRATE_TEMPLATE = {
    "instruction": CYP2D6_SUBSTRATE_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CYP2D6_SUBSTRATE_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
CYP3A4_SUBSTRATE_TEMPLATE = {
    "instruction": CYP3A4_SUBSTRATE_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CYP3A4_SUBSTRATE_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
CYP2C9_SUBSTRATE_SUBSTRATE_TEMPLATE = {
    "instruction": CYP2C9_SUBSTRATE_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": CYP2C9_SUBSTRATE_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
Clearance_Hepatocyte_AZ_TEMPLATE = {
    "instruction": Clearance_Hepatocyte_AZ_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": Clearance_Hepatocyte_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": Clearance_Hepatocyte_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW,
}
Clearance_Microsome_AZ_TEMPLATE = {
    "instruction": Clearance_Microsome_AZ_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": Clearance_Microsome_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
    "raw_few_shot_template_with_property": Clearance_Microsome_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW,
}
hERG_TEMPLATE = {
    "instruction": hERG_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": hERG_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
AMES_TEMPLATE = {
    "instruction": AMES_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": AMES_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
DILI_TEMPLATE = {
    "instruction": DILI_TEMPLATE_INSTRUCTION_TEMPLATE,
    "few_shot_template_with_property": DILI_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY,
    "example_template_with_properties": EXAMPLE_TEMPLATE_WITH_PROPERTIES,
    "example_template": EXAMPLE_TEMPLATE,
}
config_dict = {
    "caco2_wang": {
        "dataset": "Caco2_Wang",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CACO_TEMPLATE
    },
    "hia_hou": {
        "dataset": "HIA_Hou",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HIA_HOU_TEMPLATE
    },
    "pgp_broccatelli": {
        "dataset": "Pgp_Broccatelli",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PGP_TEMPLATE
    },
    "bioavailability_ma": {
        "dataset": "Bioavailability_Ma",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": MA_TEMPLATE
    },
    "lipophilicity_astrazeneca": {
        "dataset": "Lipophilicity_AstraZeneca",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LIPO_TEMPLATE
    },
    "solubility_aqsoldb": {
        "dataset": "Solubility_AqSolDB",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Solubility_TEMPLATE
    },
    "bbb_martins": {
        "dataset": "BBB_Martins",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": BBB_TEMPLATE
    },
    "ppbr_az": {
        "dataset": "PPBR_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": PPBR_TEMPLATE
    },
    "vdss_lombardo": {
        "dataset": "VDss_Lombardo",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": VDSS_TEMPLATE
    },
    "cyp2d6_veith": {
        "dataset": "CYP2D6_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_TEMPLATE
    },
    "cyp3a4_veith": {
        "dataset": "CYP3A4_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # 2,
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_TEMPLATE
    },
    "cyp2c9_veith": {
        "dataset": "CYP2C9_Veith",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_Veith_TEMPLATE
    },
    "cyp2d6_substrate_carbonmangels": {
        "dataset": "CYP2D6_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2D6_SUBSTRATE_TEMPLATE
    },
    "cyp3a4_substrate_carbonmangels": {
        "dataset": "CYP3A4_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP3A4_SUBSTRATE_TEMPLATE
    },
    "cyp2c9_substrate_carbonmangels": {
        "dataset": "CYP2C9_Substrate_CarbonMangels",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],  # [2],#
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": CYP2C9_SUBSTRATE_SUBSTRATE_TEMPLATE
    },
    "half_life_obach": {
        "dataset": "Half_Life_Obach",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": HL_TEMPLATE
    },
    "clearance_microsome_az": {
        "dataset": "Clearance_Microsome_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Microsome_AZ_TEMPLATE
    },
    "clearance_hepatocyte_az": {
        "dataset": "Clearance_Hepatocyte_AZ",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": Clearance_Hepatocyte_AZ_TEMPLATE
    },
    "herg": {
        "dataset": "hERG",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": hERG_TEMPLATE
    },
    "ames": {
        "dataset": "AMES",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": AMES_TEMPLATE
    },
    "dili": {
        "dataset": "DILI",
        "task": "binary_classification",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": DILI_TEMPLATE
    },
    "ld50_zhu": {
        "dataset": "LD50_Zhu",
        "task": "regression",
        "example_strategy": "similarity",  # ["similarity","random"]
        "k": [1, 2, 5, 8, 10],
        "properties": "ruleof5",  # [None,"ruleof5"]
        "split": "test",  # ["train+valid","test"]
        "template": LD50_TEMPLATE
    },
}


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


def calculate_similarity_cls(data_df, ref_df, file_dir, num_of_neighbors_per_cat=5):
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
    smiles_list_test, smiles_list_ref_0, smiles_list_ref_1 = data_df["Drug"].to_list(), ref_df[ref_df["Y"] == 0][
        "Drug"].to_list(), ref_df[ref_df["Y"] == 1]["Drug"].to_list()
    ref_df_0_index, ref_df_1_index = ref_df[ref_df["Y"] == 0].index, ref_df[ref_df["Y"] == 1].index
    # calculate similarity
    molecules_test = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_test]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_test]
    molecules_ref_0 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_ref_0]
    molecules_ref_1 = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_ref_1]
    fps_ref_0 = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_ref_0]
    fps_ref_1 = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_ref_1]
    nearest_neighbors = {i: [] for i in range(len(smiles_list_test))}
    for i, fp1 in enumerate(fps_test):
        print(i)
        neighbor_0 = []
        neighbor_1 = []
        for j, fp2 in enumerate(fps_ref_0):
            if fp1 != fp2:
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                if len(neighbor_0) < num_of_neighbors_per_cat:
                    neighbor_0.append((ref_df_0_index[j], similarity))
                else:
                    neighbor_0.append((ref_df_0_index[j], similarity))
                    neighbor_0.sort(key=lambda x: x[1], reverse=True)
                    neighbor_0 = neighbor_0[:num_of_neighbors_per_cat]
        for j, fp2 in enumerate(fps_ref_1):
            if fp1 != fp2:
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                if len(neighbor_1) < num_of_neighbors_per_cat:
                    neighbor_1.append((ref_df_1_index[j], similarity))
                else:
                    neighbor_1.append((ref_df_1_index[j], similarity))
                    neighbor_1.sort(key=lambda x: x[1], reverse=True)
                    neighbor_1 = neighbor_1[:num_of_neighbors_per_cat]
        nearest_neighbors[i] = neighbor_0 + neighbor_1
        assert ref_df["Y"].to_list()[nearest_neighbors[i][0][0]] == 0
        assert ref_df["Y"].to_list()[nearest_neighbors[i][5][0]] == 1
        result = {"smiles": smiles_list_test, "neighbor_index": nearest_neighbors}
    # save to file_dir
    with open(file_dir, 'wb') as f:
        pickle.dump(result, f)
    return result


def get_fixed_interval_examples(ref_df):
    # Case1 ： use quantiles
    # Case 2: use 100,200,300,400
    # quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    quantile_values = ref_df['Y'].quantile(quantiles)
    record_dict = {}
    for quantile, quantile_value in zip(quantiles, quantile_values):
        mask = ref_df['Y'].apply(lambda x: np.isclose(x, quantile_value, atol=0.01))
        rows = ref_df[mask]
        if len(rows) == 0:
            if quantile - 0.1 > 0:
                quantile_values1 = ref_df['Y'].quantile(quantile - 0.1)
                mask1 = ref_df['Y'].apply(lambda x: np.isclose(x, quantile_values1, atol=0.02))
                rows1 = ref_df[mask1]
                if len(rows1) > 0:
                    record_dict[quantile] = rows1.iloc[0].name
            else:
                print(f"{quantile} failed!")
        else:
            record_dict[quantile] = rows.iloc[0].name
    return [value for value in record_dict.values()]


def get_normalized_fixed_interval_examples(ref_df):
    # Case1 ： use quantiles
    # Case 2: use 100,200,300,400
    # quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    quantiles = [0.9, 0.7, 0.5, 0.3, 0.1]
    quantile_values = ref_df['Y'].quantile(quantiles)
    record_dict = {}
    for quantile, quantile_value in zip(quantiles, quantile_values):
        mask = ref_df['Y'].apply(lambda x: np.isclose(x, quantile_value, atol=0.01))
        rows = ref_df[mask]
        if len(rows) == 0:
            if quantile - 0.1 > 0:
                quantile_values1 = ref_df['Y'].quantile(quantile - 0.1)
                mask1 = ref_df['Y'].apply(lambda x: np.isclose(x, quantile_values1, atol=0.02))
                rows1 = ref_df[mask1]
                if len(rows1) > 0:
                    record_dict[quantile] = rows1.iloc[0].name
            else:
                print(f"{quantile} failed!")
        else:
            record_dict[quantile] = rows.iloc[0].name
    return [value for value in record_dict.values()]


def get_examples(ref_list, property_type, min_value, max_value, template_dict, task):
    if len(ref_list) == 0:
        return None
    example_prompt = ""
    properties = PROPERTIES_DICT.get(property_type, None)
    for neighbor in ref_list:
        cur_target_value = neighbor["Target"]
        cur_drug_id = neighbor["Name"]
        cur_drug_smiles = neighbor["SMILES"]
        if min_value and max_value:
            processed_answer = normalize(cur_target_value, min_value, max_value)
        else:
            if task != "regression":
                processed_answer = int(cur_target_value)
            else:
                processed_answer = float(cur_target_value)
        if properties:
            # get properties
            property_values = get_property_values(cur_drug_smiles, properties)
            property_description = get_property_prompt(property_values, property_values)
            cur_example = template_dict["example_template_with_properties"].format(
                CUR_ANSWER=processed_answer).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES",
                                                            property_description)  # replace("CUR_DRUG_ID",cur_drug_id)
        else:
            cur_example = template_dict["example_template"].format(CUR_ANSWER=processed_answer).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles)  # replace("CUR_DRUG_ID",cur_drug_id).
        example_prompt += cur_example

    return example_prompt


def get_prompt_body(cur_drug_dict, property_strategy, example_prompt, template_dict):
    cur_drug_id = str(cur_drug_dict["Drug_ID"])
    cur_drug_smiles = str(cur_drug_dict["Drug"])
    if not example_prompt:
        pass
        raise NotImplementedError
    else:
        if property_strategy:
            properties = PROPERTIES_DICT[property_strategy]
            property_values = get_property_values(cur_drug_smiles, properties)
            property_prompt = get_property_prompt(property_values, properties)
            return template_dict["few_shot_template_with_property"].replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_prompt).replace(
                "CUR_EXAMPLES", example_prompt)  # .replace("CUR_DRUG_ID", cur_drug_id)
        else:
            return template_dict["few_shot_template"].replace("CUR_DRUG_SMILES", cur_drug_smiles).replace(
                "CUR_EXAMPLES", example_prompt)  # .replace("CUR_DRUG_ID", cur_drug_id)
            raise NotImplementedError


def generate_formatted_prompt(cur_drug_dict, ref_list, min_value, max_value, template_dict,
                              property_type=None, task="regression"):
    examples = get_examples(ref_list, property_type, min_value, max_value, template_dict, task)
    prompt_body = get_prompt_body(cur_drug_dict, property_type, examples, template_dict)
    # if task == "regression":
    #     output_value = normalize(cur_drug_dict["Y"], min_value, max_value)
    # else:
    #     output_value = int(cur_drug_dict["Y"])
    if min_value and max_value:
        processed_answer = normalize(cur_drug_dict["Y"], min_value, max_value)
    else:
        if task != "regression":
            processed_answer = int(cur_drug_dict["Y"])
        else:
            processed_answer = float(cur_drug_dict["Y"])
    res = {
        "instruction": template_dict["instruction"],
        "input": prompt_body,
        "output": processed_answer
    }
    return res


if __name__ == "__main__":
    project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"
    # existing_files = os.listdir(f"{project_path}/data/ADMET_benchmark/")
    for dataset_name, config in config_dict.items():
        # dataset_name = "bbb_martins"
        # config = config_dict[dataset_name]
        if config["task"] != "regression":
            config["k"] = [2]

            # config["k"] = [1, 2, 5]
            # config["example_strategy"] = "similarity"
            # config["normalize"] = False
            # if not config["normalize"]:
            #     config["few_shot_template_with_property"] = config["raw_few_shot_template_with_property"]

            print("processing dataset {}".format(dataset_name))

            # get_Data (scaffold split)
            # https://tdcommons.ai/benchmark/admet_group/overview/
            group = admet_group(path='data/')
            benchmark = group.get(config["dataset"])
            predictions = {}
            name = benchmark['name']
            train_val_df, test_df = benchmark['train_val'], benchmark['test']
            if config["split"] == "test":
                data_df = test_df
                ref_df = train_val_df
                # import pandas as pd
                # ref_df = pd.concat([data_df, train_val_df], axis=1)
            else:
                data_df = train_val_df
                ref_df = test_df
            # get few-shot example
            use_q1q3 = False
            if config["task"] == "regression":
                if use_q1q3:
                    Q1 = ref_df['Y'].quantile(0.25)
                    Q3 = ref_df['Y'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    ref_df = ref_df[(ref_df['Y'] >= lower_bound) & (ref_df['Y'] <= upper_bound)]
                    neighbor_dict = calculate_similarity(data_df, ref_df,
                                                         f"{project_path}/data/Caco2_benchmark/q1q3_{config["dataset"]}-smiles-neighbors_{config["split"]}.pkl")
                else:
                    neighbor_dict = calculate_similarity(data_df, ref_df,
                                                         f"{project_path}/data/Caco2_benchmark/{config["dataset"]}-smiles-neighbors_{config["split"]}.pkl")
            else:
                # for classification task
                # neighbor_dict = calculate_similarity(data_df, ref_df,
                #                                      f"{project_path}/data/Caco2_benchmark/{config["dataset"]}-smiles-neighbors_{config["split"]}_cls.pkl")
                neighbor_dict = calculate_similarity_cls(data_df, ref_df,
                                                         f"{project_path}/data/Caco2_benchmark/cls_{config["dataset"]}-smiles-neighbors_{config["split"]}_cls.pkl")
            quantile_neighbor_idx_list = get_fixed_interval_examples(ref_df)
            min_value, max_value = get_min_max_value(ref_df, "Y")
            # min_value, max_value = None, None
            # if config.get("normalize", None):
            #     min_value, max_value = get_min_max_value(ref_df, "Y")

            # get prompt

            assert neighbor_dict["smiles"] == test_df["Drug"].to_list()
            for k in config["k"]:
                prompt_list = []
                if use_q1q3:
                    file_name = f"q1q3_{config["dataset"]}-{config['example_strategy']}-{k}shot-{config['properties']}-{config['split']}.json"
                else:
                    file_name = f"balanced_{config["dataset"]}-{config['example_strategy']}-{k}shot-{config['properties']}-{config['split']}.json"

                for idx, row in test_df.iterrows():
                    # get neighbors
                    ref_list = []
                    if config['example_strategy'] == "similarity":
                        if config["task"] == "regression":
                            for j in range(k):
                                neighbor_idx = neighbor_dict["neighbor_index"][idx][j][0]
                                # neighbor_smiles = ref_df["Drug"].to_list()[neighbor_idx]
                                mol = Chem.MolFromSmiles(ref_df["Drug"].to_list()[neighbor_idx])
                                neighbor_smiles = Chem.MolToSmiles(mol)
                                neighbor_id = ref_df["Drug_ID"].to_list()[neighbor_idx]
                                neighbor_target = ref_df["Y"].to_list()[neighbor_idx]
                                ref_list.append(
                                    {"Target": neighbor_target, "Name": str(neighbor_id),
                                     "SMILES": str(neighbor_smiles)})

                        # elif config['example_strategy'] == "quantile":
                        #     for neighbor_idx in quantile_neighbor_idx_list:
                        #         # neighbor_smiles = ref_df["Drug"].to_list()[neighbor_idx]
                        #         mol = Chem.MolFromSmiles(ref_df["Drug"].to_list()[neighbor_idx])
                        #         neighbor_smiles = Chem.MolToSmiles(mol)
                        #         neighbor_id = ref_df["Drug_ID"].to_list()[neighbor_idx]
                        #         neighbor_target = ref_df["Y"].to_list()[neighbor_idx]
                        #         ref_list.append(
                        #             {"Target": neighbor_target, "Name": str(neighbor_id), "SMILES": str(neighbor_smiles)})

                        # add properties to prompt
                        else:
                            assert k == 2
                            j = 0
                            neighbor_idx = neighbor_dict["neighbor_index"][idx][j][0]
                            # neighbor_smiles = ref_df["Drug"].to_list()[neighbor_idx]
                            mol = Chem.MolFromSmiles(ref_df["Drug"].to_list()[neighbor_idx])
                            neighbor_smiles = Chem.MolToSmiles(mol)
                            neighbor_id = ref_df["Drug_ID"].to_list()[neighbor_idx]
                            neighbor_target = ref_df["Y"].to_list()[neighbor_idx]
                            assert neighbor_target == 0
                            ref_list.append(
                                {"Target": neighbor_target, "Name": str(neighbor_id),
                                 "SMILES": str(neighbor_smiles)})

                            neighbor_idx = neighbor_dict["neighbor_index"][idx][5 + j][0]
                            # neighbor_smiles = ref_df["Drug"].to_list()[neighbor_idx]

                            mol = Chem.MolFromSmiles(ref_df["Drug"].to_list()[neighbor_idx])
                            neighbor_smiles = Chem.MolToSmiles(mol)
                            neighbor_id = ref_df["Drug_ID"].to_list()[neighbor_idx]
                            neighbor_target = ref_df["Y"].to_list()[neighbor_idx]
                            assert neighbor_target == 1
                            ref_list.append(
                                {"Target": neighbor_target, "Name": str(neighbor_id),
                                 "SMILES": str(neighbor_smiles)})
                    print(len(ref_list), " example done!")
                    prompt_json = generate_formatted_prompt(row.to_dict(), ref_list, min_value, max_value,
                                                            config["template"],
                                                            config["properties"], config["task"])
                    prompt_list.append(prompt_json)

                with open(f"{project_path}/data/ADMET_benchmark/{file_name}", 'w') as json_file:
                    json.dump(prompt_list, json_file)
                print(f"{project_path}/data/ADMET_benchmark/{file_name}")
