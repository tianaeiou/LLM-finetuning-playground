# %%
import json
import sys
import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski
from dataset.utils import normalize, denormalize, get_random_index
from template.cpds_pk_prompt_template import *

sys.path.append("..")

TARGET_RANGE = {
    'human VDss (L/kg)': [0.0, 700.0],
    'human CL (mL/min/kg)': [0.0, 1070.0],
    'fraction unbound \nin plasma (fu)': [0.0, 1.0],
    'terminal  t1/2 (h)': [0.0, 1344.0],
}
TARGET_TYPE = {
    'human VDss (L/kg)': "VDss",
    'human CL (mL/min/kg)': "CL",
    'fraction unbound \nin plasma (fu)': "fu",
    'terminal  t1/2 (h)': "t1/2",
}
PROPERTIES_DICT = {
    "all": ["MW", "HBA", "HBD", "TPSA_NO", "RotBondCount", "moka_ionState7.4", "MoKa.LogP", "MoKa.LogD7.4"],
    "normal": ["MW", "HBA", "HBD", "TPSA_NO", "RotBondCount"],
    "moka": ["moka_ionState7.4", "MoKa.LogP", "MoKa.LogD7.4"],
    "ruleof5": ["MW", "CLogP", "HBD", "HBA", "RB", "TPSA"]
}


def get_description(cur_drug_property_values, properties):
    property_prompt = ""
    for property in properties:
        property_prompt += PROPERTIES_TEMPLATES[property].format(value=cur_drug_property_values[property])
    return property_prompt


def get_examples(ref_list, property_strategy, target_type):
    if len(ref_list) == 0:
        return None

    example_prompt = ""
    for neighbor in ref_list:
        cur_target_value = neighbor[target_type]
        cur_drug_id = neighbor["Name"]
        cur_drug_smiles = neighbor["SMILES"]
        [min_value, max_value] = TARGET_RANGE[target_type]
        normalized_answer = normalize(cur_target_value, min_value, max_value)
        if property_strategy:
            properties = PROPERTIES_DICT[property_strategy]
            property_description = get_description(neighbor, properties)
            cur_example = EXAMPLE_TEMPLATE_WITH_DESCRIPTION.format(CUR_ANSWER=normalized_answer).replace("CUR_DRUG_ID",
                                                                                                         cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_description)
        else:
            cur_example = EXAMPLE_TEMPLATE.format(CUR_ANSWER=normalized_answer).replace("CUR_DRUG_ID",
                                                                                        cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles)
        example_prompt += cur_example

    return example_prompt


def get_prompt_body(cur_drug_dict, property_strategy, example_prompt, no_context=False):
    cur_drug_id = cur_drug_dict["Name"]
    cur_drug_smiles = cur_drug_dict["SMILES"]
    if not example_prompt:
        if property_strategy:
            properties = PROPERTIES_DICT[property_strategy]
            property_prompt = get_description(cur_drug_dict, properties)
            return ZERO_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY.replace("CUR_DRUG_ID", cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_prompt).replace(
                "CUR_TARGET", TARGET_TYPE[target_type])
        else:
            return ZERO_SHOT_QUESTION_TEMPLATE.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                                           cur_drug_smiles).replace(
                "CUR_TARGET", TARGET_TYPE[target_type])
    else:
        if property_strategy:
            properties = PROPERTIES_DICT[property_strategy]
            property_prompt = get_description(cur_drug_dict, properties)
            return FEW_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY.replace("CUR_DRUG_ID", cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_prompt).replace(
                "CUR_EXAMPLES", example_prompt).replace("CUR_TARGET", TARGET_TYPE[target_type])

        else:
            return FEW_SHOT_QUESTION_TEMPLATE.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                                          cur_drug_smiles).replace(
                "CUR_EXAMPLES", example_prompt).replace("CUR_TARGET", TARGET_TYPE[target_type])


def generated_formatted_prompt(cur_durg_dict, target_type, ref_list, property_strategy):
    examples = get_examples(ref_list, property_strategy, target_type)
    prompt_body = get_prompt_body(cur_durg_dict, property_strategy, examples)
    [min_value, max_value] = TARGET_RANGE[target_type]
    res = {
        "instruction": CPDS_PK_INSTRUCTION_TEMPLATE,
        "input": prompt_body,
        "output": normalize(cur_durg_dict[target_type], min_value, max_value)
    }
    return res


THRESHOLD = {
    'human VDss (L/kg)': {},
    'human CL (mL/min/kg)': {},
    'fraction unbound \nin plasma (fu)': {"THRESHOLD0": 200, "THRESHOLD1": 50},
    'terminal  t1/2 (h)': {},
}


def get_examples_cls(ref_list, property_strategy, target_type, threshold0, threshold1):
    if len(ref_list) == 0:
        return None

    example_prompt = ""

    for neighbor in ref_list:
        cur_target_value = neighbor[target_type]
        cur_drug_id = neighbor["Name"]
        cur_drug_smiles = neighbor["SMILES"]
        [min_value, max_value] = TARGET_RANGE[target_type]
        normalized_answer = normalize(cur_target_value, min_value, max_value)
        if normalized_answer > threshold0:
            category = 0
        elif normalized_answer in [threshold1, threshold0]:
            category = 1
        else:
            category = 2

        if property_strategy:
            properties = PROPERTIES_DICT[property_strategy]
            property_description = get_description(neighbor, properties)
            cur_example = EXAMPLE_TEMPLATE_WITH_DESCRIPTION.format(CUR_ANSWER=category).replace("CUR_DRUG_ID",
                                                                                                cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_description)
        else:
            cur_example = EXAMPLE_TEMPLATE.format(CUR_ANSWER=category).replace("CUR_DRUG_ID",
                                                                               cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles)
        example_prompt += cur_example

    return example_prompt


def get_prompt_body_cls(cur_drug_dict, property_strategy, example_prompt, threshold0, threshold1, target_type,
                        no_context=False):
    cur_drug_id = cur_drug_dict["Name"]
    cur_drug_smiles = cur_drug_dict["SMILES"]
    if not example_prompt:
        print("zero-shot")
        if property_strategy:
            print("use property")
            properties = PROPERTIES_DICT[property_strategy]
            property_prompt = get_description(cur_drug_dict, properties)
            return ZERO_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY_CLS.format(THRESHOLD0=threshold0,
                                                                        THRESHOLD1=threshold1).replace("CUR_DRUG_ID",
                                                                                                       cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_prompt).replace(
                "CUR_TARGET", TARGET_TYPE[target_type])
        else:
            print("no property")
            return ZERO_SHOT_QUESTION_TEMPLATE_CLS.format(THRESHOLD0=threshold0, THRESHOLD1=threshold1).replace(
                "CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                    cur_drug_smiles).replace("CUR_TARGET", TARGET_TYPE[target_type])
    else:
        print("few-shot")
        if property_strategy:
            print("use property")
            properties = PROPERTIES_DICT[property_strategy]
            property_prompt = get_description(cur_drug_dict, properties)
            return FEW_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY_CLS.format(THRESHOLD0=threshold0,
                                                                       THRESHOLD1=threshold1).replace("CUR_DRUG_ID",
                                                                                                      cur_drug_id).replace(
                "CUR_DRUG_SMILES", cur_drug_smiles).replace("CUR_DRUG_PROPERTIES", property_prompt).replace(
                "CUR_EXAMPLES", example_prompt).replace("CUR_TARGET", TARGET_TYPE[target_type])

        else:
            print("no property")
            return FEW_SHOT_QUESTION_TEMPLATE_CLS.format(THRESHOLD0=threshold0, THRESHOLD1=threshold1).replace(
                "CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                    cur_drug_smiles).replace(
                "CUR_EXAMPLES", example_prompt).replace("CUR_TARGET", TARGET_TYPE[target_type])


def generated_formatted_prompt_cls(cur_durg_dict, target_type, ref_list, property_strategy, threshold):
    threshold0 = threshold['THRESHOLD0']
    threshold1 = threshold['THRESHOLD1']
    examples = get_examples_cls(ref_list, property_strategy, target_type, threshold0, threshold1)

    prompt_body = get_prompt_body_cls(cur_durg_dict, property_strategy, examples, threshold0, threshold1, target_type)

    [min_value, max_value] = TARGET_RANGE[target_type]
    normalized_answer = normalize(cur_durg_dict[target_type], min_value, max_value)
    if normalized_answer > threshold0:
        category = 0
    elif normalized_answer in [threshold1, threshold0]:
        category = 1
    else:
        category = 2
    res = {
        "instruction": CPDS_PK_INSTRUCTION_TEMPLATE,
        "input": prompt_body,
        "output": category
    }
    return res


def calculate_similarity(data, target_list, file_dir, num_to_save=20):
    if os.path.isfile(file_dir):
        with open(file_dir, 'rb') as file:
            try:
                neighbor_dict = pickle.load(file)
                print("neighbor data loaded!")
                return neighbor_dict
            except Exception as e:
                print("generating neighbor data")

    neighbor_dict = {}
    for target_type in target_list:
        data_cleaned = data[use_cols].copy().dropna(subset=["SMILES", target_type])
        print(f"processing target: {target_type} with data: {data_cleaned.shape}")
        cur_smiles_list = data_cleaned["SMILES"].to_list()
        molecules = [Chem.MolFromSmiles(smiles) for smiles in cur_smiles_list]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules]
        nearest_neighbors = {i: [] for i in range(len(cur_smiles_list))}
        for i, fp1 in enumerate(fps):
            for j, fp2 in enumerate(fps):
                if i != j:
                    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                    if len(nearest_neighbors[i]) < num_to_save:
                        nearest_neighbors[i].append((j, similarity))
                    else:
                        nearest_neighbors[i].append((j, similarity))
                        nearest_neighbors[i].sort(key=lambda x: x[1], reverse=True)
                        nearest_neighbors[i] = nearest_neighbors[i][:num_to_save]
        neighbor_dict[target_type] = {"smiles": cur_smiles_list, "neighbors": nearest_neighbors}
    with open(file_dir, 'wb') as f:
        pickle.dump(neighbor_dict, f)
    return neighbor_dict


def analyze_data(df_cleaned, target_type='fraction unbound \nin plasma (fu)'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.histplot(df_cleaned[target_type], kde=False)
    plt.xlabel('Fraction Unbound in Plasma (fu)')
    plt.ylabel('Count')
    plt.title('Distribution of Fraction Unbound in Plasma (fu)')
    plt.show()


config_dict = {
    "VDss": {
        "dataset": "human VDss (L/kg)",
        "task": "regression",
        "example_strategy": "similarity",
        "k": [2, 5],
        "properties": "ruleof5",
        "split": "test",
        "template": VDSS_PKPD_TEMPLATE
    },
    "CL": {
        "dataset": "human CL (mL/min/kg)",
        "task": "regression",
        "example_strategy": "similarity",
        "k": [2, 5],
        "properties": "ruleof5",
        "split": "test",
        "template": CL_PKPD_TEMPLATE
    },
    "fu": {
        "dataset": "fraction unbound \nin plasma (fu)",
        "task": "regression",
        "example_strategy": "similarity",
        "k": [2, 5],
        "properties": "ruleof5",
        "split": "test",
        "template": FU_PKPD_TEMPLATE
    },
    "t1/2": {
        "dataset": "terminal  t1/2 (h)",
        "task": "regression",
        "example_strategy": "similarity",
        "k": [2, 5],
        "properties": "ruleof5",
        "split": "test",
        "template": T12_PKPD_TEMPLATE
    },
}


def get_data(data_dir, invalid_rows=[550, 554, 829, 1256]):
    data = pd.read_csv(data_dir).drop(invalid_rows)
    return data


dataset_name = "fu"
config = config_dict[dataset_name]
print("processing dataset {}".format(dataset_name))

project_folder = "/home/wangtian/codeSpace/LLM-finetuning-playground"
data_dir = f"{project_folder}/pd_data/cpds_pk_data.csv"
data = get_data(data_dir)
target_type = config["dataset"]

neighbor_dict_dir = f"{project_folder}/cpds_pk_data/cpds_pk-smiles-neighbors.pkl"
# calculate neighbor dict for all the four target_type
neighbor_dict = calculate_similarity(data,
                                     ['human VDss (L/kg)', 'human CL (mL/min/kg)', 'fraction unbound \nin plasma (fu)',
                                      'terminal  t1/2 (h)'], neighbor_dict_dir)
# clean the data
df_cleaned = data.copy().dropna(subset=["SMILES", target_type])
df_cleaned.reset_index(drop=True, inplace=True)

min_value, max_value = get_min_max_value(ref_df, "Y")
if __name__ == "__main__":

    # (1352, 20) -> 1348

    # processing target: human VDss (L/kg) with data: (1315, 14)
    # processing target: human CL (mL/min/kg) with data: (1346, 14)
    # processing target: fraction unbound in plasma (fu) with data: (917, 14)
    # processing target: terminal  t1/2 (h) with data: (1331, 14)

    use_cols = ['Name', 'SMILES', 'human VDss (L/kg)', 'human CL (mL/min/kg)',
                'fraction unbound \nin plasma (fu)', 'terminal  t1/2 (h)',
                'MW',
                'HBA', 'HBD', 'TPSA_NO', 'RotBondCount', 'moka_ionState7.4',
                'MoKa.LogP', 'MoKa.LogD7.4']

    TARGET_NAME = {
        'human VDss (L/kg)': "VDss",
        'human CL (mL/min/kg)': "CL",
        'fraction unbound \nin plasma (fu)': "fu",
        'terminal  t1/2 (h)': "t1/2",
    }
    k_shot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    target_type = 'fraction unbound \nin plasma (fu)'
    df_cleaned = data[use_cols].copy().dropna(subset=["SMILES", target_type])
    df_cleaned.reset_index(drop=True, inplace=True)
    example_strategy_list = ["similarity"]  # ["similarity", "random"]
    property_strategy_list = ["ruleof5"]  # ["all", "moka", "normal"]

    # task = "regression"  #"classification"
    is_True = True
    if is_True:
        # generate_file_name:
        for k in k_shot:
            # property_strategy = None
            example_strategy = "similarity"
            for property_strategy in ["all", "moka", "normal"]:
                prefix = f"cpds-pk-{k}-shot-{TARGET_NAME[target_type]}"
                if k > 0:
                    prefix += f"-{example_strategy}"
                if property_strategy:
                    prefix += f"-{property_strategy}"
                file_name = f"{prefix}.json"

                # generate_prompt
                prompt_list = []
                # 遍历数据：
                for idx, row in df_cleaned.iterrows():
                    ref_list = []
                    if k > 0:
                        # get neighbors_index_list
                        if example_strategy == "random":
                            neighbors_index_list = get_random_index(k, 0, df_cleaned.shape[0] - 1, [idx])
                        else:
                            cur_neighbor_dict = neighbor_dict[target_type]
                            neighbors_index_list = [f[0] for f in cur_neighbor_dict["neighbors"][idx][:k]]
                        # get ref_list according to neighbors_index_list
                        for neighbor_idx in neighbors_index_list:
                            ref_list.append(df_cleaned.loc[neighbor_idx].to_dict())

                    prompt_json = generated_formatted_prompt(row.to_dict(), target_type, ref_list, property_strategy)
                    prompt_list.append(prompt_json)
                    # print(prompt_json["input"])
                    # break
                with open(
                        f"/home/wangtian/codeSpace/LLM-finetuning-playground/cpds_pk_data/regression/properties/{file_name}",
                        'w') as json_file:
                    json.dump(prompt_list, json_file)
                print(
                    f"{k}-shot data saved at /home/wangtian/codeSpace/LLM-finetuning-playground/cpds_pk_data/{file_name}")

        # print("done")
