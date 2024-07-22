# Caco-2： https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al
# %%
import pickle
import json
import random
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski
from template.caco2_prompt_template import *
from dataset.utils import normalize, denormalize, get_random_index

# %%

MIN_VALUE = -7.7600002
MAX_VALUE = -3.51


# def normalize_permeability(value, min_value=MIN_VALUE, max_value=MAX_VALUE, min_output=0.0, max_output=1000.0):
#     normalized_value = (value - min_value) * (max_output - min_output) / (max_value - min_value) + min_output
#     return int(normalized_value)

# def denormalize_permeability(normalized_value, min_value=MIN_VALUE, max_value=MAX_VALUE, min_output=0.0, max_output=1000.0):
#     value = (normalized_value - min_output) * (max_value - min_value) / (max_output - min_output) + min_value
#     return value

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "MW": "undefined",
            "CLogP": "undefined",
            "HBD": "undefined",
            "HBA": "undefined",
            "RB": "undefined",
            "TPSA": "undefined"
        }

    properties = {
        "MW": round(Descriptors.MolWt(mol), 3),
        "CLogP": round(Descriptors.MolLogP(mol), 3),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RB": Lipinski.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 3)
    }

    return properties


def get_ruleof5_description(smiles):
    """generate prompt of rule of 5 info

    :param smiles: string
    :return: description prompt of rule of 5
    """
    properties = calculate_properties(smiles)
    return RULEOF5_TEMPLATE.format(MW=properties["MW"], CLogP=properties["CLogP"], HBD=properties["HBD"],
                                   HBA=properties["HBA"],
                                   RB=properties["RB"], TPSA=properties["TPSA"])


def calculate_similarity(splits,
                         save_path="/home/wangtian/codeSpace/LLM-finetuning-playground/data/Caco-smiles-neighbors.pkl"):
    result = {}

    for split in ["train", "valid", "test"]:
        cur_df = splits[split]
        cur_smiles_list = cur_df["Drug"].to_list()

        molecules = [Chem.MolFromSmiles(smiles) for smiles in cur_smiles_list]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules]

        nearest_neighbors = {i: [] for i in range(len(cur_smiles_list))}

        for i, fp1 in enumerate(fps):
            for j, fp2 in enumerate(fps):
                if i != j:
                    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                    if len(nearest_neighbors[i]) < 10:
                        nearest_neighbors[i].append((j, similarity))
                    else:
                        nearest_neighbors[i].append((j, similarity))
                        nearest_neighbors[i].sort(key=lambda x: x[1], reverse=True)
                        nearest_neighbors[i] = nearest_neighbors[i][:10]
        result[split] = {"smiles": cur_smiles_list, "neighbors": nearest_neighbors}

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    return result
    # with open('Caco-smiles-neighbors.pkl', 'rb') as f:
    #     res1 = pickle.load(f)


data = ADME(name='Caco2_Wang')
splits = data.get_split()
with open('/home/wangtian/codeSpace/LLM-finetuning-playground/data/Caco-smiles-neighbors.pkl', 'rb') as f:
    neighbor_dict = pickle.load(f)
# neighbor_dict = calculate_similarity(splits)
print("similarity calculated!")


def get_prompt_body(cur_drug_id, cur_drug_smiles, description_type, examples=None, no_context=False):
    if not examples:
        if not description_type:
            if no_context:
                return ZERO_SHOT_TEMPLATE_NO_CONTEXT.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                                                 cur_drug_smiles)
            else:
                return ZERO_SHOT_TEMPLATE.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                                      cur_drug_smiles)
        else:
            if no_context:
                # only support rule of 5
                description = get_ruleof5_description(cur_drug_smiles)
                return ZERO_SHOT_TEMPLATE_WITH_DESCRIPTION_NO_CONTEXT.replace("CUR_DRUG_DESCRIPTION",
                                                                              description).replace(
                    "CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES", cur_drug_smiles)
            else:
                # only support rule of 5
                description = get_ruleof5_description(cur_drug_smiles)
                return ZERO_SHOT_TEMPLATE_WITH_DESCRIPTION.replace("CUR_DRUG_DESCRIPTION", description).replace(
                    "CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES", cur_drug_smiles)

    else:
        if not description_type:
            if no_context:
                return FEW_SHOT_TEMPLATE_NO_CONTEXT.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                                                cur_drug_smiles).replace(
                    "CUR_EXAMPLES",
                    examples)
            else:
                return FEW_SHOT_TEMPLATE.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                                     cur_drug_smiles).replace(
                    "CUR_EXAMPLES",
                    examples)
        else:
            description = get_ruleof5_description(cur_drug_smiles)
            if no_context:
                return FEW_SHOT_TEMPLATE_WITH_DESCRIPTION_NO_CONTEXT.replace("CUR_DRUG_DESCRIPTION",
                                                                             description).replace(
                    "CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                        cur_drug_smiles).replace("CUR_EXAMPLES",
                                                                                 examples)
            else:
                return FEW_SHOT_TEMPLATE_WITH_DESCRIPTION.replace("CUR_DRUG_DESCRIPTION", description).replace(
                    "CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                        cur_drug_smiles).replace("CUR_EXAMPLES",
                                                                                 examples)


def get_examples(ref_list, description_type):
    """generate prompt for examples used in few-shot prompting strategies

    :param ref_list: top 10 neighbors [[neighbor_idx, neighbor_smiles, neighbor_id, neighbor_affinity],[],[]]
    :param description_type: currently only support rule of five
    :return: example_prompt
    """
    if len(ref_list) == 0:
        return None

    example_prompt = ""
    if description_type == "rule_of_5":
        for _, [_, neighbor_smiles, neighbor_id, neighbor_affinity] in enumerate(ref_list):
            description = get_ruleof5_description(neighbor_smiles)
            example_prompt += EXAMPLE_TEMPLATE_WITH_DESCRIPTION.format(
                CUR_ANSWER=normalize(neighbor_affinity, min_value=MIN_VALUE, max_value=MAX_VALUE)).replace(
                "CUR_DRUG_DESCRIPTION",
                description).replace(
                "CUR_DRUG_ID", neighbor_id).replace("CUR_DRUG_SMILES", neighbor_smiles)
    else:

        for _, [_, neighbor_smiles, neighbor_id, neighbor_affinity] in enumerate(ref_list):
            example_prompt += EXAMPLE_TEMPLATE.format(
                CUR_ANSWER=normalize(neighbor_affinity, min_value=MIN_VALUE, max_value=MAX_VALUE)).replace(
                "CUR_DRUG_ID", neighbor_id).replace("CUR_DRUG_SMILES", neighbor_smiles)
    return example_prompt


def generate_formatted_prompt(cur_drug_id, cur_drug_smiles, cur_drug_affinity, ref_list, description_type=None):
    examples = get_examples(ref_list, description_type)
    prompt_body = get_prompt_body(cur_drug_id, cur_drug_smiles, description_type, examples)
    res = {
        "instruction": CACO_INSTRUCTION_TEMPLATE,
        "input": prompt_body,
        "output": normalize(cur_drug_affinity, MIN_VALUE, MAX_VALUE)
    }
    return res


def generate_formatted_prompt_no_context(cur_drug_id, cur_drug_smiles, cur_drug_affinity, ref_list,
                                         description_type=None, no_context=False):
    examples = get_examples(ref_list, description_type)
    prompt_body = get_prompt_body(cur_drug_id, cur_drug_smiles, description_type, examples, no_context)
    res = {
        "instruction": CACO_INSTRUCTION_TEMPLATE,
        "input": prompt_body,
        "output": normalize(cur_drug_affinity, MIN_VALUE, MAX_VALUE)
    }
    return res


# raw_title_dict = {0: "caco2-0-shot.json", 1: "caco2-1-shot-similarity.json", 2: "caco2-2-shot-similarity.json",
#                   3: "caco2-3-shot-similarity.json", 4: "caco2-3-shot-similarity.json",
#                   5: "caco2-3-shot-similarity.json"}
raw_title_dict = {6: "caco2-6-shot.json", 7: "caco2-7-shot-similarity.json", 8: "caco2-8-shot-similarity.json",
                  9: "caco2-9-shot-similarity.json", 10: "caco2-10-shot-similarity.json"}

# raw_title_dict_random = {1: "caco2-1-shot-random.json", 2: "caco2-2-shot-random.json",
#                          3: "caco2-3-shot-random.json", 4: "caco2-4-shot-random.json",
#                          5: "caco2-5-shot-random.json"}
raw_title_dict_random = {6: "caco2-6-shot-random.json", 7: "caco2-7-shot-random.json",
                         8: "caco2-8-shot-random.json", 9: "caco2-9-shot-random.json",
                         10: "caco2-10-shot-random.json"}

# ruleof5_title_dict = {0: "caco2-0-shot-ruleof5.json", 1: "caco2-1-shot-similarity-ruleof5.json",
#                       2: "caco2-2-shot-similarity-ruleof5.json", 3: "caco2-3-shot-similarity-ruleof5.json",
#                       4: "caco2-4-shot-similarity-ruleof5.json", 5: "caco2-5-shot-similarity-ruleof5.json"}
ruleof5_title_dict = {6: "caco2-6-shot-ruleof5.json", 7: "caco2-7-shot-similarity-ruleof5.json",
                      8: "caco2-8-shot-similarity-ruleof5.json", 9: "caco2-9-shot-similarity-ruleof5.json",
                      10: "caco2-10-shot-similarity-ruleof5.json"}

# raw_title_dict_no_context = {0: "caco2-0-sho-no-context.json", 1: "caco2-1-shot-similarity-no-context.json",2: "caco2-2-shot-similarity-no-context.json",
#                   3: "caco2-3-shot-similarity-no-context.json", 4: "caco2-4-shot-similarity-no-context.json",
#                   5: "caco2-5-shot-similarity-no-context.json"}
raw_title_dict_no_context = {6: "caco2-6-sho-no-context.json", 7: "caco2-7-shot-similarity-no-context.json",
                             8: "caco2-8-shot-similarity-no-context.json",
                             9: "caco2-9-shot-similarity-no-context.json",
                             10: "caco2-10-shot-similarity-no-context.json"}
ruleof5_title_dict_no_context = {0: "caco2-0-shot-ruleof5-no-context.json",
                                 1: "caco2-1-shot-similarity-ruleof5-no-context.json",
                                 2: "caco2-2-shot-similarity-ruleof5-no-context.json",
                                 3: "caco2-3-shot-similarity-ruleof5-no-context.json",
                                 4: "caco2-4-shot-similarity-ruleof5-no-context.json",
                                 5: "caco2-5-shot-similarity-ruleof5-no-context.json"}
use_description = True
random_sample = True
no_context = False

if use_description:
    title_dict = ruleof5_title_dict
    description_type = "rule_of_5"
    print("right")
else:
    description_type = None
    if random_sample:

        title_dict = raw_title_dict_random
    else:
        title_dict = raw_title_dict
if no_context:
    title_dict = raw_title_dict_no_context  # ruleof5_title_dict_no_context

for split in ["train", "valid", "test"]:  # ["test"]:  #
    for n, file_name in title_dict.items():
        prompt_dict = {"train": [], "valid": [], "test": []}
        split_df = splits[split]
        split_neighbor = neighbor_dict[split]

        assert split_neighbor["smiles"] == split_df["Drug"].to_list()

        for idx, row in split_df.iterrows():
            ref_list = []
            if n > 0:
                if random_sample:
                    random_neighbor_idx = get_random_index(n, 0, split_df.shape[0] - 1, [idx])
                    for neighbor_idx in random_neighbor_idx:
                        neighbor_smiles = split_neighbor["smiles"][neighbor_idx]
                        neighbor_id = split_df["Drug_ID"].to_list()[neighbor_idx]
                        neighbor_affinity = split_df["Y"].to_list()[neighbor_idx]
                        ref_list.append([neighbor_idx, neighbor_smiles, neighbor_id, neighbor_affinity])
                else:
                    for j in range(n):
                        neighbor_idx = split_neighbor["neighbors"][idx][j][0]
                        neighbor_smiles = split_neighbor["smiles"][neighbor_idx]
                        neighbor_id = split_df["Drug_ID"].to_list()[neighbor_idx]
                        neighbor_affinity = split_df["Y"].to_list()[neighbor_idx]
                        ref_list.append([neighbor_idx, neighbor_smiles, neighbor_id, neighbor_affinity])
            cur_drug_id = row["Drug_ID"]
            cur_drug_smiles = row["Drug"]
            cur_drug_affinity = row["Y"]
            if no_context:
                prompt_json = generate_formatted_prompt_no_context(cur_drug_id, cur_drug_smiles, cur_drug_affinity,
                                                                   ref_list,
                                                                   description_type, no_context=True)
            else:
                prompt_json = generate_formatted_prompt(cur_drug_id, cur_drug_smiles, cur_drug_affinity, ref_list,
                                                        description_type)
            prompt_dict[split].append(prompt_json)
        with open(f"/home/wangtian/codeSpace/LLM-finetuning-playground/data/{split}/{split}-{file_name}",
                  'w') as json_file:
            json.dump(prompt_dict[split], json_file)
        print(
            f"split: {split}, {n}-shot data saved at /home/wangtian/codeSpace/LLM-finetuning-playground/data/{split}/{split}-{file_name}")
        # print("done")
"""
# TODO: 

"""
