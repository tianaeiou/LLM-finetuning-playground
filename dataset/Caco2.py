# Caco-2： https://tdcommons.ai/single_pred_tasks/adme/#caco-2-cell-effective-permeability-wang-et-al
#%%
import pickle
import json
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


def calculate_similarity(splits, save_path="/home/wangtian/codeSpace/LLM-finetuning-playground/data/Caco-smiles-neighbors.pkl"):
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

CACO_INSTRUCTION_TEMPLATE = """ Predict the Caco-2 cell effective permeability of the given drug. """
ZERO_SHOT_TEMPLATE = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000 where 0 is the minimum effective permeability and 1000 is maximum effective permeability. The predicted value should be returned as a JSON object with the key "answer". Format: {"answer": predicted_value}

- drug id: CUR_DRUG_ID
- drug SMILES: CUR_DRUG_SMILES

Please provide your prediction in the specified JSON format.
"""
FEW_SHOT_TEMPLATE = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value should be returned as a JSON object with the key "answer". You only need to return the predicted value in the following format: {"answer": predicted_value}.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
- drug id: CUR_DRUG_ID
- drug SMILES: CUR_DRUG_SMILES

Please provide your prediction in the specified JSON format.

"""


def get_prompt_body(cur_drug_id, cur_drug_smiles, examples=None):
    if not examples:
        return ZERO_SHOT_TEMPLATE.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES", cur_drug_smiles)
    else:
        return FEW_SHOT_TEMPLATE.replace("CUR_DRUG_ID", cur_drug_id).replace("CUR_DRUG_SMILES",
                                                                             cur_drug_smiles).replace("CUR_EXAMPLES",
                                                                                                       examples)
def get_examples(ref_list):
    if len(ref_list) == 0:
        return None
    example = """"""
    few_shot_template = """
    - drug id: CUR_DRUG_ID
    - drug SMILES: CUR_DRUG_SMILES
    {"answer":"""
    for _, [_, neighbor_smiles, neighbor_id, neighbor_affinity] in enumerate(ref_list):
        example += few_shot_template.replace("CUR_DRUG_ID", neighbor_id).replace("CUR_DRUG_SMILES",
                                                                                 neighbor_smiles) + str(
            neighbor_affinity) + "}" + "\n"
    return example

def generate_formatted_prompt(cur_drug_id, cur_drug_smiles, cur_drug_affinity, ref_list):
    examples = get_examples(ref_list)
    prompt_body = get_prompt_body(cur_drug_id, cur_drug_smiles, examples)
    res = {
        "instruction": CACO_INSTRUCTION_TEMPLATE,
        "input": prompt_body,
        "output": cur_drug_affinity
    }
    return res


prompt_dict = {"train": [], "valid": [], "test": []}
title_dict = {0: "caco2-0-shot.json", 1: "caco2-1-shot-similarity.json", 5: "caco2-5-shot-similarity.json",
              10: "caco2-10-shot-similarity.json"}
for split in ["train", "valid", "test"]:  # ["test"]:  #
    for n, file_name in title_dict.items():

        split_df = splits[split]
        split_neighbor = neighbor_dict[split]

        assert split_neighbor["smiles"] == split_df["Drug"].to_list()

        for idx, row in split_df.iterrows():
            ref_list = []
            for j in range(n):
                neighbor_idx = split_neighbor["neighbors"][idx][j][0]
                neighbor_smiles = split_neighbor["smiles"][neighbor_idx]
                neighbor_id = split_df["Drug_ID"].to_list()[neighbor_idx]
                neighbor_affinity = split_df["Y"].to_list()[neighbor_idx]
                ref_list.append([neighbor_idx, neighbor_smiles, neighbor_id, neighbor_affinity])
            cur_drug_id = row["Drug_ID"]
            cur_drug_smiles = row["Drug"]
            cur_drug_affinity = row["Y"]
            prompt_json = generate_formatted_prompt(cur_drug_id, cur_drug_smiles, cur_drug_affinity, ref_list)
            prompt_dict[split].append(prompt_json)

        with open(f"/home/wangtian/codeSpace/LLM-finetuning-playground/data/{split}/{file_name}", 'w') as json_file:
            json.dump(prompt_dict[split], json_file)
        print(f"split: {split}, {n}-shot data saved at /home/wangtian/codeSpace/LLM-finetuning-playground/data/{split}/{file_name}")

#%%
# generate data for rule of 5
from rdkit.Chem import Descriptors,Lipinski
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

