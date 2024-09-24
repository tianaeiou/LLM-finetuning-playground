# %%

""" data cleaning """
import pandas as pd

data = pd.read_csv("instruction_tuning/data/il23_monocycle_category.csv")
columns = list(data.columns)

not_empty_sgf = data[data['SGF (hr)'].notnull()]
not_empty_sif = data[data['SIF (hr)'].notnull()]

""" data split """
from sklearn.model_selection import train_test_split


def get_train_val_test(df, random_seed=2024, test_size1=0.2, test_size2=0.5):
    X_train, X_remaining = train_test_split(df, test_size=test_size1, random_state=random_seed)
    X_val, X_test = train_test_split(X_remaining, test_size=test_size2, random_state=random_seed)
    return X_train, X_val, X_test


not_empty_sif.loc[not_empty_sif['SIF (hr)'] > 24, 'SIF (hr)'] = 24
sgf_train, sgf_val, sgf_test = get_train_val_test(not_empty_sgf, 2024, test_size1=0.3, test_size2=0.6666667)
sif_train, sif_val, sif_test = get_train_val_test(not_empty_sif, 2024, test_size1=0.2, test_size2=0.5)
print(f"sgf: train_size:{sgf_train.shape[0]},val_size:{sgf_val.shape[0]},test_size:{sgf_test.shape[0]}")
print(f"sif: train_size:{sif_train.shape[0]},val_size:{sif_val.shape[0]},test_size:{sif_test.shape[0]}")
# sgf: train_size:143, val_size:18, test_size:18
# sif: train_size:358, val_size:45, test_size:45
sif_list = [sif_train, sif_val, sif_test]
sgf_list = [sgf_train, sgf_val, sgf_test]
data_list = [sif_list, sgf_list]
mode_list = ["train", "val", "test"]
# %%
""" generate prompt """
import json
import os
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski

SYSTEM_INSTRUCTION = "You are an AI assistant specializing in half life prediction for drug discovery. The user will ask you to predict the half life properties of a molecule. Please think step by step, provide well-considered predictions and follow instructions based on your knowledge."

EXAMPLE_TEMPLATE_TxLLM = """drug SMILES: {CUR_DRUG_SMILES}
answer: {CUR_ANSWER}
"""
EXAMPLE_TEMPLATE_TxLLM_PROPERTY = """drug SMILES: {CUR_DRUG_SMILES}
{CUR_DRUG_PROPERTIES}answer: {CUR_ANSWER}
"""

PROMPT_TEMPLATE_CLS_TxLLM = """Context: {CUR_CONTEXT_INFO}

Question: Predict the half-life category of a drug based on its SMILES string:
{LABEL_DESCRIPTION}

drug SMILES: {CUR_DRUG_SMILES}
answer: """

PROMPT_TEMPLATE_CLS_WITH_EXAMPLE_TxLLM = """Context: {CUR_CONTEXT_INFO}

Question: Predict the half-life category of a drug based on its SMILES string:
{LABEL_DESCRIPTION}
Examples: 
{CUR_EXAMPLES}
Now, using the information provided, predict the result for the following drug. IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
drug SMILES: {CUR_DRUG_SMILES}
"""

PROMPT_TEMPLATE_REG_WITH_EXAMPLE_TxLLM = """Context: {CUR_CONTEXT_INFO}

Question: Predict the half-life value of a drug based on its SMILES string, where the predicted values are integer hours within the range of 0 to 24.
{LABEL_DESCRIPTION}
Examples: 
{CUR_EXAMPLES}
Now, using the information provided, predict the result for the following drug. IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
drug SMILES: {CUR_DRUG_SMILES}
"""
INSTRUCTION = {
    "sif": "Predict the half-life duration of a substance when exposed to Simulated Intestinal Fluid conditions.",
    "sgf": "Predict the half-life duration of a substance when exposed to Simulated Gastric Fluid conditions."
}
CONTEXT = {
    "sif": "Simulated Intestinal Fluid (SIF) is a lab solution that mimics the human small intestine, used to study how nutrients and drugs are digested and absorbed. The time for a substance's concentration to halve in SIF, known as its half-life, indicates its stability and digestion rate.",
    "sgf": "Simulated Gastric Fluid (SGF) is a laboratory-created medium that replicates the acidic environment of the stomach. The half-life in SGF—the duration required for the compound's concentration to decrease by half—serves as a key measure of its gastric stability and degradation pace."}

QUESTION = {
    "sif": {"0": "Indicates a drug with a half life of 12 hours or less in SIF.",
            "1": "Indicates a drug with a half life between 12 to 24 hours in SIF.",
            "2": "Indicates a drug with a half life more than 24 hours in SIF."},
    "sgf": {"0": "Indicates a drug with a half life of 24 hours or less in SGF.",
            "1": "Indicates a drug with a half life more than 24 hours in SGF."}
}
tar_col_dict = {
    "cls": {"sif": "SIF Category", "sgf": "SGF Category"},
    "reg": {"sif": "SIF (hr)", "sgf": "SGF (hr)"}
}
PROMPT_TEMPLATE_REG_WITH_EXAMPLE_PROPERTY, PROMPT_TEMPLATE_REG, PROMPT_TEMPLATE_CLS_WITH_EXAMPLE_TxLLM_PROPERTY = "", "", ""


def calculate_similarity(data_df, ref_df, file_dir, N=10):
    # Check if the result file exists
    if os.path.isfile(file_dir):
        with open(file_dir, 'rb') as file:
            try:
                neighbor_dict = pickle.load(file)
                print("Neighbor info loaded!")
                return neighbor_dict
            except Exception as e:
                print(f"Failed to load neighbor info: {e}")

    # Calculate similarity
    smiles_list_test, smiles_list_ref = data_df["SMILES"].to_list(), ref_df["SMILES"].to_list()
    molecules_test = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_test]
    molecules_ref = [Chem.MolFromSmiles(smiles) for smiles in smiles_list_ref]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_test]
    fps_ref = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2) for m in molecules_ref]
    nearest_neighbors = {i: [] for i in range(len(smiles_list_test))}
    for i, fp1 in enumerate(fps_test):
        for j, fp2 in enumerate(fps_ref):
            if fp1 != fp2:
                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
                if len(nearest_neighbors[i]) < N:
                    nearest_neighbors[i].append((j, similarity))
                else:
                    nearest_neighbors[i].append((j, similarity))
                    nearest_neighbors[i].sort(key=lambda x: x[1], reverse=True)
                    nearest_neighbors[i] = nearest_neighbors[i][:N]
    result = {"smiles": smiles_list_test, "neighbor_index": nearest_neighbors}

    # Save to file_dir
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    with open(file_dir, 'wb') as f:
        pickle.dump(result, f)
    return result


def get_label_prompt(label_dict):
    label_prompt = ""
    if not label_dict:
        return label_prompt
    for label, description in label_dict.items():
        label_prompt += f"- {label}: {description}\n"
    return label_prompt


def get_example_prompt(cur_neighbor_dict, num_examples, data_index, ref_df, tar_col,
                       is_regression_task=True, example_template=EXAMPLE_TEMPLATE_TxLLM, use_property=False,
                       property_template=EXAMPLE_TEMPLATE_TxLLM_PROPERTY):
    example_prompt = ""
    for j in range(num_examples):
        neighbor_idx = cur_neighbor_dict["neighbor_index"][data_index][j][0]
        mol = Chem.MolFromSmiles(ref_df["SMILES"].to_list()[neighbor_idx])
        neighbor_smiles = Chem.MolToSmiles(mol)

        neighbor_target = int(ref_df[tar_col].to_list()[neighbor_idx])
        if use_property:
            # cur_example = property_template.format(CUR_ANSWER=neighbor_target, CUR_DRUG_SMILES=str(neighbor_smiles),
            #                                        CUR_DRUG_PROPERTIES=get_property_description(mol))
            pass
        else:
            cur_example = example_template.format(CUR_ANSWER=neighbor_target, CUR_DRUG_SMILES=str(neighbor_smiles))
        example_prompt += cur_example
    return example_prompt


def create_prompt_for_single_row(row, example_prompt, instruction_description, context_description, label_description,
                                 tar_col,
                                 min_value=0, max_value=1, is_regression_task=True,
                                 reg_template_with_example=PROMPT_TEMPLATE_REG_WITH_EXAMPLE_TxLLM,
                                 reg_template_with_example_property=PROMPT_TEMPLATE_REG_WITH_EXAMPLE_PROPERTY,
                                 reg_template=PROMPT_TEMPLATE_REG,
                                 cls_template_with_example=PROMPT_TEMPLATE_CLS_WITH_EXAMPLE_TxLLM,
                                 cls_template_with_example_property=PROMPT_TEMPLATE_CLS_WITH_EXAMPLE_TxLLM_PROPERTY,
                                 cls_template=PROMPT_TEMPLATE_CLS_TxLLM, use_property=False):
    mol = Chem.MolFromSmiles(row["SMILES"])

    property_description = ""

    if is_regression_task:
        if example_prompt:
            if use_property:
                # template = reg_template_with_example_property
                # property_description = get_property_description(mol)
                pass
            else:
                template = reg_template_with_example
        else:
            template = reg_template
        processed_output = str(row[tar_col])
        cur_input = template.format(CUR_CONTEXT_INFO=context_description, CUR_DRUG_SMILES=Chem.MolToSmiles(mol),
                                    CUR_EXAMPLES=example_prompt,
                                    LABEL_DESCRIPTION=label_description,
                                    CUR_DRUG_PROPERTIES=property_description)

    else:
        if example_prompt:
            if use_property:
                # template = cls_template_with_example_property
                # property_description = get_property_description(mol)
                pass
            else:
                template = cls_template_with_example
        else:
            template = cls_template

        processed_output = str(row[tar_col])
        cur_input = template.format(CUR_CONTEXT_INFO=context_description, CUR_DRUG_SMILES=Chem.MolToSmiles(mol),
                                    CUR_EXAMPLES=example_prompt,
                                    LABEL_DESCRIPTION=label_description,
                                    CUR_DRUG_PROPERTIES=property_description)
    return {
        "system": SYSTEM_INSTRUCTION,
        "instruction": instruction_description,
        "input": cur_input,
        "output": processed_output
    }


def process_dataset(data_df, ref_df, dataset_name, task_type, neighbor_info_dir, k, use_property):
    instruction_prompt = INSTRUCTION[dataset_name]
    context_prompt = CONTEXT[dataset_name]

    if task_type == "reg":
        label_descriptions = {}
        is_regression_task = True
    else:
        label_descriptions = QUESTION[dataset_name]
        is_regression_task = False

    neighbor_dict = calculate_similarity(data_df, ref_df, neighbor_info_dir)

    prompts_list = []
    tar_col = tar_col_dict[task_type][dataset_name]
    label_prompt = get_label_prompt(label_descriptions)
    for idx, row in data_df.iterrows():
        example_prompt = get_example_prompt(neighbor_dict, k, idx, ref_df, tar_col,
                                            is_regression_task=is_regression_task, use_property=use_property)

        cur_prompt = create_prompt_for_single_row(row, example_prompt, instruction_prompt, context_prompt, label_prompt,
                                                  tar_col,
                                                  is_regression_task=is_regression_task,
                                                  use_property=use_property)
        prompts_list.append(cur_prompt)
    return prompts_list


project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"
save_dir = f"{project_path}/instruction_tuning/instructions"
dataset_list = ["sif", "sgf"]
task_type = "reg"  # "cls"
sif_list = [sif_train, sif_val, sif_test]
sgf_list = [sgf_train, sgf_val, sgf_test]
data_list = [sif_list, sgf_list]
use_property = False
k = 5

for i in range(2):
    dataset = dataset_list[i]
    df_list = data_list[i]
    for j in range(3):
        mode = mode_list[j]
        data_df = df_list[j]
        data_df = data_df.reset_index()
        ref_df = df_list[0]
        print(f"Dataset: {dataset}, Mode: {mode}, Size: {len(data_df)}")
        file_name = f"{dataset}_instruction-{k}-shot-{mode}-{task_type}.json"
        neighbor_info_dir = f"{project_path}/instruction_tuning/neighbor_info/{dataset}-neighbors-{mode}-2024.pkl"
        prompts_list = process_dataset(data_df, ref_df, dataset, task_type, neighbor_info_dir, k=k,
                                       use_property=use_property)
        with open(f"{save_dir}/{file_name}", 'w') as json_file:
            json.dump(prompts_list, json_file)
        print(f"{save_dir}/{file_name}, Done! Length: {len(prompts_list)}")


print(prompts_list[0]["input"])

# %% evaluate
""" classification """
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# use seaborn to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()  # figsize=(10, 10)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()


with open("instruction_tuning/result/hl/Llama3-sgf_instruction-5-shot-test.json", "r") as f:
    results = json.load(f)

pred_values = [int(float(f)) for f in results["pred_values"]]
real_values = [int(float(f)) for f in results["real_values"]]

plot_confusion_matrix(real_values, pred_values, "Confusion Matrix of Llama3-8B")

from sklearn.metrics import accuracy_score, f1_score

overall_accuracy = accuracy_score(real_values, pred_values)
f1_per_class = f1_score(real_values, pred_values, average=None)
print(f"Overall Accuracy: {overall_accuracy}")
print(f"F1 Score per Class: {f1_per_class}")
#%% evaluate

""" regression """
from scipy.stats import pearsonr, spearmanr
import json
import seaborn as sns
import matplotlib.pyplot as plt
def plot_scatter_plot(x, y, title, x_label, y_label, hue=None):
    fig, ax = plt.subplots()# figsize=(8, 8)
    # make the dot larger and more transparent
    # sns.scatterplot(x=x, y=y, ax=ax, s=10, alpha=0.5, legend=None)  # hue=hue,
    sns.scatterplot(x=x, y=y, ax=ax, s=50,  legend=None)
    sns.regplot(x=x, y=y, scatter=False, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    try:
        pearson_corr = pearsonr(x, y)[0]
    except:
        pearson_corr = 0.0

    try:
        spearman_corr = spearmanr(x, y)[0]
    except:
        spearman_corr = 0.0

    ax.set_title(title + "\n Pearson: {:.2f}, Spearman: {:.2f}".format(
        pearson_corr, spearman_corr))
    plt.tight_layout()
    plt.show()
    return



with open("instruction_tuning/result/hl/outbestmodel/sif_instruction-5-shot-test-reg.json","r") as f:
    result = json.load(f)

pred_values = [float(f) for f in result["pred_values"]]
real_values = [float(f) for f in result["real_values"]]

plot_scatter_plot(pred_values, real_values, "ADMET SFT Llama3 Prediction of Half Life on in SIF", "pred values", "real_values", hue=None)

#%% evaluate: use regression result to perform classification task

import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
with open("instruction_tuning/result/hl/reg+cls-20epoch/sgf_instruction-5-shot-test-reg.json","r") as f:
    reg_result = json.load(f)
with open("instruction_tuning/result/hl/reg+cls-20epoch/sgf_instruction-5-shot-test.json","r") as f:
    cls_result = json.load(f)

model_name = "regcls20-epoch reg2cls"
def convert_sif_reg2cls(reg_result):
    cls_result = []
    for v in reg_result["pred_values"]:
        if float(v)<12:
            cls_result.append(0)
        elif float(v)<24:
            cls_result.append(1)
        else:
            cls_result.append(2)
    return cls_result

def convert_sgf_reg2cls(reg_result):
    cls_result = []
    for v in reg_result["pred_values"]:

        if float(v)<24:
            cls_result.append(0)
        else:
            cls_result.append(1)
    return cls_result

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()  # figsize=(10, 10)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()

pred_values = convert_sgf_reg2cls(reg_result)
real_values = [int(float(f)) for f in cls_result["real_values"]]
plot_confusion_matrix(real_values, pred_values, f"Confusion Matrix of {model_name}")

overall_accuracy = accuracy_score(real_values, pred_values)
f1_per_class = f1_score(real_values, pred_values, average=None)
print(f"Overall Accuracy: {overall_accuracy}")
print(f"F1 Score per Class: {f1_per_class}")
# use seaborn to plot the confusion matrix


