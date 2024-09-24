# %%
""" KNN |  MACCS + hamming距离 | 08.01 """
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from tdc.benchmark_group import admet_group
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

CLASSIFICATION_TASKS = ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp2d6_veith',
                        'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels',
                        'cyp3a4_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'herg', 'ames', 'dili']


def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    # 将MACCS keys转换为布尔数组，167位中第0位是占位符，我们只取后166位
    return [int(bit) for bit in maccs.ToBitString()[1:]]


def hamming_distance(x1, x2):
    return np.sum(x1 != x2)


group = admet_group(path='data/')
predictions = {}
k = 5
for dataset_name in CLASSIFICATION_TASKS:
    benchmark = group.get(dataset_name)
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    train_smiles = train_val['Drug'].to_list()
    test_smiles = test["Drug"].to_list()
    X_train = np.array([smiles_to_maccs(smiles) for smiles in train_smiles])
    X_text = np.array([smiles_to_maccs(smiles) for smiles in test_smiles])
    knn = KNeighborsClassifier(n_neighbors=k, metric=hamming_distance, algorithm='brute')
    knn.fit(X_train, train_val['Y'])
    predicted_labels = knn.predict(X_text)
    predictions[dataset_name] = [int(v) for v in predicted_labels]
print(group.evaluate(predictions))
# %%
"""
k = 10
{'hia_hou': {'roc-auc': 0.722}, 'pgp_broccatelli': {'roc-auc': 0.767}, 'bioavailability_ma': {'roc-auc': 0.465}, 'bbb_martins': {'roc-auc': 0.766}, 'cyp2d6_veith': {'pr-auc': 0.311}, 'cyp3a4_veith': {'pr-auc': 0.619}, 'cyp2c9_veith': {'pr-auc': 0.489}, 'cyp2d6_substrate_carbonmangels': {'pr-auc': 0.409}, 'cyp3a4_substrate_carbonmangels': {'roc-auc': 0.594}, 'cyp2c9_substrate_carbonmangels': {'pr-auc': 0.281}, 'herg': {'roc-auc': 0.655}, 'ames': {'roc-auc': 0.701}, 'dili': {'roc-auc': 0.73}}
k = 5
{'hia_hou': {'roc-auc': 0.722}, 'pgp_broccatelli': {'roc-auc': 0.795}, 'bioavailability_ma': {'roc-auc': 0.525}, 'bbb_martins': {'roc-auc': 0.768}, 'cyp2d6_veith': {'pr-auc': 0.325}, 'cyp3a4_veith': {'pr-auc': 0.609}, 'cyp2c9_veith': {'pr-auc': 0.5}, 'cyp2d6_substrate_carbonmangels': {'pr-auc': 0.437}, 'cyp3a4_substrate_carbonmangels': {'roc-auc': 0.633}, 'cyp2c9_substrate_carbonmangels': {'pr-auc': 0.299}, 'herg': {'roc-auc': 0.683}, 'ames': {'roc-auc': 0.721}, 'dili': {'roc-auc': 0.751}}
k = 2
"""
# %%
# visualization

# %%
# neighbors_indices, distances = knn.kneighbors(X_text, return_distance=True)
# for i, (sample_neighbors, sample_distances) in enumerate(zip(neighbors_indices, distances)):
#     print(f"Sample {i} predicted label: {predicted_labels[i]}")
#     print(f"Neighbors and distances for sample {i}: {list(zip(sample_neighbors, sample_distances))}")
#     break
# %%
""" 直接加载 neighbor_dict 将大多数作为结果！"""
""" 08.01 """
from tdc.benchmark_group import admet_group
import pickle

CLASSIFICATION_TASKS = ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp2d6_veith',
                        'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels',
                        'cyp3a4_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'herg', 'ames', 'dili']
project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"
mode = "test"
k = 5

group = admet_group(path='data/')
predictions = {}
for dataset_name in CLASSIFICATION_TASKS:
    print(dataset_name)
    benchmark = group.get(dataset_name)  # group.get(sys.argv[1])
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    train_smiles = train_val['Drug'].to_list()
    test_smiles = test["Drug"].to_list()

    neighbor_info_dir = f"{project_path}/instruction_tuning/neighbor_info/{dataset_name}-neighbors-{mode}.pkl"
    with open(neighbor_info_dir, 'rb') as file:
        neighbor_dict = pickle.load(file)
    preds = []
    for i in range(len(test_smiles)):
        neighbor_idxs = [neighbor_dict["neighbor_index"][i][j][0] for j in range(k)]
        preds.append(train_val.iloc[neighbor_idxs]["Y"].value_counts().idxmax())
    predictions[dataset_name] = preds

print(group.evaluate(predictions))
# %%
"""
k = 10
{'hia_hou': {'roc-auc': 0.704}, 'pgp_broccatelli': {'roc-auc': 0.779}, 'bioavailability_ma': {'roc-auc': 0.522}, 'bbb_martins': {'roc-auc': 0.753}, 'cyp2d6_veith': {'pr-auc': 0.358}, 'cyp3a4_veith': {'pr-auc': 0.646}, 'cyp2c9_veith': {'pr-auc': 0.534}, 'cyp2d6_substrate_carbonmangels': {'pr-auc': 0.48}, 'cyp3a4_substrate_carbonmangels': {'roc-auc': 0.607}, 'cyp2c9_substrate_carbonmangels': {'pr-auc': 0.284}, 'herg': {'roc-auc': 0.599}, 'ames': {'roc-auc': 0.74}, 'dili': {'roc-auc': 0.744}}
k = 5
{'hia_hou': {'roc-auc': 0.704}, 'pgp_broccatelli': {'roc-auc': 0.779}, 'bioavailability_ma': {'roc-auc': 0.522}, 'bbb_martins': {'roc-auc': 0.753}, 'cyp2d6_veith': {'pr-auc': 0.358}, 'cyp3a4_veith': {'pr-auc': 0.646}, 'cyp2c9_veith': {'pr-auc': 0.534}, 'cyp2d6_substrate_carbonmangels': {'pr-auc': 0.48}, 'cyp3a4_substrate_carbonmangels': {'roc-auc': 0.607}, 'cyp2c9_substrate_carbonmangels': {'pr-auc': 0.284}, 'herg': {'roc-auc': 0.599}, 'ames': {'roc-auc': 0.74}, 'dili': {'roc-auc': 0.744}}"
"""
# %%
""" load result of current best-model"""
# KNN
dataset_name = "dili"
project_path = "/home/wangtian/codeSpace/LLM-finetuning-playground"
mode = "test"

benchmark = group.get(dataset_name)
name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test']
test_smiles = test["Drug"].to_list()
neighbor_info_dir = f"{project_path}/instruction_tuning/neighbor_info/{dataset_name}-neighbors-{mode}.pkl"
with open(neighbor_info_dir, 'rb') as file:
    neighbor_dict = pickle.load(file)
preds = []
for i in range(len(test_smiles)):
    neighbor_idxs = [neighbor_dict["neighbor_index"][i][j][0] for j in range(k)]
    preds.append(train_val.iloc[neighbor_idxs]["Y"].value_counts().idxmax())

tar = test["Y"].to_list()

"""analyze KNN & """
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title + "\n Accuracy: {:.3f}".format(acc))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()
    return


plot_confusion_matrix(tar, preds, "5-shot KNN")

import json

result_file = "/home/wangtian/codeSpace/LLM-finetuning-playground/instruction_tuning/result/cls_v1/5-shot/ADMET_reg_clsv1_all_five_shot_checkpoint-500/dili_instruction-5-shot-test.json"
with open(f"{result_file}", "r") as f:
    result = json.load(f)
preds_our = [float(f) for f in result["pred_values"]]
plot_confusion_matrix(tar, preds_our, "reg+clsv1 5-shot")
# %%
import pandas as pd
df = pd.DataFrame({"tar":tar,"KNN":preds,"ours":preds_our})
df["KNN_correct"] = df["KNN"] == df["tar"]
df["our_correct"] = df["ours"] == df["tar"]

#%%
k=5
neighbor_info_sum = []
neighbor_0 = []
neighbor_1 = []
neighbor_2 = []
neighbor_3 = []
neighbor_4 = []

for idx in range(len(tar)):
    neighbor_idxs = [neighbor_dict["neighbor_index"][idx][j][0] for j in range(k)]
    neighbor_info_sum.append(sum(train_val.iloc[neighbor_idxs]["Y"]))
    neighbor_0.append(train_val.iloc[neighbor_idxs[0]]["Y"])
    neighbor_1.append(train_val.iloc[neighbor_idxs[1]]["Y"])
    neighbor_2.append(train_val.iloc[neighbor_idxs[2]]["Y"])
    neighbor_3.append(train_val.iloc[neighbor_idxs[3]]["Y"])
    neighbor_4.append(train_val.iloc[neighbor_idxs[4]]["Y"])

df["neighbor_cls_sum"] = neighbor_info_sum
df["neighbor0"] = neighbor_0
df["neighbor1"] = neighbor_1
df["neighbor2"] = neighbor_2
df["neighbor3"] = neighbor_3
df["neighbor4"] = neighbor_4

selected_data= df[(df['KNN_correct'] == True) & (df['our_correct'] == False)]
selected_data1 = df[(df['KNN_correct'] == False) & (df['our_correct'] == True)]
selected_data2 = df[(df['KNN_correct'] == False) & (df['our_correct'] == False)]
selected_data3 = df[(df['neighbor_cls_sum'] == 5) & (df['our_correct'] == True)]
# 找到这些index,看他们的neighbor情况
# %%
# adapted from : https://github.com/smu-tao-group/ADMET_XGBoost/blob/main/src/featurize.py
