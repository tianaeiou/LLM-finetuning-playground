""" Construction Mini Trainset """
import json
import random
with open("instruction_tuning/instructions/regression_all-5-shot.json","r") as f:
    dataset = json.load(f)

sample_size = int(len(dataset) * 0.1)
sampled_data = random.sample(dataset, sample_size)
with open("instruction_tuning/instructions/mini_regression_all-5-shot.json", 'w') as file:
    json.dump(sampled_data, file)
print(len(dataset),len(sampled_data))


#%%
import sys
import json
import os
from tdc.benchmark_group import admet_group
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
sys.path.append("..")

with open("instruction_tuning/train_evaluation/ADMET/reg_clsv1_on_data_reg5shot.json","r") as f:
    results = json.load(f)

pred_values = [float(f) for f in results["pred_values"]]
real_values = [float(f) for f in results["real_values"]]


def plot_scatter_plot(x, y, title, x_label, y_label, hue=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    # make the dot larger and more transparent
    sns.scatterplot(x=x, y=y, ax=ax, s=10, alpha=0.5, legend=None)  # hue=hue,
    sns.regplot(x=x, y=y, scatter=False, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # relocate the legend, make it smaller, make it outside the plot
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=4)

    # set the title, and add pearson and spearman in the title
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
    print("show")
    return
plot_scatter_plot(pred_values,real_values,"Performance on Train Set (reg_clsv1 5-shot)","pred","real")