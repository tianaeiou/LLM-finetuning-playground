import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski
from .prompt_template import *


def get_min_max_value(ref_df, target_col="Y"):
    """get min and max values of target column, used for normalization

    :param ref_df:
    :param target_col:
    :return:
    """
    min = ref_df[target_col].min()
    max = ref_df[target_col].max()
    return min, max


def normalize(value, min_value, max_value, min_output=0.0, max_output=1000.0):
    """for prompt generation, normalize the target value from [min_value, max_value] to [min_output, max_output]

    :param value:
    :param min_value:
    :param max_value:
    :param min_output:
    :param max_output:
    :return:
    """
    normalized_value = (value - min_value) * (max_output - min_output) / (max_value - min_value) + min_output
    normalized_value = min(max_output, max(normalized_value, min_value))
    return round(normalized_value, 2)


def denormalize(normalized_value, min_value, max_value, min_output=0.0, max_output=1000.0):
    """for performance evaluation, converting normalized value from [min_output, max_output] to [min_value, max_value]

    :param normalized_value:
    :param min_value:
    :param max_value:
    :param min_output:
    :param max_output:
    :return:
    """
    value = (normalized_value - min_output) * (max_value - min_value) / (max_output - min_output) + min_value
    return value


def get_property_values(smiles, properties=None, digit=3):
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

    property_dict = {
        "MW": round(Descriptors.MolWt(mol), digit),
        "CLogP": round(Descriptors.MolLogP(mol), digit),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RB": Lipinski.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), digit)
    }

    return property_dict


def get_property_prompt(cur_drug_property_values, properties):
    property_prompt = ""
    for property in properties:
        property_prompt += PROPERTIES_TEMPLATES[property].format(value=cur_drug_property_values[property])
    return property_prompt


def get_mae(pred, real):
    pred = torch.tensor(pred)
    real = torch.tensor(real)
    mae = torch.mean(torch.abs(real - pred))
    return {"mae": mae.item()}
