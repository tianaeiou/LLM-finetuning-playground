CPDS_PK_INSTRUCTION_TEMPLATE = "Predict the pharmacokinetic parameter for the given drug."

CPDS_PK_CONTEXT_TEMPLATE = "The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes."

ZERO_SHOT_QUESTION_TEMPLATE = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id string and the drug SMILES string, predict the normalized CUR_TARGET from 0 to 1000, where 0 is the minimum CUR_TARGET and 1000 is the maximum. The predicted value must be an integer with no decimals.
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

ZERO_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id, the drug SMILES string and physiochemical properties, predict the normalized CUR_TARGET from 0 to 1000, where 0 is the minimum CUR_TARGET and 1000 is the maximum. The predicted value must be an integer with no decimals.
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_QUESTION_TEMPLATE = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id string and the drug SMILES string, predict the normalized CUR_TARGET from 0 to 1000, where 0 is the minimum CUR_TARGET and 1000 is the maximum. The predicted value must be an integer with no decimals.
Examples:
CUR_EXAMPLESNow, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id, the drug SMILES string and physiochemical properties, predict the normalized CUR_TARGET from 0 to 1000, where 0 is the minimum CUR_TARGET and 1000 is the maximum. The predicted value must be an integer with no decimals.
Examples:
CUR_EXAMPLESNow, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

EXAMPLE_TEMPLATE = """```
{{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
    "answer": {CUR_ANSWER}
}}
```
"""

EXAMPLE_TEMPLATE_WITH_DESCRIPTION = """
```
{{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES    "answer": {CUR_ANSWER}
}}
```
"""

PROPERTIES_TEMPLATES = {
    "MW": """    "MW": {value}\n""",
    "HBA": """    "HBA": {value}\n""",
    "HBD": """    "HBD": {value}\n""",
    "TPSA_NO": """    "TPSA_NO": {value}\n""",
    "RotBondCount": """    "RotBondCount": {value}\n""",
    "moka_ionState7.4": """    "Ion State at pH 7.4": {value}\n""",
    "MoKa.LogP": """    Logarithm of Octanol-water Partition Coefficient": {value}\n""",
    "MoKa.LogD7.4": """    "Logarithm of Distribution at pH 7.4": {value}\n"""
}

# Result interpretation: >20%: High Fu; 5-20%: medium Fu; <5% low Fu.
ZERO_SHOT_QUESTION_TEMPLATE_CLS = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id string and the drug SMILES string, classify the drug based on the normalized CUR_TARGET. Predict one of the following three categories:
- '0' if the normalized CUR_TARGET value is greater than {THRESHOLD0}
- '1' if the CUR_TARGET value is between {THRESHOLD1} and {THRESHOLD0}
- '2' if the CUR_TARGET value is less than {THRESHOLD1}
```
{{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY_CLS = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id string and the drug SMILES string, classify the drug based on the normalized CUR_TARGET. Predict one of the following three categories:
- '0' if the normalized CUR_TARGET value is greater than {THRESHOLD0}
- '1' if the CUR_TARGET value is between {THRESHOLD1} and {THRESHOLD0}
- '2' if the CUR_TARGET value is less than {THRESHOLD1}
Examples:
CUR_EXAMPLES
Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
ZERO_SHOT_QUESTION_TEMPLATE_WITH_PROPERTY_CLS = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id string and the drug SMILES string, classify the drug based on the normalized CUR_TARGET. Predict one of the following three categories:
- '0' if the normalized CUR_TARGET value is greater than {THRESHOLD0}
- '1' if the CUR_TARGET value is between {THRESHOLD1} and {THRESHOLD0}
- '2' if the CUR_TARGET value is less than {THRESHOLD1}
```
{{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
FEW_SHOT_QUESTION_TEMPLATE_CLS = """Context: The efficacy of a pharmaceutical compound can vary significantly across patient populations, highlighting the need for tailored therapeutics. Predicting pharmacokinetic parameters such as the volume of distribution at steady-state (human VDss), clearance rate (human CL), fraction unbound in plasma (fu), and terminal half-life (terminal t1/2) is crucial for customizing dosage and optimizing outcomes.
Question: Given the drug id string and the drug SMILES string, classify the drug based on the normalized CUR_TARGET. Predict one of the following three categories:
- '0' if the normalized CUR_TARGET value is greater than {THRESHOLD0}
- '1' if the CUR_TARGET value is between {THRESHOLD1} and {THRESHOLD0}
- '2' if the CUR_TARGET value is less than {THRESHOLD1}
Examples:
CUR_EXAMPLES
Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""