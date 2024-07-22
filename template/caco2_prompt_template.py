CACO_INSTRUCTION_TEMPLATE = """ Predict the Caco-2 cell effective permeability of the given drug. """

ZERO_SHOT_TEMPLATE_NO_CONTEXT = """Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000 where 0 is the minimum effective permeability and 1000 is maximum effective permeability. The predicted value must be an integer with no decimals.

```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

ZERO_SHOT_TEMPLATE = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000 where 0 is the minimum effective permeability and 1000 is maximum effective permeability. The predicted value must be an integer with no decimals.

```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""


ZERO_SHOT_TEMPLATE_WITH_DESCRIPTION_NO_CONTEXT = """Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000 where 0 is the minimum effective permeability and 1000 is maximum effective permeability. The predicted value must be an integer with no decimals.

```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_DESCRIPTION
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

ZERO_SHOT_TEMPLATE_WITH_DESCRIPTION = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000 where 0 is the minimum effective permeability and 1000 is maximum effective permeability. The predicted value must be an integer with no decimals.

```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_DESCRIPTION
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_TEMPLATE_NO_CONTEXT = """Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_TEMPLATE = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_TEMPLATE_WITH_DESCRIPTION_NO_CONTEXT = """Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_DESCRIPTION
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

FEW_SHOT_TEMPLATE_WITH_DESCRIPTION = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug id string and the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug id": CUR_DRUG_ID
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_DESCRIPTION
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

EXAMPLE_TEMPLATE = """
```
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
    CUR_DRUG_DESCRIPTION
    "answer": {CUR_ANSWER}
}}
```
"""

RULEOF5_TEMPLATE = """"Rule of Five":
        {{
            "MW": {MW}
            "CLogP": {CLogP}
            "HBA": {HBA}
            "HBD": {HBD}
            "RB": {RB}
            "TPSA": {TPSA}
        }}"""

DESCRIPTION_TEMPLATE_DICT = {"rule_of_5": RULEOF5_TEMPLATE}
