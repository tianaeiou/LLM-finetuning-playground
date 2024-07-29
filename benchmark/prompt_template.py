PROPERTIES_DICT = {"ruleof5": ["MW", "CLogP", "HBA", "HBD", "RB", "TPSA"]}

PROPERTIES_TEMPLATES = {
    "MW": """    "MW": {value}\n""",
    "CLogP": """    "CLogP": {value}\n""",
    "HBA": """    "HBA": {value}\n""",
    "HBD": """    "HBD": {value}\n""",
    "RB": """    "RB": {value}\n""",
    "TPSA": """    "TPSA": {value}\n""",
    "TPSA_NO": """    "TPSA_NO": {value}\n""",
    "RotBondCount": """    "RotBondCount": {value}\n""",
    "moka_ionState7.4": """    "Ion State at pH 7.4": {value}\n""",
    "MoKa.LogP": """    Logarithm of Octanol-water Partition Coefficient": {value}\n""",
    "MoKa.LogD7.4": """    "Logarithm of Distribution at pH 7.4": {value}\n"""
}

EXAMPLE_TEMPLATE = """
```
{{
    "drug SMILES": CUR_DRUG_SMILES
    "answer": {CUR_ANSWER}
}}
```
"""

EXAMPLE_TEMPLATE_WITH_PROPERTIES = """
```
{{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES    "answer": {CUR_ANSWER}
}}
```
"""

BBB_INSTRUCTION_TEMPLATE = "Predict whether the compound can penetrate the Blood-Brain Barrier."
BBB_FEW_SHOT_TEMPLATE_WITH_PROPERTY_v0 = """Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system From MoleculeNet.

Question: Given the drug SMILES string, determine whether the compound can or cannot penetrate the blood-brain barrier. Predict one of the following categories:
- 0 if the compound cannot effectively penetrate the blood-brain barrier
- 1 if the compound is capable of penetrating the blood-brain barrier

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
BBB_FEW_SHOT_TEMPLATE_v0 = """Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system From MoleculeNet.

Question: Given the drug SMILES string, determine whether the compound can or cannot penetrate the blood-brain barrier. Predict one of the following categories:
- 0 if the compound cannot effectively penetrate the blood-brain barrier
- 1 if the compound is capable of penetrating the blood-brain barrier

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
BBB_FEW_SHOT_TEMPLATE_WITH_PROPERTY_v1 = """Context: The Blood-Brain Barrier (BBB) is a selective barrier that limits the passage of substances into the brain, crucial for protecting the central nervous system. Effective BBB penetration is a key determinant for drugs targeting neurological conditions, as it allows them to reach their therapeutic sites within the brain.

Question: Given the drug SMILES string, determine whether the compound can or cannot penetrate the blood-brain barrier. Predict one of the following categories:
- 0 if the compound cannot effectively penetrate the blood-brain barrier
- 1 if the compound is capable of penetrating the blood-brain barrier

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
BBB_FEW_SHOT_TEMPLATE_v1 = """Context: The Blood-Brain Barrier (BBB) is a selective barrier that limits the passage of substances into the brain, crucial for protecting the central nervous system. Effective BBB penetration is a key determinant for drugs targeting neurological conditions, as it allows them to reach their therapeutic sites within the brain.

Question: Given the drug SMILES string, determine whether the compound can or cannot penetrate the blood-brain barrier. Predict one of the following categories:
- 0 if the compound cannot effectively penetrate the blood-brain barrier
- 1 if the compound is capable of penetrating the blood-brain barrier

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
BBB_FEW_SHOT_TEMPLATE_WITH_PROPERTY_v2 = """Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.

Question: Given the drug SMILES string, determine the likelihood of the compound penetrating the BBB. Classify the prediction as
- 0: The compound is not expected to effectively penetrate the BBB.
- 1: The compound is likely to penetrate the BBB.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
BBB_FEW_SHOT_TEMPLATE_v2 = """Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.

Question: Given the drug SMILES string, determine the likelihood of the compound penetrating the BBB. Classify the prediction as
- 0: The compound is not expected to effectively penetrate the BBB.
- 1: The compound is likely to penetrate the BBB.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
BBB_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.

Question: Given the drug SMILES string, determine the likelihood of the compound penetrating the BBB. Classify the prediction as
- 0: The compound is not expected to effectively penetrate the BBB.
- 1: The compound is likely to penetrate the BBB.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CACO_INSTRUCTION_TEMPLATE = """Predict the Caco-2 cell effective permeability of the given drug. """
CACO_FEW_SHOT_TEMPLATE = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CACO_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CACO_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug SMILES string, predict the normalized Caco-2 cell effective permeability from 0 to 1000, where 0 is the minimum effective permeability and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CACO_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW = """Context: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Question: Given the drug SMILES string, predict the value of Caco-2 cell effective permeability.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CYP2C9_Veith_INSTRUCTION_TEMPLATE = "Predict whether the given drug can act as an inhibitor of the CYP2C9 enzyme."
CYP2C9_Veith_FEW_SHOT_TEMPLATE_v0 = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.

Question: Given the drug SMILES string, determine whether given drug will act as an inhibitor of the CYP2C9 enzyme. Predict one of the following categories:
- 0 if the drug molecule does not inhibit the CYP2C9 enzyme.
- 1 if the drug molecule does inhibit the CYP2C9 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2C9_Veith_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.

Question: Given the drug SMILES string, determine whether given drug will act as an inhibitor of the CYP2C9 enzyme. Predict one of the following categories:
- 0 if the drug molecule does not inhibit the CYP2C9 enzyme.
- 1 if the drug molecule does inhibit the CYP2C9 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2C9_Veith_FEW_SHOT_TEMPLATE_v1 = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP2C9 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP2C9 enzyme.
- 1: The drug molecule is likely to inhibit the CYP2C9 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2C9_Veith_FEW_SHOT_TEMPLATE_WITH_PROPERTY_v1 = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP2C9 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP2C9 enzyme.
- 1: The drug molecule is likely to inhibit the CYP2C9 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2C9_Veith_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP2C9 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP2C9 enzyme.
- 1: The drug molecule is likely to inhibit the CYP2C9 enzyme.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

HR_INSTRUCTION_TEMPLATE = "Predict the half life duration of the given drug."
HR_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Half life of a drug is the duration for the concentration of the drug in the body to be reduced by half. It measures the duration of actions of a drug.

Question: Given the drug SMILES string, predict the normalized half life from 0 to 1000, where 0 is the minimum half life and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
HR_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Half life of a drug is the duration for the concentration of the drug in the body to be reduced by half. It measures the duration of actions of a drug.

Question: Given the drug SMILES string, predict the normalized half life from 0 to 1000, where 0 is the minimum half life and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
HR_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW= """Context: Half life of a drug is the duration for the concentration of the drug in the body to be reduced by half. It measures the duration of actions of a drug.

Question: Given the drug SMILES string, predict the value of half life.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
HR_FEW_SHOT_TEMPLATE = """Context: Half life of a drug is the duration for the concentration of the drug in the body to be reduced by half. It measures the duration of actions of a drug.

Question: Given the drug SMILES string, predict the normalized half life from 0 to 1000, where 0 is the minimum half life and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""


LD50_INSTRUCTION_TEMPLATE = "Predict the acute toxicity of the given drug."
LD50_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug.

Question: Given the drug SMILES string, predict the normalized acute toxicity LD50 from 0 to 1000, where 0 is the minimum acute toxicity and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
LD50_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug.

Question: Given the drug SMILES string, predict the normalized acute toxicity LD50 from 0 to 1000, where 0 is the minimum acute toxicity and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
LD50_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW = """Context: Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug.

Question: Given the drug SMILES string, predict the value of acute toxicity LD50.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
LD50_FEW_SHOT_TEMPLATE = """Context: Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug.

Question: Given the drug SMILES string, predict the normalized acute toxicity LD50 from 0 to 1000, where 0 is the minimum acute toxicity and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""


HIA_HOU_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the activity of human intestinal absorption (HIA) of the given drug."
HIA_HOU_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY= """Context: When a drug is orally administered, it needs to be absorbed from the human gastrointestinal system into the bloodstream of the human body. This ability of absorption is called human intestinal absorption (HIA) and it is crucial for a drug to be delivered to the target.

Question: Given the drug SMILES string, determine the HIA classification of the drug. Classify the prediction as:
- 0: The drug molecule is not expected to have high HIA.
- 1: The drug molecule is likely to have high HIA.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
HIA_HOU_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY= """Context: When a drug is orally administered, it needs to be absorbed from the human gastrointestinal system into the bloodstream of the human body. This ability of absorption is called human intestinal absorption (HIA) and it is crucial for a drug to be delivered to the target.

Question: Given the drug SMILES string, determine the HIA classification of the drug. Classify the prediction as:
- 0: The drug molecule is not expected to have high HIA.
- 1: The drug molecule is likely to have high HIA.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

PGP_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the activity of P-glycoprotein (Pgp) inhibition of the given drug"
PGP_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: P-glycoprotein (Pgp) is an ABC transporter protein involved in intestinal absorption, drug metabolism, and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety. In addition, inhibitors of Pgp can be used to overcome multidrug resistance.

Question: Given the drug SMILES string, determine if the drug is likely to be a substrate or inhibitor of P-glycoprotein (Pgp). Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate or inhibitor of Pgp.
- 1: The drug molecule is likely to be a substrate or inhibitor of Pgp.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
PGP_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: P-glycoprotein (Pgp) is an ABC transporter protein involved in intestinal absorption, drug metabolism, and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety. In addition, inhibitors of Pgp can be used to overcome multidrug resistance.

Question: Given the drug SMILES string, determine if the drug is likely to be a substrate or inhibitor of P-glycoprotein (Pgp). Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate or inhibitor of Pgp.
- 1: The drug molecule is likely to be a substrate or inhibitor of Pgp.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

MA_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the activity of bioavailability of the given drug."
MA_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Oral bioavailability is defined as “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”.

Question: Given the drug SMILES string, predict the bioavailability classification of the compound. Classify the prediction as:
- 0: The compound is predicted to have low bioavailability.
- 1: The compound is predicted to have high bioavailability.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
MA_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Oral bioavailability is defined as “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”.

Question: Given the drug SMILES string, predict the bioavailability classification of the compound. Classify the prediction as:
- 0: The compound is predicted to have low bioavailability.
- 1: The compound is predicted to have high bioavailability.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

LIPO_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the activity of lipophilicity of the given drug. "
LIPO_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption.

Question: Given the drug SMILES string, predict the normalized activity of lipophilicity from 0 to 1000, where 0 is the minimum activity of lipophilicity and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
LIPO_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption.

Question: Given the drug SMILES string, predict the normalized activity of lipophilicity from 0 to 1000, where 0 is the minimum activity of lipophilicity and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
LIPO_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW = """Context: Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption.

Question: Given the drug SMILES string, predict the value of activity of lipophilicity.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

Solubility_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the activity of solubility of the given drug."
Solubility_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.

Question: Given the drug SMILES string, predict the normalized activity of solubility from 0 to 1000, where 0 is the minimum solubility and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Solubility_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.

Question: Given the drug SMILES string, predict the normalized activity of solubility from 0 to 1000, where 0 is the minimum solubility and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Solubility_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW = """Context: Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.

Question: Given the drug SMILES string, predict the value of activity of solubility.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

PPBR_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the human plasma protein binding rate (PPBR) of the given drug."
PPBR_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions.

Question: Given the drug SMILES string, predict the normalized human PPBR from 0 to 1000, where 0 is the minimum PPBR and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
PPBR_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions.

Question: Given the drug SMILES string, predict the normalized human PPBR from 0 to 1000, where 0 is the minimum PPBR and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
PPBR_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW = """Context: The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions.

Question: Given the drug SMILES string, predict the value of human PPBR.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

VDSS_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict the volumn of Distribution at steady state (VDss) of the given drug."
VDSS_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.

Question: Given the drug SMILES string, predict the normalized VDss from 0 to 1000, where 0 is the minimum VDss and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
VDSS_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.

Question: Given the drug SMILES string, predict the normalized VDss from 0 to 1000, where 0 is the minimum VDss and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
VDSS_TEMPLATE_FEW_SHOT_TEMPLATE = """Context: The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.

Question: Given the drug SMILES string, predict the normalized VDss from 0 to 1000, where 0 is the minimum VDss and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
VDSS_TEMPLATE_ZERO_SHOT_TEMPLATE = """Context: The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.

Question: Given the drug SMILES string, predict the normalized VDss from 0 to 1000, where 0 is the minimum VDss and 1000 is the maximum. The predicted value must be an integer with no decimals.

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
VDSS_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW= """Context: The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.

Question: Given the drug SMILES string, predict the VDss value.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CYP2D6_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug can act as an inhibitor of the CYP2D6 enzyme."
CYP2D6_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP2D6 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP2D6 enzyme.
- 1: The drug molecule is likely to inhibit the CYP2D6 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2D6_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP2D6 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP2D6 enzyme.
- 1: The drug molecule is likely to inhibit the CYP2D6 enzyme.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CYP3A4_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug can act as an inhibitor of the CYP3A4 enzyme."
CYP3A4_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP3A4 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP3A4 enzyme.
- 1: The drug molecule is likely to inhibit the CYP3A4 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP3A4_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.

Question: Given the drug SMILES string, determine if the drug will inhibit the CYP3A4 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to inhibit the CYP3A4 enzyme.
- 1: The drug molecule is likely to inhibit the CYP3A4 enzyme.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CYP2D6_SUBSTRATE_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug is a substrate to the CYP2D6 enzyme."
CYP2D6_SUBSTRATE_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.

Given the drug SMILES string, determine if the drug will act as a substrate of the CYP2D6 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate to the CYP2D6 enzyme.
- 1: The drug molecule is likely to be a substrate to the CYP2D6 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2D6_SUBSTRATE_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.

Given the drug SMILES string, determine if the drug will act as a substrate of the CYP2D6 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate to the CYP2D6 enzyme.
- 1: The drug molecule is likely to be a substrate to the CYP2D6 enzyme.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CYP3A4_SUBSTRATE_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug is a substrate to the CYP3A4 enzyme."
CYP3A4_SUBSTRATE_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.

Given the drug SMILES string, determine if the drug will act as a substrate of the CYP3A4 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate to the CYP3A4 enzyme.
- 1: The drug molecule is likely to be a substrate to the CYP3A4 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP3A4_SUBSTRATE_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.

Given the drug SMILES string, determine if the drug will act as a substrate of the CYP3A4 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate to the CYP3A4 enzyme.
- 1: The drug molecule is likely to be a substrate to the CYP3A4 enzyme.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

CYP2C9_SUBSTRATE_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug is a substrate to the CYP P450 2C9 enzyme."
CYP2C9_SUBSTRATE_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY= """Context: CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. Substrates are drugs that are metabolized by the enzyme.

Given the drug SMILES string, determine if the drug will act as a substrate of the CYP P450 2C9 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate to the CYP P450 2C9 enzyme.
- 1: The drug molecule is likely to be a substrate to the CYP P450 2C9 enzyme.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
CYP2C9_SUBSTRATE_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY= """Context: CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. Substrates are drugs that are metabolized by the enzyme.

Given the drug SMILES string, determine if the drug will act as a substrate of the CYP P450 2C9 enzyme. Classify the prediction as:
- 0: The drug molecule is not expected to be a substrate to the CYP P450 2C9 enzyme.
- 1: The drug molecule is likely to be a substrate to the CYP P450 2C9 enzyme.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

Clearance_Microsome_AZ_TEMPLATE_INSTRUCTION_TEMPLATE ="Predict the activity of microsome clearance of the given drug."
Clearance_Microsome_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. 

Question: Given the drug SMILES string, predict the normalized microsome clearance from 0 to 1000, where 0 is the minimum microsome clearance and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Clearance_Microsome_AZ_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. 

Question: Given the drug SMILES string, predict the normalized microsome clearance from 0 to 1000, where 0 is the minimum microsome clearance and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Clearance_Microsome_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW="""Context: Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. 

Question: Given the drug SMILES string, predict the value of microsome clearance.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Clearance_Hepatocyte_AZ_TEMPLATE_INSTRUCTION_TEMPLATE ="Predict the activity of hepatocyte clearance of the given drug."
Clearance_Hepatocyte_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. 

Question: Given the drug SMILES string, predict the normalized hepatocyte clearance from 0 to 1000, where 0 is the minimum hepatocyte clearance and 1000 is the maximum. The predicted value must be an integer with no decimals.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Clearance_Hepatocyte_AZ_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY ="""Context: Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. 

Question: Given the drug SMILES string, predict the normalized hepatocyte clearance from 0 to 1000, where 0 is the minimum hepatocyte clearance and 1000 is the maximum. The predicted value must be an integer with no decimals.

```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
Clearance_Hepatocyte_AZ_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY_RAW ="""Context: Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body. 

Question: Given the drug SMILES string, predict the value of hepatocyte clearance.

Examples:
CUR_EXAMPLES

Now, using the information provided, predict the normalized permeability for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
    CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

hERG_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug can block human ether-à-go-go related gene."
hERG_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages.

Given the drug SMILES string, determine if the drug will block hERG. Classify the prediction as:
- 0: The drug is not expected to block hERG.
- 1: The drug is likely to block hERG.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
hERG_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages.

Given the drug SMILES string, determine if the drug will block hERG. Classify the prediction as:
- 0: The drug is not expected to block hERG.
- 1: The drug is likely to block hERG.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

AMES_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug is mutagenic or not mutagenic."
AMES_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Mutagenicity means the ability of a drug to induce genetic alterations. Drugs that can cause damage to the DNA can result in cell death or other severe adverse effects. Nowadays, the most widely used assay for testing the mutagenicity of compounds is the Ames experiment which was invented by a professor named Ames. The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations.

Given the drug SMILES string, determine if the drug is mutagenic. Classify the prediction as:
- 0: The drug is not mutagenic.
- 1: The drug is mutagenic.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
AMES_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context: Mutagenicity means the ability of a drug to induce genetic alterations. Drugs that can cause damage to the DNA can result in cell death or other severe adverse effects. Nowadays, the most widely used assay for testing the mutagenicity of compounds is the Ames experiment which was invented by a professor named Ames. The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations.

Given the drug SMILES string, determine if the drug is mutagenic. Classify the prediction as:
- 0: The drug is not mutagenic.
- 1: The drug is mutagenic.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

DILI_TEMPLATE_INSTRUCTION_TEMPLATE = "Predict whether the given drug can cause liver injury or not."
DILI_TEMPLATE_FEW_SHOT_TEMPLATE_WITH_PROPERTY = """Context:  Drug-induced liver injury (DILI) is fatal liver disease caused by drugs and it has been the single most frequent cause of safety-related drug marketing withdrawals for the past 50 years (e.g. iproniazid, ticrynafen, benoxaprofen).

Given the drug SMILES string, determine if the drug can cause liver injury. Classify the prediction as:
- 0: The drug cannot cause liver injury.
- 1: The drug can cause liver injury.

Examples:
CUR_EXAMPLES


Now, using the information provided, predict the CUR_TARGET for the following drug:
```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
DILI_TEMPLATE_ZERO_SHOT_TEMPLATE_WITH_PROPERTY = """Context:  Drug-induced liver injury (DILI) is fatal liver disease caused by drugs and it has been the single most frequent cause of safety-related drug marketing withdrawals for the past 50 years (e.g. iproniazid, ticrynafen, benoxaprofen).

Given the drug SMILES string, determine if the drug can cause liver injury. Classify the prediction as:
- 0: The drug cannot cause liver injury.
- 1: The drug can cause liver injury.

```
{
    "drug SMILES": CUR_DRUG_SMILES
CUR_DRUG_PROPERTIES
}
```

IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""