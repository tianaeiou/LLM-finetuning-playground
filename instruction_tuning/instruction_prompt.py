ALL_TASKS = ['caco2_wang', 'hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'lipophilicity_astrazeneca',
             'solubility_aqsoldb', 'bbb_martins', 'ppbr_az', 'vdss_lombardo', 'cyp2d6_veith', 'cyp3a4_veith',
             'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels', 'cyp3a4_substrate_carbonmangels',
             'cyp2c9_substrate_carbonmangels', 'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az',
             'herg', 'ames', 'dili', 'ld50_zhu']
REGRESSION_TASKS = ['caco2_wang', 'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'ppbr_az', 'vdss_lombardo',
                    'half_life_obach', 'clearance_microsome_az', 'clearance_hepatocyte_az', 'ld50_zhu']
CLASSIFICATION_TASKS = ['hia_hou', 'pgp_broccatelli', 'bioavailability_ma', 'bbb_martins', 'cyp2d6_veith',
                        'cyp3a4_veith', 'cyp2c9_veith', 'cyp2d6_substrate_carbonmangels',
                        'cyp3a4_substrate_carbonmangels', 'cyp2c9_substrate_carbonmangels', 'herg', 'ames', 'dili']

SYSTEM_INSTRUCTION = "You are an AI assistant specializing in ADMET property prediction for drug discovery. The user may ask you to predict the absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties of a molecule. You should provide insightful predictions and follow instructions based on your knowledge."

INSTRUCTION = {
    "bbb_martins": "Predict whether the compound can penetrate the Blood-Brain Barrier.",
    "caco2_wang": "Predict the Caco-2 cell effective permeability of the given drug.",
    "hia_hou": "Predict the activity of human intestinal absorption (HIA) of the given drug.",
    "pgp_broccatelli": "Predict the activity of P-glycoprotein (Pgp) inhibition of the given drug.",
    "bioavailability_ma": "Predict the activity of bioavailability of the given drug.",
    "lipophilicity_astrazeneca": "Predict the activity of lipophilicity of the given drug.",
    "solubility_aqsoldb": "Predict the activity of solubility of the given drug.",
    "ppbr_az": "Predict the human plasma protein binding rate (PPBR) of the given drug.",
    "vdss_lombardo": "Predict the volumn of Distribution at steady state (VDss) of the given drug.",
    "cyp2d6_veith": "Predict whether the given drug can act as an inhibitor of the CYP2D6 enzyme.",
    "cyp3a4_veith": "Predict whether the given drug can act as an inhibitor of the CYP3A4 enzyme.",
    "cyp2c9_veith": "Predict whether the given drug can act as an inhibitor of the CYP2C9 enzyme.",
    "cyp2d6_substrate_carbonmangels": "Predict whether the given drug is a substrate to the CYP2D6 enzyme.",
    "cyp3a4_substrate_carbonmangels": "Predict whether the given drug is a substrate to the CYP3A4 enzyme.",
    "cyp2c9_substrate_carbonmangels": "Predict whether the given drug is a substrate to the CYP P450 2C9 enzyme.",
    "half_life_obach": "Predict the half life duration of the given drug.",
    "clearance_microsome_az": "Predict the activity of microsome clearance of the given drug.",
    "clearance_hepatocyte_az": "Predict the activity of hepatocyte clearance of the given drug.",
    "herg": "Predict whether the given drug can block human ether-à-go-go related gene.",
    "ames": "Predict whether the given drug is mutagenic or not mutagenic.",
    "dili": "Predict whether the given drug can cause liver injury or not.",
    "ld50_zhu": "Predict the acute toxicity of the given drug."
}

CONTEXT = {
    "bbb_martins": "The Blood-Brain Barrier (BBB) is a selective barrier that limits the passage of substances into the brain, crucial for protecting the central nervous system. Effective BBB penetration is a key determinant for drugs targeting neurological conditions, as it allows them to reach their therapeutic sites within the brain.",
    "caco2_wang": "The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.",
    "hia_hou": "When a drug is orally administered, it needs to be absorbed from the human gastrointestinal system into the bloodstream of the human body. This ability of absorption is called human intestinal absorption (HIA) and it is crucial for a drug to be delivered to the target.",
    "pgp_broccatelli": "P-glycoprotein (Pgp) is an ABC transporter protein involved in intestinal absorption, drug metabolism, and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety. In addition, inhibitors of Pgp can be used to overcome multidrug resistance.",
    "bioavailability_ma": "Oral bioavailability is defined as “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action.",
    "lipophilicity_astrazeneca": "Lipophilicity measures the ability of a drug to dissolve in a lipid (e.g. fats, oils) environment. High lipophilicity often leads to high rate of metabolism, poor solubility, high turn-over, and low absorption.",
    "solubility_aqsoldb": "Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble.",
    "ppbr_az": "The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions.",
    "vdss_lombardo": "The volume of distribution at steady state (VDss) measures the degree of a drug's concentration in body tissue compared to concentration in blood. Higher VD indicates a higher distribution in the tissue and usually indicates the drug with high lipid solubility, low plasma protein binidng rate.",
    "cyp2d6_veith": "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.",
    "cyp3a4_veith": "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.",
    "cyp2c9_veith": "CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. Substrates are drugs that are metabolized by the enzyme.",
    "cyp2d6_substrate_carbonmangels": "CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra.",
    "cyp3a4_substrate_carbonmangels": "CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body.",
    "cyp2c9_substrate_carbonmangels": "CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. Substrates are drugs that are metabolized by the enzyme.",
    "half_life_obach": "Half life of a drug is the duration for the concentration of the drug in the body to be reduced by half. It measures the duration of actions of a drug.",
    "clearance_microsome_az": "Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body.",
    "clearance_hepatocyte_az": "Drug clearance is defined as the volume of plasma cleared of a drug over a specified time period and it measures the rate at which the active drug is removed from the body.",
    "herg": "Human ether-à-go-go related gene (hERG) is crucial for the coordination of the heart's beating. Thus, if a drug blocks the hERG, it could lead to severe adverse effects. Therefore, reliable prediction of hERG liability in the early stages of drug design is quite important to reduce the risk of cardiotoxicity-related attritions in the later development stages.",
    "ames": "Mutagenicity means the ability of a drug to induce genetic alterations. Drugs that can cause damage to the DNA can result in cell death or other severe adverse effects. Nowadays, the most widely used assay for testing the mutagenicity of compounds is the Ames experiment which was invented by a professor named Ames. The Ames test is a short-term bacterial reverse mutation assay detecting a large number of compounds which can induce genetic damage and frameshift mutations.",
    "dili": "Drug-induced liver injury (DILI) is fatal liver disease caused by drugs and it has been the single most frequent cause of safety-related drug marketing withdrawals for the past 50 years (e.g. iproniazid, ticrynafen, benoxaprofen).",
    "ld50_zhu": "Acute toxicity LD50 measures the most conservative dose that can lead to lethal adverse effects. The higher the dose, the more lethal of a drug."
}

EXAMPLE_TEMPLATE = """```
{{
    "drug SMILES": {CUR_DRUG_SMILES}
    "answer": {CUR_ANSWER}
}}
```
"""

PROMPT_TEMPLATE_REG = """Context: {CUR_CONTEXT_INFO}
Question: Given the drug SMILES string, predict the normalized {CUR_TARGET} from 0 to 1000, where 0 is the minimum {CUR_TARGET} and 1000 is the maximum.
```
{{
    "drug SMILES": {CUR_DRUG_SMILES}
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
PROMPT_TEMPLATE_REG_WITH_EXAMPLE = """Context: {CUR_CONTEXT_INFO}
Question: Given the drug SMILES string, predict the normalized {CUR_TARGET} from 0 to 1000, where 0 is the minimum {CUR_TARGET} and 1000 is the maximum.
Examples:
{CUR_EXAMPLES}Now, using the information provided, predict the {CUR_TARGET} for the following drug:
```
{{
    "drug SMILES": {CUR_DRUG_SMILES}
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
PROMPT_TEMPLATE_CLS = """Context: {CUR_CONTEXT_INFO}
Question: Given the drug SMILES string, determine {CUR_TARGET}. Classify the prediction as:
- 0: {LABEL0_DESCRIPTION}
- 1: {LABEL1_DESCRIPTION}
```
{{
    "drug SMILES": {CUR_DRUG_SMILES}
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""
PROMPT_TEMPLATE_CLS_WITH_EXAMPLE = """Context: {CUR_CONTEXT_INFO}
Question: Given the drug SMILES string, determine {CUR_TARGET}. Classify the prediction as:
- 0: {LABEL0_DESCRIPTION}
- 1: {LABEL1_DESCRIPTION}
Examples:
{CUR_EXAMPLES}Now, using the information provided, predict the {CUR_TARGET} for the following drug:
```
{{
    "drug SMILES": {CUR_DRUG_SMILES}
}}
```
IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
"""

EXAMPLE_TEMPLATE_TxLLM = """drug SMILES: {CUR_DRUG_SMILES}
answer: {CUR_ANSWER}
"""
PROMPT_TEMPLATE_CLS_TxLLM = """Context: {CUR_CONTEXT_INFO}
Question: Given the drug SMILES string, predict whether it
- 0: {LABEL0_DESCRIPTION}
- 1: {LABEL1_DESCRIPTION}

drug SMILES: {CUR_DRUG_SMILES}
answer: """
PROMPT_TEMPLATE_CLS_WITH_EXAMPLE_TxLLM = """Context: {CUR_CONTEXT_INFO}
Question: Given the drug SMILES string, predict whether it
- 0: {LABEL0_DESCRIPTION}
- 1: {LABEL1_DESCRIPTION}

{CUR_EXAMPLES}Now, using the information provided, predict the result for the following drug. IMPORTANT: Please provide your predicted value and DO NOT RETURN ANYTHING ELSE.
drug SMILES: {CUR_DRUG_SMILES}
"""

QUESTION = {
    "bbb_martins": {"target": "the likelihood of the compound penetrating the BBB",
                    "label0": "The compound is not expected to effectively penetrate the BBB.",
                    "label1": "The compound is likely to penetrate the BBB."},
    "caco2_wang": {"target": "Caco-2 cell effective permeability"},
    "hia_hou": {"target": "the HIA classification of the drug",
                "label0": "The drug molecule is not expected to have high HIA.",
                "label1": "The drug molecule is likely to have high HIA."},
    "pgp_broccatelli": {"target": "if the drug is likely to be a substrate or inhibitor of P-glycoprotein (Pgp)",
                        "label0": "The drug molecule is not expected to be a substrate or inhibitor of Pgp.",
                        "label1": "The drug molecule is likely to be a substrate or inhibitor of Pgp."},
    "lipophilicity_astrazeneca": {"target": "activity of lipophilicity"},
    "solubility_aqsoldb": {"target": "activity of solubility"},
    "ppbr_az": {"target": "human PPBR"},
    "vdss_lombardo": {"target": "VDss"},
    "half_life_obach": {"target": "half life"},
    "clearance_microsome_az": {"target": "microsome clearance"},
    "clearance_hepatocyte_az": {"target": "hepatocyte clearance"},
    "ld50_zhu": {"target": "acute toxicity LD50"},
    "bioavailability_ma": {"target": "the bioavailability classification of the compound",
                           "label0": "The compound is predicted to have low bioavailability.",
                           "label1": "The compound is predicted to have high bioavailability."},
    "cyp2d6_veith": {"target": "if the drug will inhibit the CYP2D6 enzyme",
                     "label0": "The drug molecule is not expected to inhibit the CYP2D6 enzyme.",
                     "label1": "The drug molecule is likely to inhibit the CYP2D6 enzyme."},
    "cyp3a4_veith": {"target": "if the drug will inhibit the CYP3A4 enzyme",
                     "label0": "The drug molecule is not expected to inhibit the CYP3A4 enzyme.",
                     "label1": "The drug molecule is likely to inhibit the CYP3A4 enzyme."},
    "cyp2c9_veith": {"target": "whether given drug will act as an inhibitor of the CYP2C9 enzyme",
                     "label0": "If the drug molecule does not inhibit the CYP2C9 enzyme.",
                     "label1": "If the drug molecule does inhibit the CYP2C9 enzyme."},
    "cyp2d6_substrate_carbonmangels": {"target": "if the drug will act as a substrate of the CYP2D6 enzyme",
                                       "label0": "The drug molecule is not expected to be a substrate to the CYP2D6 enzyme.",
                                       "label1": "The drug molecule is likely to be a substrate to the CYP2D6 enzyme."},
    "cyp3a4_substrate_carbonmangels": {"target": "if the drug will act as a substrate of the CYP3A4 enzyme",
                                       "label0": "The drug molecule is not expected to be a substrate to the CYP3A4 enzyme.",
                                       "label1": "The drug molecule is likely to be a substrate to the CYP3A4 enzyme."},

    "cyp2c9_substrate_carbonmangels": {"target": "if the drug will act as a substrate of the CYP P450 2C9 enzyme",
                                       "label0": "The drug molecule is not expected to be a substrate to the CYP P450 2C9 enzyme.",
                                       "label1": "The drug molecule is likely to be a substrate to the CYP P450 2C9 enzyme."},
    "herg": {"target": "if the drug will block hERG",
             "label0": "The drug is not expected to block hERG.",
             "label1": "The drug is likely to block hERG."},
    "ames": {"target": "if the drug is mutagenic",
             "label0": "The drug is not mutagenic.",
             "label1": "The drug is mutagenic."},
    "dili": {"target": "if the drug can cause liver injury",
             "label0": "The drug cannot cause liver injury.",
             "label1": "The drug can cause liver injury."},

}
