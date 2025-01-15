# -*- coding: utf-8 -*-

import json
import pandas as pd

def get_drug_dict_smiles_drugbank():
    id_to_smiles = {}
    
    with open("../data/raw_data/drugbank.txt", "r") as f:
        lines = f.readlines()[1 : ]
    
    for line in lines:
        parts = line[ : -1].split("\t")
        parts = [part.replace("\"", "").strip() for part in parts]
        id_1, id_2, ddi_type, _, smiles_1, smiles_2 = parts
        id_to_smiles[id_1] = smiles_1
        id_to_smiles[id_2] = smiles_2
        
    id_to_smiles = sorted(list(id_to_smiles.items()))
    drug_dict, smiles_list = zip(*id_to_smiles)
    drug_dict = {item : i for i, item in enumerate(drug_dict)}
    smiles_list = list(smiles_list)
    
    with open("../data/drugbank/drugbank_drug_dict.json", "w") as f:
        json.dump(drug_dict, f)
    
    with open("../data/drugbank/drugbank_smiles.json", "w") as f:
        json.dump(smiles_list, f)


def get_drug_dict_smiles_twosides():
    id_to_smiles = {}
    data = pd.read_csv('../data/raw_data/twosides_ge_500.csv', delimiter=',')

    for id1, id2, smiles1, smiles2, relation in zip(data['Drug1_ID'], data['Drug2_ID'], data['Drug1'], data['Drug2'], data['New Y']):
        id_to_smiles[id1] = smiles1
        id_to_smiles[id2] = smiles2

    id_to_smiles = sorted(list(id_to_smiles.items()))
    drug_dict, smiles_list = zip(*id_to_smiles)
    drug_dict = {item: i for i, item in enumerate(drug_dict)}
    smiles_list = list(smiles_list)

    with open("../data/twosides/twosides_drug_dict.json", "w") as f:
        json.dump(drug_dict, f, indent=4)

    with open("../data/twosides/twosides_smiles.json", "w") as f:
        json.dump(smiles_list, f)

#get_drug_dict_smiles_drugbank()
get_drug_dict_smiles_twosides()

