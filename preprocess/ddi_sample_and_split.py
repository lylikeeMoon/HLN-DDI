# -*- coding: utf-8 -*-

import json
import random

from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

def get_pos_pairs_drugbank():
    pos_pairs = []
    num_ddi_types = 0
    
    with open(f"../data/drugbank/drugbank_drug_dict.json", "r") as f:
        drug_dict = json.load(f)

    with open("../data/raw_data/drugbank.txt", "r") as f:
        lines = f.readlines()[1 : ]
    
    for line in lines:
        parts = line[ : -1].split("\t")
        parts = [part.replace("\"", "").strip() for part in parts]
        id_1, id_2, ddi_type, _, smiles_1, smiles_2 = parts
        
        ddi_type = int(ddi_type) - 1
        if ddi_type + 1 > num_ddi_types:
            num_ddi_types = ddi_type + 1
        drug_1=drug_dict[id_1]
        drug_2=drug_dict[id_2]
        pos_pairs.append([drug_1,drug_2,ddi_type])
    
    num_drugs = len(drug_dict)

    return pos_pairs, num_drugs, num_ddi_types

def get_pos_pairs_twosides():
    pos_pairs = []
    num_ddi_types = 0

    with open(f"../data/twosides/twosides_drug_dict.json", "r") as f:
        drug_dict = json.load(f)

    data = pd.read_csv('../data/raw_data/twosides_ge_500.csv', delimiter=',')

    for id_1, id_2, smiles_1, smiles_2, ddi_type in zip(data['Drug1_ID'], data['Drug2_ID'], data['Drug1'], data['Drug2'],
                                                    data['New Y']):
        ddi_type = int(ddi_type)
        if ddi_type + 1 > num_ddi_types:
            num_ddi_types = ddi_type + 1
        pos_pairs.append([drug_dict[id_1], drug_dict[id_2], ddi_type])

    num_drugs = len(drug_dict)

    return pos_pairs, num_drugs, num_ddi_types

def get_data_stats_drugbank(pos_pairs, num_ddi_types):
    drug_2_type_to_drug_1_set = defaultdict(set)
    drug_1_type_to_drug_2_set = defaultdict(set)
    ddi_type_freq = [0] * num_ddi_types

    type_to_drug_1_set = [set() for i in range(num_ddi_types)]
    type_to_drug_2_set = [set() for i in range(num_ddi_types)]
    
    for drug_1,drug_2,ddi_type in pos_pairs:
        drug_2_type_to_drug_1_set[(drug_2, ddi_type)].add(drug_1)
        drug_1_type_to_drug_2_set[(drug_1, ddi_type)].add(drug_2)
        
        ddi_type_freq[ddi_type] += 1
        type_to_drug_1_set[ddi_type].add(drug_1)
        type_to_drug_2_set[ddi_type].add(drug_2)

    type_to_drug_1_cnt = [len(type_to_drug_1_set[i]) for i in range(num_ddi_types)]
    type_to_drug_2_cnt = [len(type_to_drug_2_set[i]) for i in range(num_ddi_types)]
    
    return {
        "drug_2_type_to_drug_1_set" : drug_2_type_to_drug_1_set,
        "drug_1_type_to_drug_2_set" : drug_1_type_to_drug_2_set,
        "ddi_type_freq" : ddi_type_freq,
        "type_to_drug_1_cnt" : type_to_drug_1_cnt,
        "type_to_drug_2_cnt" : type_to_drug_2_cnt
    }


def get_data_stats_twosides(pos_pairs, num_ddi_types):
    drug_2_type_to_drug_1_set = defaultdict(set)
    drug_1_type_to_drug_2_set = defaultdict(set)
    ddi_type_freq = [0] * num_ddi_types

    type_to_drug_1_set = [set() for i in range(num_ddi_types)]
    type_to_drug_2_set = [set() for i in range(num_ddi_types)]

    for drug_1, drug_2, ddi_type in pos_pairs:
        drug_2_type_to_drug_1_set[(drug_2, ddi_type)].add(
            drug_1)
        drug_1_type_to_drug_2_set[(drug_1, ddi_type)].add(
            drug_2)

        ddi_type_freq[ddi_type] += 1
        type_to_drug_1_set[ddi_type].add(
            drug_1)
        type_to_drug_2_set[ddi_type].add(drug_2)

    type_to_drug_1_cnt = [len(type_to_drug_1_set[i]) for i in range(num_ddi_types)]
    type_to_drug_2_cnt = [len(type_to_drug_2_set[i]) for i in range(num_ddi_types)]

    return {
        "drug_2_type_to_drug_1_set": drug_2_type_to_drug_1_set,
        "drug_1_type_to_drug_2_set": drug_1_type_to_drug_2_set,
        "ddi_type_freq": ddi_type_freq,
        "type_to_drug_1_cnt": type_to_drug_1_cnt,
        "type_to_drug_2_cnt": type_to_drug_2_cnt
    }

def do_sample_transductive(num_drugs, pos_drugs):
    return random.choice(list(set(range(num_drugs)) - pos_drugs))

def do_sample_neg_transductive(drug_1, drug_2, ddi_type, data_stats, num_drugs):
    cnt_1 = data_stats["type_to_drug_1_cnt"][ddi_type]
    cnt_2 = data_stats["type_to_drug_2_cnt"][ddi_type]
    prob = cnt_2 / (cnt_1 + cnt_2)
    
    if random.random() < prob:
        drug_2_to_drug_1 = data_stats["drug_2_type_to_drug_1_set"][(drug_2, ddi_type)]
        return do_sample_transductive(num_drugs, drug_2_to_drug_1), 1
    
    drug_1_to_drug_2 = data_stats["drug_1_type_to_drug_2_set"][(drug_1, ddi_type)]
    return do_sample_transductive(num_drugs, drug_1_to_drug_2), 2


def get_pairs_drugbank(seed=0):
    random.seed(seed)
    
    pos_pairs, num_drugs, num_ddi_types = get_pos_pairs_drugbank()
    
    print(f"{num_drugs} drugs")
    print(f"{num_ddi_types} ddi types")
    
    data_stats = get_data_stats_drugbank(pos_pairs, num_ddi_types)
    
    neg_pairs = []
    for drug_1, drug_2, ddi_type in pos_pairs:
        sampled, flag = do_sample_neg_transductive(
            drug_1, drug_2, ddi_type, data_stats, num_drugs
        )

        if flag == 1:
            neg_pairs.append([sampled, drug_2,ddi_type])
        else:
            neg_pairs.append([drug_1, sampled,  ddi_type])
    
    print(f"{len(pos_pairs)} pos pairs")
    print(f"{len(neg_pairs)} neg pairs")
    
    with open("../data/drugbank/drugbank_all_pairs.json", "w") as f:
        json.dump({"pos" : pos_pairs, "neg" : neg_pairs}, f)


def get_pairs_twosides(seed=0):
    random.seed(seed)

    pos_pairs, num_drugs, num_ddi_types = get_pos_pairs_twosides()

    print(f"{num_drugs} drugs")
    print(f"{num_ddi_types} ddi types")

    data_stats = get_data_stats_twosides(pos_pairs, num_ddi_types)

    neg_pairs = []
    for drug_1, drug_2, ddi_type in pos_pairs:
        sampled, flag = do_sample_neg_transductive(
            drug_1, drug_2, ddi_type, data_stats, num_drugs
        )
        if flag == 1:
            neg_pairs.append([sampled, drug_2, ddi_type])
        else:
            neg_pairs.append([drug_1, sampled, ddi_type])

    print(f"{len(pos_pairs)} pos pairs")
    print(f"{len(neg_pairs)} neg pairs")

    with open("../data/twosides/twosides_all_pairs.json", "w") as f:
        json.dump({"pos": pos_pairs, "neg": neg_pairs}, f)



def split_transductive(num_folds, dataset, seed):
    random.seed(0)

    with open(f"../data/{dataset}/{dataset}_all_pairs.json",
              "r") as f:
        all_pairs = json.load(f)

    _, _, y = zip(*all_pairs["pos"])
    y = np.array(y)

    train_valid_and_test = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.2,
                                                  random_state=seed)

    for fold, (train_valid_index, test_index) in enumerate(train_valid_and_test.split(X=np.zeros(y.shape[0]), y=y)):
        train_valid_index = train_valid_index.tolist()
        test_index = test_index.tolist()

        test_pos, test_neg = [], []
        for cur_idx in test_index:
            test_pos.append(all_pairs["pos"][cur_idx])
            test_neg.append(all_pairs["neg"][cur_idx])

        with open(f"../data/{dataset}/{dataset}_test_{fold}.json", "w") as f:
            json.dump({"pos": test_pos, "neg": test_neg}, f)

        cur_y = y[train_valid_index]
        train_and_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_index, valid_index = next(train_and_valid.split(X=np.zeros(cur_y.shape[0]), y=cur_y))

        train_pos, train_neg = [], []
        for cur_idx in train_index:
            train_pos.append(all_pairs["pos"][train_valid_index[cur_idx]])
            train_neg.append(all_pairs["neg"][train_valid_index[cur_idx]])

        with open(f"../data/{dataset}/{dataset}_train_{fold}.json", "w") as f:
            json.dump({"pos": train_pos, "neg": train_neg}, f)

        valid_pos, valid_neg = [], []
        for cur_idx in valid_index:
            valid_pos.append(all_pairs["pos"][train_valid_index[cur_idx]])
            valid_neg.append(all_pairs["neg"][train_valid_index[cur_idx]])

        with open(f"../data/{dataset}/{dataset}_valid_{fold}.json", "w") as f:
            json.dump({"pos": valid_pos, "neg": valid_neg}, f)


def split_transductive_for_vis(dataset, seed):
    random.seed(0)

    with open(f"../data/{dataset}/{dataset}_all_pairs.json",
              "r") as f:
        all_pairs = json.load(f)

    _, _, y = zip(*all_pairs["pos"])
    y = np.array(y)

    train_and_vis = StratifiedShuffleSplit(n_splits=1, test_size=0.005,
                                                  random_state=seed)

    for fold, (train_index, vis_index) in enumerate(train_and_vis.split(X=np.zeros(y.shape[0]), y=y)):
        train_index = train_index.tolist()
        vis_index = vis_index.tolist()

        vis_pos, vis_neg = [], []
        for cur_idx in vis_index:
            vis_pos.append(all_pairs["pos"][cur_idx])
            vis_neg.append(all_pairs["neg"][cur_idx])

        with open(f"../data/{dataset}/{dataset}_vis_0.json", "w") as f:
            json.dump({"pos": vis_pos, "neg": vis_neg}, f)

        train_pos, train_neg = [], []
        for cur_idx in train_index:
            train_pos.append(all_pairs["pos"][cur_idx])
            train_neg.append(all_pairs["neg"][cur_idx])

        with open(f"../data/{dataset}/{dataset}_tra_0.json", "w") as f:
            json.dump({"pos": train_pos, "neg": train_neg}, f)


def count_ddi_types(split, num_ddi_types=86):
    ddi_type_cnt = [0] * num_ddi_types
    for _, _, ddi_type in split:
        ddi_type_cnt[ddi_type] += 1
    
    coverage = len(list(filter(lambda x : x > 0, ddi_type_cnt)))
    return ddi_type_cnt, coverage

def split_inductive(new_ratio=0.2, fold=0, dataset='drugbank',num_ddi_types=86, seed=0):
    random.seed(seed)
    
    with open(f"../data/{dataset}/{dataset}_drug_dict.json", "r") as f:
        drug_dict = json.load(f)

    num_drugs = len(drug_dict)
    num_drugs_old = int((1.0 - new_ratio) * num_drugs)
    num_drugs_new = num_drugs - num_drugs_old
    
    print(f"old: {num_drugs_old}")
    print(f"new: {num_drugs_new}")
    
    drugs_old = random.sample(range(num_drugs), num_drugs_old)
    drugs_old = set(drugs_old)
    drugs_new = set(range(num_drugs)) - drugs_old
    
    with open(f"../data/{dataset}/{dataset}_all_pairs.json", "r") as f:
        all_pairs = json.load(f)
    
    train_split = []
    either_split = []
    both_split = []
    
    for drug_1, drug_2, ddi_type in all_pairs["pos"]:
        if (drug_1 in drugs_old) and (drug_2 in drugs_old):
            train_split.append([drug_1, drug_2, ddi_type])
        elif (drug_1 in drugs_new) and (drug_2 in drugs_new):
            both_split.append([drug_1, drug_2, ddi_type])
        else:
            either_split.append([drug_1, drug_2, ddi_type])

    train_ddi_types, train_coverage = count_ddi_types(train_split,num_ddi_types)
    either_ddi_types, either_coverage = count_ddi_types(either_split,num_ddi_types)
    both_ddi_types, both_coverage = count_ddi_types(both_split,num_ddi_types)
    
    print(f"Train Edges:  {len(train_split)}")
    print(f"Either Edges: {len(either_split)}")
    print(f"Both Edges:   {len(both_split)}")
    
    print(f"Train Label Coverage:  {train_coverage}")
    print(f"Either Label Coverage: {either_coverage}")
    print(f"Both Label Coverage:   {both_coverage}")
    
    print()

    with open(f"../data/{dataset}/{dataset}_old_new_{fold}.json", "w") as f:
        json.dump({"old" : list(drugs_old), "new" : list(drugs_new)}, f)
    
    with open(f"../data/{dataset}/{dataset}_ind_train_pos_{fold}.json", "w") as f:
        json.dump(train_split, f)
    
    with open(f"../data/{dataset}/{dataset}_ind_either_pos_{fold}.json", "w") as f:
        json.dump(either_split, f)
    
    with open(f"../data/{dataset}/{dataset}_ind_both_pos_{fold}.json", "w") as f:
        json.dump(both_split, f)

def do_sample_inductive(allowable_set, pos_drugs):
    return random.choice(list(allowable_set - pos_drugs))

def do_sample_neg_inductive(
    drug_1, drug_2, ddi_type,
    data_stats, drugs_old, drugs_new, split):
    
    if split == "either":
        if drug_1 in drugs_old:
            allowable_set_2 = drugs_new
        else:
            allowable_set_2 = drugs_old
        
        if drug_2 in drugs_old:
            allowable_set_1 = drugs_new
        else:
            allowable_set_1 = drugs_old
    
    elif split == "train":
        allowable_set_1 = drugs_old
        allowable_set_2 = drugs_old
    
    else:
        allowable_set_1 = drugs_new
        allowable_set_2 = drugs_new
    
    cnt_1 = data_stats["type_to_drug_1_cnt"][ddi_type]
    cnt_2 = data_stats["type_to_drug_2_cnt"][ddi_type]
    prob = cnt_2 / (cnt_1 + cnt_2)
    
    if random.random() < prob:
        drug_2_to_drug_1 = data_stats["drug_2_type_to_drug_1_set"][(drug_2, ddi_type)]
        return do_sample_inductive(allowable_set_1, drug_2_to_drug_1), 1

    drug_1_to_drug_2 = data_stats["drug_1_type_to_drug_2_set"][(drug_1, ddi_type)]
    return do_sample_inductive(allowable_set_2, drug_1_to_drug_2), 2

def sample_inductive_neg_proc(
    data_stats, drugs_old, drugs_new, split,
    fold,dataset,num_ddi_types, seed):
    
    with open(f"../data/{dataset}/{dataset}_ind_{split}_pos_{fold}.json", "r") as f:
        pos_pairs = json.load(f)
    
    neg_pairs = []

    for drug_1, drug_2, ddi_type in pos_pairs:
        sampled, flag = do_sample_neg_inductive(
            drug_1, drug_2, ddi_type, data_stats, drugs_old, drugs_new, split
        )
        if flag == 1:
            neg_pairs.append([sampled, drug_2, ddi_type])
        else:
            neg_pairs.append([drug_1, sampled, ddi_type])
    
    if split == "train":
        _, _, y = zip(*pos_pairs)
        y = np.array(y)
        original_len = y.shape[0]

        ddi_type_cnt = np.bincount(y, minlength=num_ddi_types)
        idx_cnt_one = np.where(ddi_type_cnt == 1)[0]
        y = np.concatenate([y, idx_cnt_one])

        train_and_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_index, valid_index = next(train_and_valid.split(X=np.zeros(y.shape[0]), y=y))
        
        train_index = train_index.tolist()
        valid_index = valid_index.tolist()
        
        train_index = list(filter(lambda x : x < original_len, train_index))
        valid_index = list(filter(lambda x : x < original_len, valid_index))
        
        train_pos, train_neg = [], []
        for cur_idx in train_index:
            train_pos.append(pos_pairs[cur_idx])
            train_neg.append(neg_pairs[cur_idx])
        
        with open(f"../data/{dataset}/{dataset}_ind_train_{fold}.json", "w") as f:
            json.dump({"pos" : train_pos, "neg" : train_neg}, f)
        
        valid_pos, valid_neg = [], []
        for cur_idx in valid_index:
            valid_pos.append(pos_pairs[cur_idx])
            valid_neg.append(neg_pairs[cur_idx])
        
        with open(f"../data/{dataset}/{dataset}_ind_valid_{fold}.json", "w") as f:
            json.dump({"pos" : valid_pos, "neg" : valid_neg}, f)
        
        print(f"train: {len(train_pos)} pos pairs")
        print(f"train: {len(train_neg)} neg pairs")
        print(f"valid: {len(valid_pos)} pos pairs")
        print(f"valid: {len(valid_neg)} neg pairs")
        
        return
    
    print(f"{split}: {len(pos_pairs)} pos pairs")
    print(f"{split}: {len(neg_pairs)} neg pairs")
    
    with open(f"../data/{dataset}/{dataset}_ind_{split}_{fold}.json", "w") as f:
        json.dump({"pos" : pos_pairs, "neg" : neg_pairs}, f)


def sample_inductive_neg(fold, dataset='drugbank', seed=0):
    random.seed(seed)
    
    with open(f"../data/{dataset}/{dataset}_old_new_{fold}.json", "r") as f:
        old_new = json.load(f)

    # 并分别保存到 drugs_old 和 drugs_new 集合中。
    drugs_old = set(old_new["old"])
    drugs_new = set(old_new["new"])

    if dataset =='drugbank':
        pos_pairs, num_drugs, num_ddi_types = get_pos_pairs_drugbank()
        data_stats = get_data_stats_drugbank(pos_pairs, num_ddi_types)
    elif dataset =='twosides':
        pos_pairs, num_drugs, num_ddi_types = get_pos_pairs_twosides()
        data_stats = get_data_stats_twosides(pos_pairs, num_ddi_types)


    sample_inductive_neg_proc(
        data_stats, drugs_old, drugs_new,
        "train", fold, dataset, num_ddi_types, seed
    )
    sample_inductive_neg_proc(
        data_stats, drugs_old, drugs_new,
        "either", fold, dataset, num_ddi_types, seed,
    )
    sample_inductive_neg_proc(
        data_stats, drugs_old, drugs_new,
        "both", fold, dataset, num_ddi_types, seed
    )




#get_pairs_drugbank()
split_transductive_for_vis(dataset='drugbank', seed=0)
#split_transductive(num_folds=3, dataset='drugbank', seed=0)

#split_inductive(fold=0, dataset='drugbank',num_ddi_types=86,seed=0)
#split_inductive(fold=1, dataset='drugbank',num_ddi_types=86,seed=1)
#split_inductive(fold=2, dataset='drugbank',num_ddi_types=86,seed=2)

#sample_inductive_neg(fold=0, dataset='drugbank', seed=0)
#sample_inductive_neg(fold=1, dataset='drugbank', seed=0)
#sample_inductive_neg(fold=2, dataset='drugbank', seed=0)

#get_pairs_twosides()

#split_transductive(num_folds=3, dataset='twosides', seed=0)

#split_inductive(fold=0, dataset='twosides',num_ddi_types=963,seed=0)
#split_inductive(fold=1, dataset='twosides',num_ddi_types=963,seed=1)
#split_inductive(fold=2, dataset='twosides',num_ddi_types=963,seed=2)

#sample_inductive_neg(fold=0, dataset='twosides', seed=0)
#sample_inductive_neg(fold=1, dataset='twosides', seed=0)
#sample_inductive_neg(fold=2, dataset='twosides', seed=0)