# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import pickle

import numpy as np
import torch

def get_drug_prot(dataset_name):
    save_dir_path = f"dataset/{dataset_name}"
    os.makedirs(save_dir_path, exist_ok=True)
    
    load_dir_path = f"dataset/raw_data/{dataset_name}"
    
    load_path = f"{load_dir_path}/ligands_iso.txt"
    with open(load_path, "r") as f:
        drug_id_to_smiles = json.load(f)
    
    print(f"Drugs: {len(drug_id_to_smiles)}")
    
    drug_dict = {
        drug_id : i for i, drug_id \
        in enumerate(drug_id_to_smiles.keys())
    }

    smiles_list = list(drug_id_to_smiles.values())

    save_path = f"{save_dir_path}/drug_dict.json"
    with open(save_path, "w") as f:
        json.dump(drug_dict, f, indent=2)

    save_path = f"{save_dir_path}/smiles.json"
    with open(save_path, "w") as f:
        json.dump(smiles_list, f, indent=2)
    
    load_path = f"{load_dir_path}/proteins.txt"
    with open(load_path, "r") as f:
        prot_id_to_seq = json.load(f)
    
    print(f"Proteins: {len(prot_id_to_seq)}")
    
    prot_dict = {
        prot_id : i for i, prot_id \
        in enumerate(prot_id_to_seq.keys())
    }

    seq_list = list(prot_id_to_seq.values())
    
    save_path = f"{save_dir_path}/prot_dict.json"
    with open(save_path, "w") as f:
        json.dump(prot_dict, f, indent=2)

    save_path = f"{save_dir_path}/prot_seq_raw.json"
    with open(save_path, "w") as f:
        json.dump(seq_list, f, indent=2)

def get_label(dataset_name):
    save_dir_path = f"dataset/{dataset_name}"
    os.makedirs(save_dir_path, exist_ok=True)
    
    load_dir_path = f"dataset/raw_data/{dataset_name}"
    
    load_path = f"{load_dir_path}/Y"
    with open(load_path, "rb") as f:
        label = pickle.load(f, encoding="latin1")
    
    if dataset_name == "davis":
        label = -np.log10(label / 1e9)
    
    row_ind, col_ind = np.where(~np.isnan(label))

    row_ind = row_ind.tolist()
    col_ind = col_ind.tolist()
    
    save_path = f"{save_dir_path}/row_col_ind.json"
    with open(save_path, "w") as f:
        json.dump({"row_ind" : row_ind, "col_ind" : col_ind}, f)
    
    label = label[row_ind, col_ind]
    print(label.dtype, label.shape)
    
    label = label.tolist()
    
    save_path = f"{save_dir_path}/label.json"
    with open(save_path, "w") as f:
        json.dump(label, f)

def get_split(dataset_name):
    save_dir_path = f"dataset/{dataset_name}"
    os.makedirs(save_dir_path, exist_ok=True)
    
    load_path = f"{save_dir_path}/row_col_ind.json"
    with open(load_path, "r") as f:
        row_col_ind = json.load(f)
    
    load_path = f"{save_dir_path}/label.json"
    with open(load_path, "r") as f:
        label = json.load(f)
    
    row_ind = np.array(row_col_ind["row_ind"], dtype=np.int64)
    col_ind = np.array(row_col_ind["col_ind"], dtype=np.int64)
    label = np.array(label)
    
    load_dir_path = f"dataset/raw_data/{dataset_name}"
    
    load_path = f"{load_dir_path}/folds/train_fold_setting1.txt"
    with open(load_path, "r") as f:
        train_pair_ind_list = json.load(f)
    
    load_path = f"{load_dir_path}/folds/test_fold_setting1.txt"
    with open(load_path, "r") as f:
        test_pair_ind = json.load(f)
    
    train_valid_pairs_list = []
    
    for train_pair_ind in train_pair_ind_list:
        train_row_ind = row_ind[train_pair_ind].tolist()
        train_col_ind = col_ind[train_pair_ind].tolist()
        train_label = label[train_pair_ind].tolist()
        
        train_valid_pairs_list.append(list(zip(
            train_row_ind, train_col_ind, train_label
        )))
    
    test_row_ind = row_ind[test_pair_ind].tolist()
    test_col_ind = col_ind[test_pair_ind].tolist()
    test_label = label[test_pair_ind].tolist()
    
    test_pairs = list(zip(
        test_row_ind, test_col_ind, test_label
    ))
    
    print(f"Test: {len(test_pairs)}")
    
    num_folds = len(train_valid_pairs_list)
    
    train_valid_split = []
    train_pairs_all = []
    
    for fold in range(num_folds):
        train_pairs = []
        valid_pairs = None
        
        for i, train_valid_pairs in enumerate(train_valid_pairs_list):
            if i != fold:
                train_pairs += train_valid_pairs
            else:
                valid_pairs = train_valid_pairs
        
        print(
            f"Fold: {fold}, "
            f"Train: {len(train_pairs)}, "
            f"Valid: {len(valid_pairs)}"
        )

        if fold == 0:
            train_pairs_all = train_pairs + valid_pairs
        
        train_valid_split.append({
            "train" : train_pairs,
            "valid" : valid_pairs,
        })
    
    save_path = f"{save_dir_path}/train_valid_5_fold.json"
    with open(save_path, "w") as f:
        json.dump(train_valid_split, f)
    
    save_path = f"{save_dir_path}/train_pairs_all.json"
    with open(save_path, "w") as f:
        json.dump(train_pairs_all, f)

    save_path = f"{save_dir_path}/test_pairs.json"
    with open(save_path, "w") as f:
        json.dump(test_pairs, f)

#get_drug_prot("davis")
#get_drug_prot("kiba")

#get_label("davis")
#get_label("kiba")

#get_split("davis")
#get_split("kiba")
