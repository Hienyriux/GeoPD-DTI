# -*- coding: utf-8 -*-

import os
import sys
import json

import torch
import esm

MY_DEVICE = "cuda"

torch.hub.set_dir(".")

def get_pdb(dataset_name, excluded_list, start_idx=0, end_idx=None):
    # start_idx: inclusive
    # end_idx: exclusive
    
    dir_path = f"dataset/{dataset_name}"
    
    save_dir_path = f"{dir_path}/pdb"
    os.makedirs(save_dir_path, exist_ok=True)
    
    load_path = f"{dir_path}/prot_seq_raw.json"
    with open(load_path, "r") as f:
        seq_list = json.load(f)
    
    if end_idx is not None:
        seq_list = seq_list[ : end_idx]
    
    seq_list_len = len(seq_list)
    seq_list = seq_list[start_idx : ]
    
    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(MY_DEVICE)
    
    model.set_chunk_size(128)
    
    for i, seq in enumerate(seq_list):
        sys.stdout.write(f"\r{start_idx+i} / {seq_list_len}")
        sys.stdout.flush()
        
        if start_idx + i in excluded_list:
            continue
        
        save_path = f"{save_dir_path}/{start_idx+i}.pdb"

        if os.path.exists(save_path):
            continue
        
        with torch.no_grad():
            output = model.infer_pdb(seq)
        
        with open(save_path, "w") as f:
            f.write(output)
    
    print()

#get_pdb("davis", [215, 216, 263, 368], 0)
#get_pdb("kiba", [43, 89, 128, 178], 0)
