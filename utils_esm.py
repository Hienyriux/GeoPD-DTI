# -*- coding: utf-8 -*-

import os
import sys
import json

import torch
import esm

MY_DEVICE = "cuda"

torch.hub.set_dir(".")

def get_prot_batch(dataset_name, batch_size, start_idx):
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/prot_seq_raw.json"
    with open(load_path, "r") as f:
        prot_list = json.load(f)
    
    num_prots = len(prot_list)
    print(f"Proteins: {num_prots}")
    
    batch_list = []
    for i in range(start_idx, num_prots, batch_size):
        batch = prot_list[i : i + batch_size]
        batch_label = list(map(str, range(i, i + batch_size)))
        batch_list.append(list(zip(batch_label, batch)))
    
    return batch_list, num_prots

def get_aa(dataset_name, layer_idx, start_idx=0):
    dir_path = f"dataset/{dataset_name}"
    
    save_dir_path_esm = f"{dir_path}/esm"
    os.makedirs(save_dir_path_esm, exist_ok=True)
        
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(MY_DEVICE)
    
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    batch_list, num_prots = get_prot_batch(dataset_name, 1, start_idx)
    cur_num = start_idx
    
    for batch in batch_list:
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(dim=1)
        
        batch_tokens = batch_tokens.to(MY_DEVICE)
        
        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[layer_idx],
                return_contacts=True
            )
        
        token_rep_batch = results["representations"][layer_idx]
        
        for token_rep, tokens_len, batch_label in \
            zip(token_rep_batch, batch_lens, batch_labels):
            
            cur_token_rep = token_rep
            cur_token_rep = cur_token_rep.cpu()
            
            save_path = f"{save_dir_path_esm}/{cur_num}.pt"
            torch.save(cur_token_rep, save_path)
        
        cur_num += 1
        
        sys.stdout.write(f"\r{cur_num} / {num_prots}")
        sys.stdout.flush()
    
    print()

#get_aa("davis", 33, 0)
#get_aa("kiba", 33, 0)
