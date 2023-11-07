# -*- coding: utf-8 -*-

import sys
import json

import torch

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

def get_ecfp(dataset_name, num_feats, radius):
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/smiles.json"
    with open(load_path, "r") as f:
        smiles_list = json.load(f)
    
    fp_list = []
    
    for i, smiles in enumerate(smiles_list):
        sys.stdout.write(f"\r{i} / {len(smiles_list)}")
        sys.stdout.flush()
        
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        
        fp_obj = GetMorganFingerprintAsBitVect(
            mol, radius, nBits=num_feats
        )
        fp_vec = torch.Tensor(fp_obj.ToList())
        
        fp_list.append(fp_vec)
    
    print()
    
    fp_list = torch.stack(fp_list)
    
    save_path = f"{dir_path}/ecfp.pt"
    torch.save(fp_list, save_path)

#get_ecfp("davis", 1024, 3)
#get_ecfp("kiba", 1024, 3)
