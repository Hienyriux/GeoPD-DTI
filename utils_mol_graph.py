# -*- coding: utf-8 -*-

import sys
import json

import torch
import torch.nn.functional as F

from rdkit import Chem
from rdkit import RDLogger

def set_feat_dict(mol, feat_dicts_raw):
    for atom in mol.GetAtoms():
        feat_dicts_raw[0].add(atom.GetSymbol())
        feat_dicts_raw[1].add(atom.GetDegree())
        feat_dicts_raw[2].add(atom.GetImplicitValence())
        feat_dicts_raw[3].add(atom.GetFormalCharge())
        feat_dicts_raw[4].add(atom.GetNumRadicalElectrons())
        feat_dicts_raw[5].add(int(atom.GetHybridization()))
        feat_dicts_raw[6].add(atom.GetTotalNumHs())
        feat_dicts_raw[7].add(int(atom.GetIsAromatic()))

def get_node_feat(mol, feat_dicts_raw):
    atom_idx_to_node_idx = {}
    x = [[] for i in range(8)]
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom_idx_to_node_idx[atom.GetIdx()] = i
        
        x[0].append(feat_dicts_raw[0][atom.GetSymbol()])
        x[1].append(feat_dicts_raw[1][atom.GetDegree()])
        x[2].append(feat_dicts_raw[2][atom.GetImplicitValence()])
        x[3].append(feat_dicts_raw[3][atom.GetFormalCharge()])
        x[4].append(feat_dicts_raw[4][atom.GetNumRadicalElectrons()])
        x[5].append(feat_dicts_raw[5][int(atom.GetHybridization())])
        x[6].append(feat_dicts_raw[6][atom.GetTotalNumHs()])
        x[7].append(feat_dicts_raw[7][int(atom.GetIsAromatic())])
    
    feat_dim = [len(feat) for feat in feat_dicts_raw]
    
    for i in range(8):
        cur = torch.LongTensor(x[i])
        cur = F.one_hot(cur, feat_dim[i])
        x[i] = cur
    
    x = torch.cat(x, dim=-1)
    x = x.float()
    
    return x, atom_idx_to_node_idx

def get_edge_index(mol, atom_idx_to_node_idx):
    cur_edge_index = []
    
    for bond in mol.GetBonds():
        atom_1 = bond.GetBeginAtomIdx()
        atom_2 = bond.GetEndAtomIdx()
        
        node_1 = atom_idx_to_node_idx[atom_1]
        node_2 = atom_idx_to_node_idx[atom_2]

        cur_edge_index.append([node_1, node_2])
        cur_edge_index.append([node_2, node_1])
    
    if len(cur_edge_index) > 0:
        cur_edge_index = torch.LongTensor(cur_edge_index).t()
    else:
        cur_edge_index = torch.LongTensor(2, 0)
    
    return cur_edge_index

def proc_feat_dicts(dir_path, feat_dicts_raw):
    dict_names = [
        "symbol", "deg", "valence", "charge",
        "electron", "hybrid", "hydrogen", "aromatic"
    ]
    
    feat_dicts = {}
    
    output_lines = ["{\n"]
    
    for feat_dict, dict_name in zip(feat_dicts_raw, dict_names):
        feat_dict = sorted(list(feat_dict))
        feat_dict = {item : i for i, item in enumerate(feat_dict)}
        feat_dicts[dict_name] = feat_dict
    
    save_path = f"{dir_path}/feat_dict.json"
    with open(save_path, "w") as f:
        json.dump(feat_dicts, f, indent=2)

    feat_dicts = list(feat_dicts.values())
    
    return feat_dicts

def get_mol_graphs(dataset_name):
    RDLogger.DisableLog("rdApp.*")
    
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/smiles.json"
    with open(load_path, "r") as f:
        smiles_list = json.load(f)
    
    feat_dicts_raw = [set() for i in range(8)]
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        
        set_feat_dict(mol, feat_dicts_raw)
    
    feat_dicts = proc_feat_dicts(dir_path, feat_dicts_raw)
    
    mol_graphs = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        
        cur_x, atom_idx_to_node_idx = get_node_feat(mol, feat_dicts)
        cur_edge_index = get_edge_index(mol, atom_idx_to_node_idx)
        
        mol_graphs.append({"x" : cur_x, "edge_index" : cur_edge_index})

        sys.stdout.write(f"\r{i} / {len(smiles_list)}")
        sys.stdout.flush()

    print()
    print("Data:", len(mol_graphs))
    print("Feat:", mol_graphs[0]["x"].size(-1))
    
    save_path = f"{dir_path}/mol_graphs.pt"
    torch.save(mol_graphs, save_path)

#get_mol_graphs("davis")
#get_mol_graphs("kiba")
