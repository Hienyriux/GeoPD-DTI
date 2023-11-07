# -*- coding: utf-8 -*-

import os
import sys
import json

import numpy as np

import scipy.sparse as sparse
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

import torch

from torch_geometric import seed_everything
from torch_geometric.nn import fps

from rdkit import Chem
from rdkit.Chem import AllChem

def get_mol_3d(dataset_name):
    dir_path = f"dataset/{dataset_name}"
    
    save_dir_path = f"{dir_path}/mol_3d"
    os.makedirs(save_dir_path, exist_ok=True)
    
    load_path = f"{dir_path}/smiles.json"
    with open(load_path, "r") as f:
        smiles_list = json.load(f)
    
    for i, smiles in enumerate(smiles_list):
        sys.stdout.write(f"\r{i} / {len(smiles_list)}")
        sys.stdout.flush()
        
        mol = Chem.MolFromSmiles(smiles)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h)
        
        save_path = f"{save_dir_path}/{i}.sdf"
        Chem.MolToMolFile(mol_h, save_path)
    
    print()

def get_coord_lines(dir_path, mol_idx):
    load_path = f"{dir_path}/mol_3d/{mol_idx}.sdf"
    with open(load_path, "r") as f:
        lines = f.readlines()
    
    pos = 0

    coord_start = -1
    
    while pos < len(lines):
        line = lines[pos]
        num_parts = len(line[ : -1].split())

        if num_parts == 16:
            coord_start = pos
            break

        pos += 1

    coord_end = pos
    
    while pos < len(lines):
        line = lines[pos]
        num_parts = len(line[ : -1].split())

        if num_parts != 16:
            coord_end = pos
            break

        pos += 1

    coord_lines = lines[coord_start : coord_end]

    pos = len(coord_lines) - 1
    
    while pos > -1:
        line = coord_lines[pos]
        parts = line[ : -1].split()
        sym = parts[3]
        
        if sym != "H":
            pos += 1
            break
        
        pos -= 1

    heavy_end = pos
    coord_lines = coord_lines[ : heavy_end]

    return coord_lines

def get_coord(dataset_name):
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/mol_graphs.pt"
    mol_graphs = torch.load(load_path)
    
    coord_list = []
    
    for i, mol_graph in enumerate(mol_graphs):
        coord_lines = get_coord_lines(dir_path, i)
        
        x = mol_graph["x"]
        num_nodes = x.size(0)
        
        assert len(coord_lines) == num_nodes
        
        cur_coord_list = []
        
        for line in coord_lines:
            parts = line[ : -1].split()
            coord = parts[ : 3]
            coord = list(map(float, coord))
            cur_coord_list.append(coord)
        
        coord_list.append(cur_coord_list)
    
    save_path = f"{dir_path}/mol_coord.json"
    with open(save_path, "w") as f:
        json.dump(coord_list, f)

def get_dis_mat(dataset_name, entity_type, suffix):
    prefix = f"dataset/{dataset_name}/{entity_type}"
    
    if suffix is not None:
        load_path = f"{prefix}_coord_{suffix}.json"
    else:
        load_path = f"{prefix}_coord.json"
    
    with open(load_path, "r") as f:
        coord_list = json.load(f)
    
    dis_list = []
    
    for coord in coord_list:
        coord = torch.Tensor(coord)
        cur_dis = torch.cdist(coord, coord)
        dis_list.append(cur_dis)
    
    if suffix is not None:
        save_path = f"{prefix}_dis_mat_{suffix}.pt"
    else:
        save_path = f"{prefix}_dis_mat.pt"
    
    torch.save(dis_list, save_path)

def get_shortest_path(dataset_name):
    dir_path = f"dataset/{dataset_name}"

    load_path = f"{dir_path}/mol_graphs.pt"
    mol_graphs = torch.load(load_path)

    sp_list = []
    
    for mol_graph in mol_graphs:
        x = mol_graph["x"]
        num_nodes = x.size(0)

        edge_index = mol_graph["edge_index"]

        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        adj = adj.numpy()

        sp = shortest_path(adj, directed=False)
        sp = sp.astype(np.float32)
        sp = torch.from_numpy(sp)
        
        sp_list.append(sp)

    save_path = f"{dir_path}/mol_sp.pt"
    torch.save(sp_list, save_path)

def get_mol_graphs_3d(dataset_name, threshold):
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/mol_dis_mat.pt"
    dis_list = torch.load(load_path)
    
    load_path = f"{dir_path}/mol_sp.pt"
    sp_list = torch.load(load_path)

    load_path = f"{dir_path}/mol_graphs.pt"
    mol_graphs_raw = torch.load(load_path)
    
    mol_graphs = []
    
    for dis_mat, sp_mat, mol_graph in zip(dis_list, sp_list, mol_graphs_raw):
        num_nodes = dis_mat.size(0)
        
        adj_mat = (sp_mat == 1).float()
        
        mask = (dis_mat < threshold)
        edge_index = mask.nonzero().t()

        src, tgt = edge_index

        adj_attr = adj_mat[tgt, src]
        sp_attr = sp_mat[tgt, src]
        dis_attr = dis_mat[tgt, src]
        
        sp_attr = sp_attr + 1
        dis_attr = dis_attr + 1
        
        sp_attr = 1.0 / sp_attr
        dis_attr = 1.0 / dis_attr
        
        mol_graphs.append({
            "x" : mol_graph["x"],
            "edge_index" : edge_index,
            "edge_attr" : torch.stack([adj_attr, sp_attr, dis_attr], dim=-1)
        })
    
    save_path = f"{dir_path}/mol_graphs_3d.pt"
    torch.save(mol_graphs, save_path)

def get_break_ignored(seq, seq_pdb, break_list):
    flag_list = [1] * len(seq)

    for start, end in break_list:
        for j in range(start, end):
            flag_list[j] = 0

    seq_proc = []

    for cur, flag in zip(seq, flag_list):
        if flag == 0:
            continue

        seq_proc.append(cur)

    seq_proc = "".join(seq_proc)

    assert seq_pdb == seq_proc
    
    return flag_list

def get_aa_coord(dataset_name):
    load_path = "dataset/aa_name_to_sym.json"
    with open(load_path, "r") as f:
        aa_name_to_sym = json.load(f)
    
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/prot_seq_raw.json"
    with open(load_path, "r") as f:
        seq_list = json.load(f)
    
    load_dir_path = f"{dir_path}/pdb_crawled"
    file_list = os.listdir(load_dir_path)
    
    coord_list = []
    modified_dict = {}
    
    for i, seq in enumerate(seq_list):
        load_path = f"{dir_path}/pdb/{i}.pdb"
        
        if not os.path.exists(load_path):
            cur_file_list = list(filter(
                lambda x: x.startswith(f"{i}"),
                file_list
            ))
            
            assert len(cur_file_list) == 1

            file_name = cur_file_list[0]
            load_path = f"{load_dir_path}/{file_name}"
        
        with open(load_path, "r") as f:
            lines = f.readlines()[1 : -1]
        
        aa_coord = []
        
        break_list = []
        last_aa_id = 0
        
        for line in lines:
            if not line.startswith("ATOM"):
                continue
            
            atom_name = line[12 : 16].strip()
            alt_loc = line[16]
            aa_name = line[17 : 20].strip()
            chain_name = line[21]
            aa_id = int(line[22 : 26].strip())
            
            if atom_name != "CA":
                continue

            if alt_loc == "B":
                continue
            
            if chain_name != "A":
                continue

            if aa_id != last_aa_id + 1:
                break_list.append((last_aa_id, aa_id - 1))
            
            last_aa_id = aa_id
            
            aa_sym = aa_name_to_sym[aa_name]
            
            if len(line) < 30:
                x_coord = 0.0
                y_coord = 0.0
                z_coord = 0.0
            else:
                x_coord = float(line[30 : 38])
                y_coord = float(line[38 : 46])
                z_coord = float(line[46 : 54])
            
            aa_coord.append((aa_sym, x_coord, y_coord, z_coord))
        
        seq = seq.replace("X", "")
        seq_pdb = "".join([cur[0] for cur in aa_coord])
        
        if dataset_name == "kiba" and i == 128:
            modified_dict[i] = get_break_ignored(seq, seq_pdb, break_list)
        else:
            assert seq_pdb == seq, f"{i}"
        
        cur_coord_list = [cur[1 : ] for cur in aa_coord]
        coord_list.append(cur_coord_list)
    
    save_path = f"{dir_path}/prot_aa_coord.json"
    with open(save_path, "w") as f:
        json.dump(coord_list, f)
    
    if len(modified_dict) == 0:
        return
    
    save_path = f"{dir_path}/prot_modified.json"
    with open(save_path, "w") as f:
        json.dump(modified_dict, f)

def partition_prot_fps(dataset_name, part_len, r, seed):
    seed_everything(seed)
    
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/prot_seq_raw.json"
    with open(load_path, "r") as f:
        seq_list = json.load(f)
    
    load_path = f"{dir_path}/prot_aa_coord.json"
    with open(load_path, "r") as f:
        coord_list = json.load(f)
    
    modified_dict = {}
    
    load_path = f"{dir_path}/prot_modified.json"
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            modified_dict = json.load(f)
    
    ratio = 1.0 / part_len
    
    feat_list = []
    
    coord_mean_list = []
    seq_pos_mean_list = []
    
    for i, (seq, coord) in enumerate(zip(seq_list, coord_list)):
        sys.stdout.write(f"\r{i} / {len(seq_list)}")
        sys.stdout.flush()
        
        coord = np.array(coord)
        seq_pos = np.arange(coord.shape[0])[ : , None]
        
        load_path = f"{dir_path}/esm/{i}.pt"
        esm_feat = torch.load(load_path)
        
        assert esm_feat.size(0) == len(seq) + 2
        
        esm_feat = esm_feat[1 : -1]
        
        if str(i) in modified_dict:
            flag_list = np.array(modified_dict[str(i)])
            ind = np.nonzero(flag_list)[0].tolist()
        else:
            seq_with_idx = list(enumerate(seq))
            seq_with_idx = list(filter(lambda x: x[1] != "X", seq_with_idx))
            ind, _ = zip(*seq_with_idx)
            ind = list(ind)
        
        seq_len = len(ind)
        
        esm_feat = esm_feat[ind]
        assert coord.shape[0] == seq_len
        
        fps_ind = fps(
            torch.from_numpy(coord.astype(np.float32)),
            ratio=ratio,
            random_start=True
        )
        
        fps_ind = fps_ind.tolist()
        
        center_coord = coord[fps_ind]
        dist = cdist(center_coord, coord)
        
        mask = (dist < r)
        
        row_ind, col_ind = np.nonzero(mask)
        row_ind = row_ind.tolist()
        col_ind = col_ind.tolist()
        
        adj = mask.astype(np.float64)        
        num_neighbors = adj.sum(axis=-1, keepdims=True)
        
        aggr_coord = adj @ coord
        aggr_seq_pos = adj @ seq_pos
        
        aggr_coord = aggr_coord / num_neighbors
        aggr_coord = aggr_coord.tolist()
        
        aggr_seq_pos = aggr_seq_pos / num_neighbors
        aggr_seq_pos = aggr_seq_pos[ : , 0].tolist()
        
        adj = torch.from_numpy(adj.astype(np.float32))
        aggr_feat = adj @ esm_feat
        
        feat_list.append(aggr_feat)
        coord_mean_list.append(aggr_coord)
        seq_pos_mean_list.append(aggr_seq_pos)
    
    print()
    
    save_path = f"{dir_path}/prot_super_feat_fps_R{r}.pt"
    torch.save(feat_list, save_path)
    
    save_path = f"{dir_path}/prot_super_coord_fps_R{r}.json"
    with open(save_path, "w") as f:
        json.dump(coord_mean_list, f)
    
    save_path = f"{dir_path}/prot_super_seq_pos_fps_R{r}.json"
    with open(save_path, "w") as f:
        json.dump(seq_pos_mean_list, f)

def get_prot_graphs_3d(dataset_name, r, threshold):
    dir_path = f"dataset/{dataset_name}"
    
    load_path = f"{dir_path}/prot_super_dis_mat_fps_R{r}.pt"    
    dis_list = torch.load(load_path)
    
    load_path = f"{dir_path}/prot_super_seq_pos_fps_R{r}.json"
    with open(load_path, "r") as f:
        seq_pos_list = json.load(f)
    
    load_path = f"{dir_path}/prot_super_feat_fps_R{r}.pt"
    esm_feat_list = torch.load(load_path)
    
    prot_graphs = []
    
    for dis_mat, seq_pos, esm_feat in \
        zip(dis_list, seq_pos_list, esm_feat_list):
        
        num_nodes = dis_mat.size(0)
        
        pos = torch.Tensor(seq_pos).unsqueeze(-1)
        pos_mat = torch.cdist(pos, pos)

        assert tuple(dis_mat.size()) == tuple(pos_mat.size())
        
        mask = (dis_mat < threshold)
        edge_index = mask.nonzero().t()
        
        src, tgt = edge_index
        
        pos_attr = pos_mat[tgt, src]
        dis_attr = dis_mat[tgt, src]
        
        pos_attr = pos_attr + 1
        dis_attr = dis_attr + 1
        
        pos_attr = 1.0 / pos_attr
        dis_attr = 1.0 / dis_attr
        
        prot_graphs.append({
            "x" : esm_feat,
            "edge_index" : edge_index,
            "edge_attr" : torch.stack([pos_attr, dis_attr], dim=-1)
        })
    
    save_path = f"{dir_path}/prot_graphs_3d_fps_R{r}_E{threshold}.pt"
    torch.save(prot_graphs, save_path)

#get_mol_3d("davis")
#get_mol_3d("kiba")

#get_coord("davis")
#get_coord("kiba")

#get_dis_mat("davis", "mol", None)
#get_dis_mat("kiba", "mol", None)

#get_shortest_path("davis")
#get_shortest_path("kiba")

#get_mol_graphs_3d("davis", 10)
#get_mol_graphs_3d("kiba", 10)

#get_aa_coord("davis")
#get_aa_coord("kiba")

#partition_prot_fps("davis", 24, 20, 0)
#partition_prot_fps("kiba", 24, 20, 0)

#get_dis_mat("davis", "prot_super", "fps_R20")
#get_dis_mat("kiba", "prot_super", "fps_R20")

#get_prot_graphs_3d("davis", 20, 20)
#get_prot_graphs_3d("kiba", 20, 40)
