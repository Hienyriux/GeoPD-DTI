# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse

import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_max_pool

NUM_FP_FEATS = 1024
NUM_PROT_FEATS = 1280

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_ci(y_true, y_pred):
    row_ind, col_ind = np.tril_indices(y_true.shape[0], -1)    
    mask = (y_true[row_ind] > y_true[col_ind])
    
    row_ind_masked = row_ind[mask]
    col_ind_masked = col_ind[mask]

    total = row_ind_masked.shape[0]

    if total == 0:
        return 0

    one_cnt = (
        y_pred[row_ind_masked] > y_pred[col_ind_masked]
    ).sum().item()
    
    half_cnt = (
        y_pred[row_ind_masked] == y_pred[col_ind_masked]
    ).sum().item()
    
    acc = 1 * one_cnt + 0.5 * half_cnt
    
    return acc / total

def get_r2(y_true, y_pred):
    y_pred_centered = y_pred - y_pred.mean(keepdims=True)
    y_true_centered = y_true - y_true.mean(keepdims=True)
    
    upper = np.square((y_true_centered * y_pred_centered).sum())
    lower = np.square(y_true_centered).sum() * \
            np.square(y_pred_centered).sum()
    
    return upper / lower

def get_r0_2(y_true, y_pred):
    factor = (y_pred * y_true).sum() / np.square(y_pred).sum()
    y_true_centered = y_true - y_true.mean(keepdims=True)
    
    upper = np.square(y_true - factor * y_pred).sum()
    lower = np.square(y_true_centered).sum()
    
    return 1 - upper / lower

def get_rm2(y_true, y_pred):
    r2 = get_r2(y_true, y_pred)
    r0_2 = get_r0_2(y_true, y_pred)
    return r2 * (1 - np.sqrt(np.abs(np.square(r2) - np.square(r0_2))))

def calc_metrics(y_pred, y_true, calc_ci=True):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    if calc_ci:
        ci = get_ci(y_true, y_pred)
    else:
        ci = 0.0
    
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = get_r2(y_true, y_pred)
    r = r2 ** 0.5 if r2 > 0 else 0
    rm2 = get_rm2(y_true, y_pred)
    
    return ci, mse, r, rm2

class DTIDataset(Dataset):
    def __init__(self, dir_path, split, fold):
        if fold is None:
            if split == "train":
                load_path = f"{dir_path}/train_pairs_all.json"
            else:
                load_path = f"{dir_path}/test_pairs.json"
        else:
            load_path = f"{dir_path}/train_valid_5_fold.json"
        
        with open(load_path, "r") as f:
            pairs = json.load(f)

        if fold is None:
            self.pairs = pairs
        else:
            self.pairs = pairs[fold][split]
    
    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)

class LoadHelper:
    def __init__(self, dataset_name, device):
        self.device = device
        
        dir_path = f"dataset/{dataset_name}"
        
        load_path = f"{dir_path}/mol_graphs_3d.pt"
        self.mol_graphs = torch.load(load_path)
        
        load_path = f"{dir_path}/ecfp.pt"
        self.fp = torch.load(load_path).to(device)
        
        if dataset_name == "davis":
            load_path = f"{dir_path}/prot_graphs_3d_fps_R20_E20.pt"
        else:
            load_path = f"{dir_path}/prot_graphs_3d_fps_R20_E40.pt"
        
        self.prot_graphs = torch.load(load_path)
    
    def get_drug_batch(self, entity_list):
        graph_batch = []
        
        for entity in entity_list:
            graph = self.mol_graphs[entity]
            
            x = graph["x"]
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
            data = Data(x, edge_index, edge_attr)
            
            graph_batch.append(data)
        
        graph_batch = Batch.from_data_list(graph_batch).to(self.device)
        
        return graph_batch
    
    def get_prot_batch(self, entity_list):
        graph_batch = []
        
        for entity in entity_list:
            graph = self.prot_graphs[entity]
            
            x = graph["x"]
            edge_index = graph["edge_index"]
            edge_attr = graph["edge_attr"]
            data = Data(x, edge_index, edge_attr)
            
            graph_batch.append(data)
        
        graph_batch = Batch.from_data_list(graph_batch).to(self.device)
        
        return graph_batch
    
    def collate_fn(self, batch):
        drug_list, prot_list, y_true = zip(*batch)

        drug_list = list(drug_list)
        prot_list = list(prot_list)
        
        drug_batch = self.get_drug_batch(drug_list)
        fp_batch = self.fp[drug_list]
        
        prot_batch = self.get_prot_batch(prot_list)
        
        y_true = torch.Tensor(y_true).to(self.device)
        
        return drug_batch, fp_batch, prot_batch, y_true

class GINConv(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x, edge_index, edge_attr, size=None):
        out = self.propagate(
            edge_index, x=(x, x),
            edge_attr=edge_attr, size=size
        )
        out = self.mlp(out)      
        return out
    
    def message(self, x_j, edge_attr):
        msg = edge_attr * x_j
        return msg

class DTIPredictor(nn.Module):
    def __init__(self, dataset_name, num_node_feats, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.mol_fc = nn.Linear(num_node_feats, hidden_dim)
        self.prot_fc = nn.Linear(NUM_PROT_FEATS, hidden_dim)
        
        self.fp_mlp = nn.Sequential(
            nn.Linear(NUM_FP_FEATS, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.mol_edge_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.prot_edge_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        if dataset_name == "davis":
            mol_gnn_layers = 3
        else:
            mol_gnn_layers = 5
        
        self.mol_gnn = nn.ModuleList([
            GINConv(hidden_dim) for i in range(mol_gnn_layers)
        ])
        
        self.prot_gnn = nn.ModuleList([
            GINConv(hidden_dim) for i in range(3)
        ])
        
        self.gate = nn.GRUCell(hidden_dim, hidden_dim)
        
        if dataset_name == "davis":
            mlp_middle_dim = 512
        else:
            mlp_middle_dim = 1024
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2048),
            nn.GELU(),
            
            nn.Linear(2048, mlp_middle_dim),
            nn.GELU(),
            
            nn.Linear(mlp_middle_dim, 1)
        )
    
    def do_gnn(self, graph_batch, entity_type):        
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        graph_ind = graph_batch.batch
        
        fc = getattr(self, f"{entity_type}_fc")
        edge_mlp = getattr(self, f"{entity_type}_edge_mlp")
        gnn_module = getattr(self, f"{entity_type}_gnn")
        
        x = fc(x)
        edge_attr = edge_mlp(edge_attr)
        
        for gnn in gnn_module:
            x = gnn(x, edge_index, edge_attr)
        
        out = global_max_pool(x, graph_ind)
        
        return out
    
    def forward(self, drug_batch, fp_batch, prot_batch):
        drug_batch = self.do_gnn(drug_batch, "mol")
        prot_batch = self.do_gnn(prot_batch, "prot")
        
        fp_batch = self.fp_mlp(fp_batch)
        drug_batch = self.gate(drug_batch, fp_batch)
        
        out = torch.cat([drug_batch, prot_batch], dim=-1)
        y_out = self.mlp(out).squeeze(-1)
        
        return y_out

@torch.no_grad()
def evaluate(model, loader, set_len, save_path=None):
    cur_num = 0
    y_pred_all = []
    y_true_all = []
    
    for batch in loader:
        drug_batch, fp_batch, prot_batch, y_true = batch
        
        y_pred = model(drug_batch, fp_batch, prot_batch)
        
        y_pred_all.append(y_pred.detach().cpu())
        y_true_all.append(y_true.detach().cpu())
        
        cur_num += y_true.size(0)
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
    
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)

    if save_path is not None:
        results = {
            "y_pred" : y_pred.tolist(),
            "y_true" : y_true.tolist()
        }

        with open(save_path, "w") as f:
            json.dump(results, f)
    
    return calc_metrics(y_pred, y_true, calc_ci=True)

def train_valid(dataset_name, model, fold, num_epoch, batch_size, lr, device):
    dir_path = f"dataset/{dataset_name}"
    
    train_set = DTIDataset(dir_path, "train", fold)
    valid_set = DTIDataset(dir_path, "valid", fold)
    
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    
    load_helper = LoadHelper(dataset_name, device)
    
    train_loader = DataLoader(
        train_set, batch_size, True,
        collate_fn=load_helper.collate_fn
    )
    
    valid_loader = DataLoader(
        valid_set, batch_size, False,
        collate_fn=load_helper.collate_fn
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_stats_train = {
        "epoch" : 0,
        "CI" : 0.0,
        "MSE" : 1e7,
        "R" : 0.0,
        "Rm2" : 0.0,
    }
    
    best_stats_valid = {
        "epoch" : 0,
        "CI" : 0.0,
        "MSE" : 1e7,
        "R" : 0.0,
        "Rm2" : 0.0,
    }
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=0.001,
        step_size_up=15,
        cycle_momentum=False,
        verbose=True
    )
    
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}")
        
        train_loss = 0.0
        cur_num = 0
        
        y_pred_all = []
        y_true_all = []
        
        model.train()
        for i, batch in enumerate(train_loader):
            drug_batch, fp_batch, prot_batch, y_true = batch
            
            y_pred = model(drug_batch, fp_batch, prot_batch)
            loss = criterion(y_pred, y_true)
            
            train_loss += loss.item()
            
            y_pred_all.append(y_pred.detach().cpu())
            y_true_all.append(y_true.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_num += y_true.size(0)
            sys.stdout.write(
                f"\r{cur_num} / {train_set_len}, "
                f"{(train_loss / (i + 1)):.6f}"
                "          "
            )
            sys.stdout.flush()
        
        y_pred = torch.cat(y_pred_all)
        y_true = torch.cat(y_true_all)

        if dataset_name == "davis":
            train_ci, train_mse, train_r, train_rm2 = (
                calc_metrics(y_pred, y_true, calc_ci=True)
            )
        else:
            train_ci, train_mse, train_r, train_rm2 = (
                calc_metrics(y_pred, y_true, calc_ci=False)
            )
        
        print()
        print(
            f"Train CI: {train_ci:.4f}, "
            f"Train MSE: {train_mse:.4f}, "
            f"Train R: {train_r:.4f}, "
            f"Train Rm2: {train_rm2:.4f}"
        )
        
        sys.stdout.flush()
        
        model.eval()
        valid_ci, valid_mse, valid_r, valid_rm2 = (
            evaluate(model, valid_loader, valid_set_len)
        )
        
        print()
        print(
            f"Valid CI: {valid_ci:.4f}, "
            f"Valid MSE: {valid_mse:.4f}, "
            f"Valid R: {valid_r:.4f}, "
            f"Valid Rm2: {valid_rm2:.4f}"
        )
        
        sys.stdout.flush()
        
        if valid_ci > best_stats_valid["CI"]:
            best_stats_valid["epoch"] = epoch
            best_stats_valid["CI"] = valid_ci
            best_stats_valid["MSE"] = valid_mse
            best_stats_valid["R"] = valid_r
            best_stats_valid["Rm2"] = valid_rm2
            
            print(f"BEST VALID IN EPOCH {epoch}")
        
        print()
        print("Cur Best: Split/Epoch/CI/MSE/R/Rm2")
        
        print("Valid/%03d/%.4f/%.4f/%.4f/%.4f" % (
            best_stats_valid["epoch"],
            best_stats_valid["CI"],
            best_stats_valid["MSE"],
            best_stats_valid["R"],
            best_stats_valid["Rm2"],
        ))
        
        print()

        scheduler.step()

def train(dataset_name, model, seed, num_epoch, batch_size, lr, device):
    dir_path = f"dataset/{dataset_name}"
    
    train_set = DTIDataset(dir_path, "train", None)
    
    train_set_len = len(train_set)
    
    load_helper = LoadHelper(dataset_name, device)
    
    train_loader = DataLoader(
        train_set, batch_size, True,
        collate_fn=load_helper.collate_fn
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_stats_train = {
        "epoch" : 0,
        "CI" : 0.0,
        "MSE" : 1e7,
        "R" : 0.0,
        "Rm2" : 0.0,
    }
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0001,
        max_lr=0.001,
        step_size_up=15,
        cycle_momentum=False,
        verbose=True
    )
    
    for epoch in range(num_epoch):
        print(f"Epoch: {epoch}")
        
        train_loss = 0.0
        cur_num = 0
        
        y_pred_all = []
        y_true_all = []
        
        model.train()
        for i, batch in enumerate(train_loader):
            drug_batch, fp_batch, prot_batch, y_true = batch
            
            y_pred = model(drug_batch, fp_batch, prot_batch)
            loss = criterion(y_pred, y_true)
            
            train_loss += loss.item()
            
            y_pred_all.append(y_pred.detach().cpu())
            y_true_all.append(y_true.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_num += y_true.size(0)
            sys.stdout.write(
                f"\r{cur_num} / {train_set_len}, "
                f"{(train_loss / (i + 1)):.6f}"
                "          "
            )
            sys.stdout.flush()
        
        y_pred = torch.cat(y_pred_all)
        y_true = torch.cat(y_true_all)

        if dataset_name == "davis":
            train_ci, train_mse, train_r, train_rm2 = (
                calc_metrics(y_pred, y_true, calc_ci=True)
            )
        else:
            train_ci, train_mse, train_r, train_rm2 = (
                calc_metrics(y_pred, y_true, calc_ci=False)
            )
        
        print()
        print(
            f"Train CI: {train_ci:.4f}, "
            f"Train MSE: {train_mse:.4f}, "
            f"Train R: {train_r:.4f}, "
            f"Train Rm2: {train_rm2:.4f}"
        )
        
        sys.stdout.flush()
        
        print()
        
        scheduler.step()

    os.makedirs("model", exist_ok=True)
    
    save_path = f"model/model_{dataset_name}_{num_epoch}_{seed}.pt"
    torch.save(model.state_dict(), save_path)

def test(dataset_name, model, batch_size, device):
    dir_path = f"dataset/{dataset_name}"
    
    test_set = DTIDataset(dir_path, "test", None)
    test_set_len = len(test_set)
    
    load_helper = LoadHelper(dataset_name, device)
    
    test_loader = DataLoader(
        test_set, batch_size, False,
        collate_fn=load_helper.collate_fn
    )
    
    model.eval()
    test_ci, test_mse, test_r, test_rm2 = (
        evaluate(model, test_loader, test_set_len)
    )
    
    print()
    print(
        f"Test CI: {test_ci:.4f}, "
        f"Test MSE: {test_mse:.4f}, "
        f"Test R: {test_r:.4f}, "
        f"Test Rm2: {test_rm2:.4f}"
    )

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_name", type=str,
        choices=["davis", "kiba"], default="davis"
    )

    parser.add_argument(
        "--mode", type=str,
        choices=["train_valid", "train", "test"], default="train"
    )
    
    parser.add_argument(
        "--fold", type=int,
        choices=[0, 1, 2, 3, 4], default=0
    )

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.dataset_name == "davis":
        args.num_node_feats = 29
    else:
        args.num_node_feats = 34

    return args

if __name__ == "__main__":
    args = get_args()
    
    set_all_seeds(args.seed)
    
    model = DTIPredictor(
        args.dataset_name, args.num_node_feats, args.hidden_dim
    ).to(args.device)

    if args.mode == "train_valid":
        train_valid(
            args.dataset_name, model, args.fold,
            args.num_epoch, args.batch_size, args.lr, args.device
        )
    
    elif args.mode == "train":
        train(
            args.dataset_name, model, args.seed,
            args.num_epoch, args.batch_size, args.lr, args.device
        )
    
    else:
        load_path = (
            f"model/model_{args.dataset_name}_"
            f"{args.num_epoch}_{args.seed}.pt"
        )
        model.load_state_dict(torch.load(load_path, map_location=args.device))
        test(args.dataset_name, model, args.batch_size, args.device)
