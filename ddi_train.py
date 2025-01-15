# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics

from datasets.ddi_dataset import DDIDataset, BatchLoader
from datasets.ddi_dataset_wo_type import DDIDataset as DDIDataset_WT
from datasets.ddi_dataset_wo_type import BatchLoader as BatchLoader_WT

from torch.utils.data import DataLoader
from model.gnn_model import GNN
import torch.optim as optim
from model.ddi_predictor import InteractionPredictor


def calc_metrics(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    y_pred_label = (y_pred >= 0.5).astype(np.int32)
    
    acc = metrics.accuracy_score(y_true, y_pred_label)
    auc = metrics.roc_auc_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred_label, zero_division=0)
    
    p = metrics.precision_score(y_true, y_pred_label, zero_division=0)
    r = metrics.recall_score(y_true, y_pred_label, zero_division=0)
    ap = metrics.average_precision_score(y_true, y_pred)
    
    return acc, auc, f1, p, r, ap

def get_drop_rate_stats(drop_rate_list):
    drop_rate_stats = {
        "max" : 0.0,
        "min" : 0.0,
        "mean" : 0.0
    }
    
    if len(drop_rate_list) == 0:
        return drop_rate_stats
    
    drop_rate_stats["max"] = max(drop_rate_list)
    drop_rate_stats["min"] = min(drop_rate_list)
    drop_rate_stats["mean"] = sum(drop_rate_list) / len(drop_rate_list)

    return drop_rate_stats

@torch.no_grad()
def evaluate(models, loader, set_len):
    cur_num = 0
    y_pred_all, y_true_all = [], []
    model=models[0]
    model_predictor=models[1]
    for batch in loader:
        graph_batch_1, graph_batch_2, ddi_type, y_true = batch
        batch_size = len(ddi_type)

        node_rep_1 = model(graph_batch_1.x, graph_batch_1.edge_index, graph_batch_1.edge_attr)
        node_rep_2 = model(graph_batch_2.x, graph_batch_2.edge_index, graph_batch_2.edge_attr)

        num_part_1 = graph_batch_1.num_part
        num_part_2 = graph_batch_2.num_part

        node_rep_1, motif_rep_1, super_node_rep_1 = group_node_rep(node_rep_1, batch_size, num_part_1)
        node_rep_2, motif_rep_2, super_node_rep_2 = group_node_rep(node_rep_2, batch_size, num_part_2)

        y_pred=model_predictor.forward_func(super_node_rep_1, super_node_rep_2, node_rep_1, node_rep_2, motif_rep_1, motif_rep_2,
                                ddi_type)
        
        y_pred_all.append(y_pred.detach().sigmoid().cpu())
        y_true_all.append(torch.LongTensor(y_true))
        
        cur_num += graph_batch_1.num_graphs // 2
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
    
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)

    return calc_metrics(y_pred, y_true)


def group_node_rep(node_rep, batch_size, num_part):
    group = []
    motif_group = []
    super_group = []

    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        motif_group.append(node_rep[count + num_atom:count + num_all -1])
        super_group.append(node_rep[count + num_all -1])
        count += num_all
    return group, motif_group, super_group


def train(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.dataset == "drugbank" or args.dataset == "twosides" :
        train_set = DDIDataset(args.dataset, "train", args.fold)
        valid_set = DDIDataset(args.dataset, "valid", args.fold)
        test_set = DDIDataset(args.dataset, "test", args.fold)
        batch_loader = BatchLoader(args)
        #forward_func = model.forward_transductive
    else:
        train_set = DDIDataset_WT(args.dataset, "train")
        valid_set = DDIDataset_WT(args.dataset, "valid")
        test_set = DDIDataset_WT(args.dataset, "test")
        batch_loader = BatchLoader_WT(args)
        #forward_func = model.forward_wo_type
    
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    test_set_len = len(test_set)



    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )
    test_loader = DataLoader(
        test_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn
    )


    max_valid_acc, max_test_acc, max_valid_auc, max_valid_f1, max_valid_ap, max_test_acc, max_test_auc, max_test_f1, max_test_ap,  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0
    valid_best_epoch = 0
    test_best_epoch = 0
    model = GNN(args.num_layer, args.hidden_size, args.num_node_feats, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)

    model_predictor = InteractionPredictor(args).to(args.device)

    optimizer = optim.Adam([{"params": model.parameters()}, {"params": model_predictor.parameters()}], lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (1.0 if epoch < 200 else 0.1),
        last_epoch=args.start_epoch - 1
    )

    for epoch in range(args.num_epoch):
        print(f"Epoch: {args.start_epoch + epoch}")

        train_loss = 0.0
        cur_num = 0
        y_pred_all, y_true_all = [], []
        train_set.do_shuffle()
        
        model.train()
        model_predictor.train()

        for i, batch in enumerate(train_loader):
            graph_batch_1, graph_batch_2, ddi_type, y_true = batch
            y_true = torch.Tensor(y_true).to(args.device)
            batch_size = len(ddi_type)

            graph_batch_1 = graph_batch_1.to(device)
            graph_batch_2 = graph_batch_2.to(device)

            node_rep_1 = model(graph_batch_1.x, graph_batch_1.edge_index, graph_batch_1.edge_attr)

            node_rep_2 = model(graph_batch_2.x, graph_batch_2.edge_index, graph_batch_2.edge_attr)

            num_part_1 = graph_batch_1.num_part
            num_part_2 = graph_batch_2.num_part

            node_rep_1, motif_rep_1, super_node_rep_1 = group_node_rep(node_rep_1, batch_size, num_part_1)
            node_rep_2, motif_rep_2,super_node_rep_2 = group_node_rep(node_rep_2, batch_size, num_part_2)

            y_pred = model_predictor.forward_func(super_node_rep_1, super_node_rep_2,node_rep_1,node_rep_2,motif_rep_1,motif_rep_2,ddi_type)

            loss = criterion(y_pred, y_true)
            train_loss += loss.item()
            
            y_pred_all.append(y_pred.detach().sigmoid().cpu())
            y_true_all.append(y_true.detach().long().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.dataset == "drugbank" or args.dataset == "twosides":
                cur_num += graph_batch_1.num_graphs // 2
            else:
                cur_num += graph_batch_1.num_graphs
            
            sys.stdout.write(
                f"\r{cur_num} / {train_set_len}, "
                f"{(train_loss / (i + 1)):.6f}, "
                "          "
            )
            sys.stdout.flush()
        
        y_pred = torch.cat(y_pred_all)
        y_true = torch.cat(y_true_all)

        train_acc, train_auc, train_f1, train_p, train_r, train_ap = \
            calc_metrics(y_pred, y_true)
        print()
        print(
            f"Train ACC: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}\n"
            f"Train P:   {train_p:.4f}, Train R:   {train_r:.4f}, Train AP: {train_ap:.4f}"
        )

        valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap = \
            evaluate([model, model_predictor], valid_loader, valid_set_len)

        print()
        print(
            f"Valid ACC: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid F1: {valid_f1:.4f}\n"
            f"Valid P:   {valid_p:.4f}, Valid R:   {valid_r:.4f}, Valid AP: {valid_ap:.4f}"
        )

        test_acc, test_auc, test_f1, test_p, test_r, test_ap = \
            evaluate([model,model_predictor], test_loader, test_set_len)
        print()
        print(
            f"Test ACC:  {test_acc:.4f}, Test AUC:  {test_auc:.4f}, Test F1:  {test_f1:.4f}\n"
            f"Test P:    {test_p:.4f}, Test R:    {test_r:.4f}, Test AP:  {test_ap:.4f}"
        )
        
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            max_valid_auc = valid_auc
            max_valid_ap = valid_ap
            max_valid_f1 = valid_f1
            valid_best_epoch = args.start_epoch + epoch
            torch.save(model.state_dict(), f"model/model_{args.start_epoch + epoch}.pt")
            print(f"BEST VALID IN EPOCH {args.start_epoch + epoch}")
        
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_test_ap = test_ap
            max_test_f1 = test_f1
            max_test_auc = test_auc
            test_best_epoch = args.start_epoch + epoch
            torch.save(model.state_dict(), f"model/model_{args.start_epoch + epoch}.pt")
            print(f"BEST TEST IN EPOCH {args.start_epoch + epoch}")
        
        scheduler.step()
        
        print()

    model.eval()
    model_predictor.eval()

    print(f"BEST VALID IN EPOCH {valid_best_epoch}")
    print(
        f"BEST VALID ACC:  {max_valid_acc:.4f}, VALID AUC:  {max_valid_auc:.4f}, VALID F1:  {max_valid_f1:.4f} VALID AP:  {max_valid_ap:.4f}"
    )
    print(f"BEST TEST IN EPOCH {test_best_epoch}")
    print(
        f"BEST TEST ACC:  {max_test_acc:.4f}, TEST AUC:  {max_test_auc:.4f}, TEST F1:  {max_test_f1:.4f} TEST AP:  {max_test_ap:.4f}"
    )


