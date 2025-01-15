# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics

from datasets.ddi_dataset_inductive import DDIDataset, BatchLoader
from torch.utils.data import DataLoader
from model.gnn_model import GNN
from model.ddi_predictor import InteractionPredictor
import torch.optim as optim

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

def group_node_rep(node_rep, batch_size, num_part):
    group = []
    motif_group = []
    super_group = []
    # print('num_part', num_part)
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


@torch.no_grad()
def evaluate(models, loader, set_len):
    cur_num = 0
    y_pred_all, y_true_all = [], []
    model = models[0]
    model_predictor = models[1]
    for batch in loader:
        graph_batch_1, graph_batch_2, \
        graph_batch_old_1, graph_batch_old_2, \
        ddi_type, y_true = batch

        batch_size = len(ddi_type)

        node_rep_1 = model(graph_batch_1.x, graph_batch_1.edge_index, graph_batch_1.edge_attr)
        node_rep_2 = model(graph_batch_2.x, graph_batch_2.edge_index, graph_batch_2.edge_attr)

        num_part_1 = graph_batch_1.num_part
        num_part_2 = graph_batch_2.num_part

        node_rep_1, motif_rep_1, super_node_rep_1 = group_node_rep(node_rep_1, batch_size, num_part_1)
        node_rep_2, motif_rep_2, super_node_rep_2 = group_node_rep(node_rep_2, batch_size, num_part_2)

        node_rep_old_1 = model(graph_batch_old_1.x, graph_batch_old_1.edge_index, graph_batch_old_1.edge_attr)
        node_rep_old_2 = model(graph_batch_old_2.x, graph_batch_old_2.edge_index, graph_batch_old_2.edge_attr)

        num_part_old_1 = graph_batch_old_1.num_part
        num_part_old_2 = graph_batch_old_2.num_part

        node_rep_old_1, motif_rep_old_1, super_node_rep_old_1 = group_node_rep(node_rep_old_1, batch_size, num_part_old_1)
        node_rep_old_2, motif_rep_old_2, super_node_rep_old_2 = group_node_rep(node_rep_old_2, batch_size, num_part_old_2)


        y_pred = model_predictor.forward_func(
            super_node_rep_1, super_node_rep_2, node_rep_1,node_rep_2,motif_rep_1,motif_rep_2,super_node_rep_old_1, super_node_rep_old_2, node_rep_old_1,node_rep_old_2,motif_rep_old_1,motif_rep_old_2,ddi_type
        )
        
        y_pred_all.append(y_pred.detach().cpu())
        y_true_all.append(torch.LongTensor(y_true))
        
        cur_num += graph_batch_1.num_graphs // 2
        sys.stdout.write(f"\r{cur_num} / {set_len}")
        sys.stdout.flush()
    
    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)
    
    return calc_metrics(y_pred, y_true)

def train(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    train_set = DDIDataset(args.dataset, "train", args.fold)
    valid_set = DDIDataset(args.dataset, "valid", args.fold)
    either_set = DDIDataset(args.dataset, "either", args.fold)
    both_set = DDIDataset(args.dataset, "both", args.fold)
    
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    either_set_len = len(either_set)
    both_set_len = len(both_set)
    
    batch_loader = BatchLoader(args)
    
    train_loader = DataLoader(
        train_set, args.batch_size, True,
        collate_fn=batch_loader.collate_fn_train
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn_test
    )
    either_loader = DataLoader(
        either_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn_test
    )
    both_loader = DataLoader(
        both_set, args.batch_size, False,
        collate_fn=batch_loader.collate_fn_test
    )

    model = GNN(args.num_layer, args.hidden_size, args.num_node_feats, JK=args.JK, drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(
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
    either_best_epoch=0
    both_best_epoch = 0
    max_valid_acc, max_either_acc, max_either_auc, max_either_f1, max_either_ap, max_both_acc, max_both_auc, max_both_f1, max_both_ap, = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in range(args.num_epoch):
        print(f"Epoch: {args.start_epoch + epoch}")
        
        train_loss = 0.0
        cur_num = 0
        y_pred_all, y_true_all = [], []
        train_set.do_shuffle()

        model.train()
        model_predictor.train()

        for i, batch in enumerate(train_loader):
            graph_batch_1, graph_batch_2, _, _, ddi_type, y_true = batch
            y_true = torch.Tensor(y_true).to(args.device)
            batch_size = len(ddi_type)
            
            '''y_pred = model.forward_func(
                graph_batch_1, graph_batch_2,
                None, None, ddi_type
            )'''

            graph_batch_1 = graph_batch_1.to(device)
            graph_batch_2 = graph_batch_2.to(device)

            node_rep_1 = model(graph_batch_1.x, graph_batch_1.edge_index, graph_batch_1.edge_attr)
            node_rep_2 = model(graph_batch_2.x, graph_batch_2.edge_index, graph_batch_2.edge_attr)

            num_part_1 = graph_batch_1.num_part
            num_part_2 = graph_batch_2.num_part

            node_rep_1, motif_rep_1, super_node_rep_1 = group_node_rep(node_rep_1, batch_size, num_part_1)
            node_rep_2, motif_rep_2, super_node_rep_2 = group_node_rep(node_rep_2, batch_size, num_part_2)

            y_pred = model_predictor.forward_func(super_node_rep_1, super_node_rep_2, node_rep_1, node_rep_2, motif_rep_1,
                                             motif_rep_2, None, None, None, None, None, None,ddi_type)


            loss = criterion(y_pred, y_true)
            train_loss += loss.item()
            
            y_pred_all.append(y_pred.detach().sigmoid().cpu())
            y_true_all.append(y_true.detach().long().cpu())

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            cur_num += graph_batch_1.num_graphs // 2
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
        
        model.eval()
        model_predictor.eval()

        valid_acc, valid_auc, valid_f1, valid_p, valid_r, valid_ap = \
            evaluate([model,model_predictor], valid_loader, valid_set_len)
        print()
        print(
            f"Valid ACC:  {valid_acc:.4f}, Valid AUC:  {valid_auc:.4f}, Valid F1:  {valid_f1:.4f}\n"
            f"Valid P:    {valid_p:.4f}, Valid R:    {valid_r:.4f}, Valid AP:  {valid_ap:.4f}"
        )
        
        either_acc, either_auc, either_f1, either_p, either_r, either_ap = \
            evaluate([model,model_predictor], either_loader, either_set_len)
        print()
        print(
            f"Either ACC:  {either_acc:.4f}, Either AUC:  {either_auc:.4f}, Either F1:  {either_f1:.4f}\n"
            f"Either P:    {either_p:.4f}, Either R:    {either_r:.4f}, Either AP:  {either_ap:.4f}"
        )
        
        both_acc, both_auc, both_f1, both_p, both_r, both_ap = \
            evaluate([model,model_predictor], both_loader, both_set_len)
        print()
        print(
            f"Both ACC:    {both_acc:.4f}, Both AUC:    {both_auc:.4f}, Both F1:    {both_f1:.4f}\n"
            f"Both P:      {both_p:.4f}, Both R:      {both_r:.4f}, Both AP:    {both_ap:.4f}"
        )
        
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc
            print(f"BEST VALID IN EPOCH {args.start_epoch + epoch}")
        
        if either_acc > max_either_acc:
            max_either_acc = either_acc
            max_either_auc = either_auc
            max_either_ap = either_ap
            max_either_f1 = either_f1
            either_best_epoch = args.start_epoch + epoch
            print(f"BEST EITHER IN EPOCH {args.start_epoch + epoch}")
        
        if both_acc > max_both_acc:
            max_both_acc = both_acc
            max_both_ap = both_ap
            max_both_f1 = both_f1
            max_both_auc = both_auc
            both_best_epoch = args.start_epoch + epoch
            print(f"BEST BOTH IN EPOCH {args.start_epoch + epoch}")


        scheduler.step()
        
        print()

    print(f"BEST EITHER IN EPOCH {either_best_epoch}")
    print(
        f"BEST Either ACC:  {max_either_acc:.4f}, Either AUC:  {max_either_auc:.4f}, Either F1:  {max_either_f1:.4f} Either AP:  {max_either_ap:.4f}"
    )
    print(f"BEST BOTH IN EPOCH {both_best_epoch}")
    print(
        f"BEST BOTH ACC:  {max_both_acc:.4f}, Both AUC:  {max_both_auc:.4f}, Both F1:  {max_both_f1:.4f} Both AP:  {max_both_ap:.4f}"
    )