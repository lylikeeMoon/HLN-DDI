# -*- coding: utf-8 -*-

import argparse

import random
import numpy as np

import torch

from ddi_train import train as train_tranductive
from ddi_train_inductive import train as train_inductive


num_node_feats_dict = {"drugbank" : 2, "twosides" : 2}
num_ddi_types_dict = {"drugbank" : 86, "twosides" : 963,}

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument(
        "--dataset", type=str, choices=[
            "DrugBank", "Twosides"
        ], default="DrugBank"
    )
    
    parser.add_argument("--inductive", action="store_true", default=False)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2], default=0)


    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_epoch', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain.pth',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=64, help='hidden size')
    parser.add_argument("--start_epoch", type=int, default=0)

    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.num_node_feats = num_node_feats_dict[args.dataset]
    args.num_ddi_types = num_ddi_types_dict[args.dataset]

    
    return args

if __name__ == "__main__":
    args = main()
    
    #model.load_state_dict(torch.load("model/model_232.pt", map_location=MY_DEVICE))
    
    set_all_seeds(args.seed)

    if args.dataset != "drugbank" and args.dataset != "twosides":
        args.inductive = False
    
    #args.fold = 0
    #model = InteractionPredictor(args).to(args.device)
    
    if not args.inductive:
        train_tranductive(args)
    else:
        train_inductive(args)
    

    '''if args.dataset != "drugbank":
        set_all_seeds(1)
    
    args.fold = 1
    model = InteractionPredictor(args).to(args.device)
    
    if not args.inductive:
        train_tranductive(model, args)
    else:
        train_inductive(model, args)

    if args.dataset != "drugbank":
        set_all_seeds(2)
    
    args.fold = 2
    model = InteractionPredictor(args).to(args.device)
    
    if not args.inductive:
        train_tranductive(model, args)
    else:
        train_inductive(model, args)
    '''
