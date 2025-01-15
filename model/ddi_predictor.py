# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from model.ddi_layers import MLP, CoAttentionLayer,RESCAL

def create_var(tensor, device, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor).to(device)
    else:
        return Variable(tensor, requires_grad=requires_grad).to(device)



class InteractionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_ddi_types = args.num_ddi_types
        self.num_ddi_types = num_ddi_types
        self.device = args.device
        self.hidden_dim = args.hidden_size
        self.batch_size=args.batch_size*2
        self.co_attention = CoAttentionLayer(self.hidden_dim)
        self.KGE = RESCAL(num_ddi_types, self.hidden_dim)


        if args.inductive:
            self.forward_func = self.forward_inductive
        else:
            self.forward_func = self.forward_transductive

    def forward_inductive(self,super_node_rep_1, super_node_rep_2, node_rep_1,node_rep_2,motif_rep_1,motif_rep_2,super_node_rep_old_1, super_node_rep_old_2, node_rep_old_1,node_rep_old_2,motif_rep_old_1,motif_rep_old_2,ddi_type):

        node_rep_1 = torch.stack([torch.sum(node_rep, dim=0) for node_rep in node_rep_1],dim=0).to(self.device)
        node_rep_2 = torch.stack([torch.sum(node_rep, dim=0) for node_rep in node_rep_2],dim=0).to(self.device)

        motif_rep_1 = torch.stack([torch.sum(motif_rep, dim=0) for motif_rep in motif_rep_1], dim=0).to(self.device)
        motif_rep_2 = torch.stack([torch.sum(motif_rep, dim=0) for motif_rep in motif_rep_2], dim=0).to(self.device)

        super_node_rep_1 = torch.stack(super_node_rep_1, dim=0).to(self.device)
        super_node_rep_2 = torch.stack(super_node_rep_2, dim=0).to(self.device)

        ddi_type = torch.LongTensor(list(ddi_type)).to(self.device)

        V_i = torch.stack([node_rep_1, motif_rep_1, super_node_rep_1], dim=1)
        V_j = torch.stack([node_rep_2, motif_rep_2, super_node_rep_2], dim=1)

        attentions = self.co_attention(V_i, V_j)
        scores = self.KGE(V_i, V_j, ddi_type, attentions)

        if super_node_rep_old_1 is None:
            return scores
        node_rep_old_1 = torch.stack([torch.sum(node_rep_old, dim=0) for node_rep_old in node_rep_old_1], dim=0).to(self.device)
        node_rep_old_2 = torch.stack([torch.sum(node_rep_old, dim=0) for node_rep_old in node_rep_old_2], dim=0).to(self.device)

        motif_rep_old_1 = torch.stack([torch.sum(motif_rep_old, dim=0) for motif_rep_old in motif_rep_old_1], dim=0).to(self.device)
        motif_rep_old_2 = torch.stack([torch.sum(motif_rep_old, dim=0) for motif_rep_old in motif_rep_old_2], dim=0).to(self.device)

        super_node_rep_old_1 = torch.stack(super_node_rep_old_1, dim=0).to(self.device)
        super_node_rep_old_2 = torch.stack(super_node_rep_old_2, dim=0).to(self.device)

        V_i_old = torch.stack([node_rep_old_1, motif_rep_old_1, super_node_rep_old_1], dim=1)
        V_j_old = torch.stack([node_rep_old_2, motif_rep_old_2, super_node_rep_old_2], dim=1)

        attentions_old = self.co_attention(V_i_old, V_j_old)
        # print(attentions.shape)
        scores_old = self.KGE(V_i_old, V_j_old, ddi_type, attentions_old)

        scores = torch.stack([scores, scores_old]).sigmoid().mean(dim=0)
        return scores



    def forward_transductive(self,super_node_rep_1, super_node_rep_2, node_rep_1,node_rep_2,motif_rep_1,motif_rep_2,ddi_type):

        node_rep_1 = torch.stack([torch.sum(node_rep, dim=0) for node_rep in node_rep_1],dim=0).to(self.device)
        node_rep_2 = torch.stack([torch.sum(node_rep, dim=0) for node_rep in node_rep_2],dim=0).to(self.device)

        motif_rep_1 = torch.stack([torch.sum(motif_rep, dim=0) for motif_rep in motif_rep_1], dim=0).to(self.device)
        motif_rep_2 = torch.stack([torch.sum(motif_rep, dim=0) for motif_rep in motif_rep_2], dim=0).to(self.device)

        super_node_rep_1 = torch.stack(super_node_rep_1, dim=0).to(self.device)
        super_node_rep_2 = torch.stack(super_node_rep_2, dim=0).to(self.device)


        ddi_type = torch.LongTensor(list(ddi_type)).to(self.device)

        V_i=torch.stack([node_rep_1,motif_rep_1,super_node_rep_1],dim=1)
        V_j = torch.stack([node_rep_2, motif_rep_2, super_node_rep_2], dim=1)

        attentions = self.co_attention(V_i, V_j)
        scores = self.KGE(V_i, V_j, ddi_type, attentions)


        return scores

