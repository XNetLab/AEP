import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from include.AttentionGCN import SpecialSpmm


class GCNLayer(nn.Module):
    def __init__(self, embed_dim, device, aggregate_way='sum'):
        super(GCNLayer, self).__init__()
        self.w = nn.Linear(2 * embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.aggregate_way = aggregate_way
        self.device = device
        self.layer_norm = nn.LayerNorm(embed_dim)
        if aggregate_way != 'sum':
            self.special_spmm = SpecialSpmm()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.embed_dim)
        nn.init.uniform_(self.w.weight, -std, std)
        nn.init.constant_(self.w.bias, 0.)

    def forward(self, emb, edges):
        """
        emb: entity_size x embed_dim
        edges: edge_size x embed_dim
        """
        bsz = emb.shape[0]

        if self.aggregate_way == 'sum':
            aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(self.device),
                                                        torch.Size([bsz, bsz]))
            aggregations = torch.matmul(aggregations_edge, emb)
        else:
            # attn_weights: edge_size
            attn_weights = torch.sum(emb[edges[1]] * emb[edges[0]], -1)
            attn_weights = torch.exp(attn_weights)  # edge_e: edge_size
            # attn_weights_sum: entity_size,1
            attn_weights_sum = self.special_spmm(edges, attn_weights, torch.Size([bsz, bsz]),
                                                 torch.ones(size=(bsz, 1)).to(self.device))
            attn_weights_sum = attn_weights_sum.view(bsz, 1)

            # attn: entity_size x head_dim
            aggregations = self.special_spmm(edges, attn_weights, torch.Size([bsz, bsz]), emb)
            aggregations = aggregations.div(attn_weights_sum + 1e-20)

        x = self.w(torch.cat([emb, aggregations], -1))  # entity_size x embed_dim
        x = self.layer_norm(x)
        return x


class GCN(nn.Module):
    def __init__(self, ent_num, dim, num_layer, device, ent_vecs=None, aggregate_way='sum'):
        super(GCN, self).__init__()
        self.ent_num = ent_num
        self.device = device
        self.ent_emb = nn.Embedding(ent_num + 1, dim, padding_idx=ent_num)
        if ent_vecs is not None:
            self.ent_emb.weight.data.copy_(torch.cat([ent_vecs, torch.zeros([1, dim])]))
            self.ent_emb.weight.requires_grad = False
        else:
            nn.init.normal_(self.ent_emb.weight[0:-1], std=1.0 / math.sqrt(ent_num))

        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            self.layers.append(GCNLayer(dim, device, aggregate_way))

    def forward(self, edges):
        ent_emb = self.ent_emb(torch.arange(self.ent_num).to(self.device))

        for idx in range(self.num_layer):
            ent_emb = self.layers[idx](ent_emb, edges)
            if idx != self.num_layer - 1:
                ent_emb = F.relu(ent_emb)
        return ent_emb
