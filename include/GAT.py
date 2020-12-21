import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from include.AttentionGCN import SpecialSpmm


class GATLayer(nn.Module):
    def __init__(self, embed_dim, device):
        super(GATLayer, self).__init__()
        self.w = nn.Linear(embed_dim, embed_dim)
        self.a = nn.Linear(2 * embed_dim, 1)
        self.embed_dim = embed_dim
        self.device = device
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.special_spmm = SpecialSpmm()
        self.trans = nn.Linear(2 * embed_dim, embed_dim)
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
        # attn_weights: edge_size
        attn_weights = self.a(torch.cat([self.w(emb[edges[0]]), self.w(emb[edges[1]])], -1)).squeeze()
        attn_weights = torch.exp(self.leaky_relu(attn_weights))  # edge_e: edge_size
        attn_weights_sum = self.special_spmm(edges, attn_weights, torch.Size([bsz, bsz]), torch.ones(size=(bsz, 1)).to(self.device))
        attn_weights_sum = attn_weights_sum.view(bsz, 1)

        # attn: entity_size x head_dim
        aggregations = self.special_spmm(edges, attn_weights, torch.Size([bsz, bsz]), emb)
        aggregations = aggregations.div(attn_weights_sum + 1e-20)

        h = self.trans(torch.cat([emb, aggregations], -1))
        h = self.layer_norm(h)
        return h


class GAT(nn.Module):
    def __init__(self, ent_num, dim, num_layer, device, ent_vecs=None):
        super(GAT, self).__init__()
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
            self.layers.append(GATLayer(dim, device))

    def forward(self, edges):
        ent_emb = self.ent_emb(torch.arange(self.ent_num).to(self.device))

        for idx in range(self.num_layer):
            ent_emb = self.layers[idx](ent_emb, edges)
            if idx != self.num_layer - 1:
                ent_emb = F.relu(ent_emb)
        return ent_emb
