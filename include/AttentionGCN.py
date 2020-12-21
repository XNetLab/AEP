import gc
import math
from torch import optim
from torch.nn import Parameter
from tqdm import tqdm
import torch.nn as nn
from include.Util import *
from include.Eval import *


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]  # 1*E
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads_num = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.weights_dropout = weights_dropout
        self.special_spmm = SpecialSpmm()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.head_dim)
        nn.init.uniform_(self.in_proj_weight, -std, std)
        nn.init.constant_(self.in_proj_bias, 0.)

    def forward(self, emb, edges):
        """
        emb shape: entity_size X dim
        edges shape: edge_size x dim
        key_padding_mask: edge_size
        """
        bsz, embed_dim = emb.shape
        edge_size = edges.shape[1]

        q = self.in_proj_q(emb[edges[0, :], :])  # edge_size,dim
        k = self.in_proj_k(emb[edges[1, :], :])  # edge_size,dim
        v = self.in_proj_v(emb)  # entity_size,dim
        q = q.contiguous().view(edge_size, self.heads_num, self.head_dim)
        k = k.contiguous().view(edge_size, self.heads_num, self.head_dim)
        v = v.contiguous().view(bsz, self.heads_num, self.head_dim)
        q *= self.scaling

        # attn_weights: edge_size,head_num
        attn_weights = torch.einsum('ehn,ehn->eh', [q, k])
        attn_weights = torch.exp(attn_weights)  # edge_e: edge_size,heads

        # attn_weights_sum: entity_size,heads,1
        attn_weights_sum = []
        for head in range(self.heads_num):
            attn_weights_sum.append(
                self.special_spmm(edges, attn_weights[:, head], torch.Size([bsz, bsz]), torch.ones(size=(bsz, 1)).cuda()
                if next(self.parameters()).is_cuda else torch.ones(size=(bsz, 1))))
        attn_weights_sum = torch.cat(attn_weights_sum, dim=1).view(bsz, self.heads_num, 1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn: entity_size x heads x head_dim
        attn = []
        for head in range(self.heads_num):
            attn.append(self.special_spmm(edges, attn_weights[:, head], torch.Size([bsz, bsz]), v[:, head, :]))

        attn = torch.cat(attn, dim=1).view(bsz, self.heads_num, -1)
        attn = attn.div(attn_weights_sum + 1e-20)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        attn = attn.contiguous().view(bsz, embed_dim)
        # attn = F.relu(attn)

        return attn

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class AttentionGCNLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout, weights_dropout=True):
        super(AttentionGCNLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.trans = nn.Linear(2 * embed_dim, embed_dim)

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.embed_dim)
        nn.init.uniform_(self.trans.weight, -std, std)
        nn.init.constant_(self.trans.bias, 0.)

    def forward(self, emb, edges):
        """
        emb: entity_size x embed_dim
        edges: edge_size x embed_dim
        """

        residual = emb
        aggregations = self.self_attn(emb=emb, edges=edges)  # entity_size x embed_dim

        aggregations = F.dropout(aggregations, p=self.dropout, training=self.training)
        x = self.trans(torch.cat([residual, aggregations], -1))  # entity_size x embed_dim
        x = self.attn_layer_norm(x)

        return x


class AttentionGCN(nn.Module):
    def __init__(self, ent_num, dim, num_layer, n_heads, dropout, device, ent_vecs=None):
        super(AttentionGCN, self).__init__()
        self.ent_num = ent_num
        self.device = device

        self.ent_emb = nn.Embedding(ent_num + 1, dim, padding_idx=ent_num)
        if ent_vecs is not None:
            self.ent_emb.weight.data.copy_(torch.cat([ent_vecs, torch.zeros([1, dim])]))
            self.ent_emb.weight.requires_grad = False
        else:
            nn.init.normal_(self.ent_emb.weight[0:-1], std=1.0 / math.sqrt(ent_num))

        self.num_layer = num_layer
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for _ in range(self.num_layer):
            self.layers.append(AttentionGCNLayer(dim, n_heads[_], dropout, weights_dropout=True))

    def forward(self, edges):
        ent_emb = self.ent_emb(torch.arange(self.ent_num).to(self.device))

        for idx in range(self.num_layer):
            ent_emb = self.layers[idx](ent_emb, edges)

        return ent_emb


def loss_aligment(device, ent_vecs, ILL, neg_left, neg_right, gamma, k=5, dist=2):
    train_left = torch.LongTensor((np.ones((ILL.shape[0], k)) *
                                   (ILL[:, 0].reshape((ILL.shape[0], 1)))
                                   ).reshape((ILL.shape[0] * k,))
                                  ).to(device)
    train_right = torch.LongTensor((np.ones((ILL.shape[0], k)) *
                                    (ILL[:, 1].reshape((ILL.shape[0], 1)))
                                    ).reshape((ILL.shape[0] * k,))
                                   ).to(device)

    # Cross-graph model alignment loss
    loss = F.triplet_margin_loss(
        torch.cat((ent_vecs[train_left], ent_vecs[train_right]), dim=0),
        torch.cat((ent_vecs[train_right], ent_vecs[train_left]), dim=0),
        torch.cat((ent_vecs[neg_left], ent_vecs[neg_right]), dim=0),
        margin=gamma, p=dist)

    return loss


def training_for_gcn(model, epochs, device,
                     edges, train_ILL, test_ILL, ents1, ents2,
                     unlabel_weight=1, unlabel_k=1, bootstrap_threshold=-1, sub_graph_size=None, ent_map=None,
                     lr=1e-3, weight_decay=1e-7, gamma=3, k=5, dist=1,
                     bootstrap_interval=10, eval_interval=5, neg_interval=10,
                     early_stopping_threshold=200, save_path=None):
    params = [{"params": filter(lambda p: p.requires_grad,
                                list(model.parameters()))}]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    print('start to train GCN')
    max_hits1 = 0
    eval_res = None
    best_vecs = None
    early_stopping_cnt = 0
    l_cands = list(ents1 - set([i for i, j in train_ILL]))
    r_cands = list(ents2 - set([j for i, j in train_ILL]))
    bootstrap_interval = math.ceil(bootstrap_interval / neg_interval) * neg_interval
    if ent_map is not None and bootstrap_interval != 0:
        train_l = set()
        train_r = set()
        for i, j in train_ILL:
            train_l.add(i)
            train_r.add(j)
        unseed_scores = get_edit_score(model.ent_num, ents1 - train_l, ents2 - train_r, ent_map,
                                       './data/' + Config.language + '/unseed_scores.npy')
        print('unseed_scores load over')
        test_scores = get_edit_score(model.ent_num, test_ILL[:, 0], test_ILL[:, 1], ent_map,
                                     './data/' + Config.language + '/test_scores.npy')
        print('test_scores load over')
        train_scores = get_edit_score(model.ent_num, train_ILL[:, 0], train_ILL[:, 1], ent_map,
                                      './data/' + Config.language + '/train_scores.npy')
        print('train_scores load over')
        unseed_scores = torch.tensor(unseed_scores).to(device)
        test_scores = torch.tensor(test_scores).to(device)
        train_scores = torch.tensor(train_scores).to(device)
    else:
        unseed_scores = None
    bsz = model.ent_num
    new_train_ILL = []
    true_alignment = set()
    true_l = set()
    true_r = set()
    for i, j in test_ILL:
        true_alignment.add((i, j))
        true_l.add(i)
        true_r.add(j)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        # forward
        ent_vecs = model(edges)

        # bootstrapping
        if bootstrap_interval != 0 and (epoch + 1) % bootstrap_interval == 0:
            with torch.no_grad():
                if sub_graph_size is not None:
                    aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(device),
                                                                torch.Size([bsz, bsz]))
                    aggregations = torch.matmul(aggregations_edge, ent_vecs)
                    neighbor = (ent_vecs + aggregations) / (sub_graph_size + 1).unsqueeze(-1)
                    neighbor = F.normalize(neighbor, p=2, dim=-1)
                else:
                    neighbor = None

                new_train_ILL, _ = get_explicit_alignments(F.normalize(ent_vecs, p=2, dim=-1), l_cands, r_cands,
                                                           bootstrap_threshold, neighbor, unseed_scores)

        # negative sampling
        if epoch == 0 or (epoch + 1) % neg_interval == 0:
            with torch.no_grad():
                neg_left, neg_right = nearest_neighbor_sampling(ent_vecs, torch.LongTensor(train_ILL[:, 0]).to(device),
                                                                torch.LongTensor(train_ILL[:, 1]).to(device), k)
            if bootstrap_interval != 0 and (epoch + 1) >= bootstrap_interval:
                with torch.no_grad():
                    neg_left2, neg_right2 = nearest_neighbor_sampling(ent_vecs,
                                                                      torch.LongTensor(new_train_ILL[:, 0]).to(device),
                                                                      torch.LongTensor(new_train_ILL[:, 1]).to(device),
                                                                      unlabel_k)

        train_loss = loss_aligment(device, ent_vecs, train_ILL, neg_left, neg_right, gamma, k, dist)
        if bootstrap_interval != 0 and (epoch + 1) >= bootstrap_interval:
            unlabel_loss = loss_aligment(device, ent_vecs, new_train_ILL, neg_left2, neg_right2, gamma, unlabel_k, dist)
            if unlabel_weight == 'auto':
                train_loss = (len(train_ILL) * train_loss + len(new_train_ILL) * unlabel_loss) / (
                            len(train_ILL) + len(new_train_ILL))
            else:
                train_loss += unlabel_weight * unlabel_loss
        train_loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # record and save the best model
        with torch.no_grad():
            model.eval()
            ent_vecs = model(edges)
            sims = get_sims(ent_vecs, test_ILL)
            if sub_graph_size is not None and bootstrap_interval != 0:
                aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(device),
                                                            torch.Size([bsz, bsz]))
                aggregations = torch.matmul(aggregations_edge, ent_vecs)
                neighbor = (ent_vecs + aggregations) / (sub_graph_size + 1).unsqueeze(-1)
                neighbor = F.normalize(neighbor, p=2, dim=-1)
                sims2 = get_sims(neighbor, test_ILL)
                sims += sims2
            if ent_map is not None and bootstrap_interval != 0:
                sims += test_scores

            # hits1, hits10, mrr = get_hits_vec(ent_vecs, test_ILL, device)
            hits1, hits10, mrr = get_hits_ma(sims, test_ILL, device)
            if hits1 > max_hits1:
                eval_res = (epoch, hits1, hits10, mrr)
                max_hits1 = hits1
                best_vecs = ent_vecs
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
                early_stopping_cnt = 0
            del sims
            if sub_graph_size is not None:
                del sims2

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # eval
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            with torch.no_grad():
                model.eval()
                print()
                print('---------------------------------------------')
                print('training loss:', train_loss)
                if bootstrap_interval != 0 and (epoch + 1) >= bootstrap_interval:
                    print('label training loss:', train_loss - unlabel_loss)
                    print('unlabel training loss:', unlabel_loss)
                    print('new_train_ILL length:', len(new_train_ILL))
                    cnt = 0
                    true_cnt = 0
                    for i, j in new_train_ILL:
                        if i in true_l or j in true_r:
                            cnt += 1
                        if (i, j) in true_alignment:
                            true_cnt += 1
                    print("test_ILL in unlabel alignment:", cnt)
                    print("unlabel alignment precision in test_ILL:", true_cnt / cnt)
                    print("unlabel alignment recall in test_ILL:", true_cnt / len(true_alignment))
                    print('---------------------------------------------')
                print("old training entity alignment:")
                sims = get_sims(ent_vecs, train_ILL)
                if sub_graph_size is not None and bootstrap_interval != 0 and (epoch + 1) >= bootstrap_interval:
                    sims2 = get_sims(neighbor, train_ILL)
                    sims += sims2
                if ent_map is not None and bootstrap_interval != 0 and (epoch + 1) >= bootstrap_interval:
                    sims += train_scores
                train_hits1, train_hits10, train_mrr = get_hits_ma(sims, train_ILL, device)
                del sims
                if sub_graph_size is not None and bootstrap_interval != 0 and (epoch + 1) >= bootstrap_interval:
                    del sims2
                # train_hits1, train_hits10, train_mrr = get_hits_vec(ent_vecs, train_ILL, device)
                msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (train_hits1, train_hits10, train_mrr)
                print(msg)

                print('*********************************************')
                with torch.no_grad():
                    eval_neg_left, eval_neg_right = nearest_neighbor_sampling(ent_vecs,
                                                                              torch.LongTensor(test_ILL[:, 0]).to(
                                                                                  model.device),
                                                                              torch.LongTensor(test_ILL[:, 1]).to(
                                                                                  model.device), k)
                    eval_neg_left, eval_neg_right = eval_neg_left.to(model.device), eval_neg_right.to(model.device)
                eval_loss = loss_aligment(device, ent_vecs, test_ILL, eval_neg_left, eval_neg_right, gamma, k, dist)
                print('testing loss', eval_loss)
                print("entity alignment:")
                msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (hits1, hits10, mrr)
                print(msg)
                print('best performance：')
                print('epoch:', eval_res[0], ' Hits@1:', eval_res[1], ' Hits@10:', eval_res[2], ' MRR:', eval_res[3])
                print('---------------------------------------------')
                print()

        if early_stopping_cnt == early_stopping_threshold:
            print('trigger early_stopping!!!!!!!!')
            break
        early_stopping_cnt += 1

        if epoch != epochs - 1:
            del ent_vecs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("optimization finished!")
    print("best performance：")
    print('epoch:', eval_res[0], ' Hits@1:', eval_res[1], ' Hits@10:', eval_res[2], ' MRR:', eval_res[3])
    return best_vecs
