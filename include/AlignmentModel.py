import gc
from include.Eval import *
from include.Load import get_edit_score
from include.Util import *


class PropagationGCN:
    def __init__(self, num_layer, device,
                 s_hard=0.7, alpha=0.7, epsilon=1e-10):

        self.s_hard = s_hard
        self.alpha = alpha
        self.epsilon = epsilon

        self.device = device
        self.num_layer = num_layer

    def preprocess(self, hard_match, sims, ent_emb):
        weights = torch.tensor([0. if sim < 0 else sim for sim in sims]).to(self.device)

        for l, r in hard_match:
            l_t = ent_emb[l]
            r_t = ent_emb[r]
            m = (l_t + r_t) / 2
            ent_emb[l] = m
            ent_emb[r] = m
        return weights

    def update(self, ent_emb, edges, weights):
        bsz = ent_emb.shape[0]
        dim = ent_emb.shape[1]
        edges = torch.tensor(edges).to(self.device)
        weights = weights[edges[1]].view([-1, 1])

        with torch.no_grad():
            for idx in range(self.num_layer):
                messages = ent_emb[edges[1]] * weights  # edge_size * dim

                # aggregate
                aggregations = torch.sparse.FloatTensor(edges, messages, torch.Size([bsz, bsz, dim]))
                aggregations = torch.sparse.sum(aggregations, 1).to_dense()

                # update
                aggregations_norm = torch.sqrt(torch.sum(aggregations * aggregations, -1))
                emb_norm = torch.sqrt(torch.sum(ent_emb * ent_emb, -1))
                aggregations = aggregations * emb_norm.view([-1, 1]) / (aggregations_norm + self.epsilon).view([-1, 1])
                ent_emb = self.alpha * ent_emb + (1 - self.alpha) * aggregations

        return ent_emb


def propagation(ent_vecs, s_hard, alpha, num_layer, device,
                edges, ents1, ents2,
                train_ILL, test_ILL,
                epochs=3, sub_graph_size=None, ent_map=None):
    print('start to propagation')
    model = PropagationGCN(num_layer, device, s_hard, alpha)
    max_hits1 = 0
    eval_res = None
    epoch = 0
    bsz = ent_vecs.shape[0]
    ent_vecs = ent_vecs.cpu().detach().to(device)
    # all training set is asl
    asl = []
    for i, j in train_ILL:
        asl.append((i, j))

    print('----------------------------init----------------------------')
    sims = get_sims(ent_vecs, test_ILL)
    if sub_graph_size is not None:
        aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(device),
                                                    torch.Size([bsz, bsz]))
        aggregations = torch.matmul(aggregations_edge, ent_vecs)
        neighbor = (ent_vecs + aggregations) / (sub_graph_size + 1).unsqueeze(-1)
        neighbor = F.normalize(neighbor, p=2, dim=-1)
        sims2 = get_sims(neighbor, test_ILL)
        sims += sims2
    if ent_map is not None:
        test_scores = get_edit_score(bsz, test_ILL[:, 0], test_ILL[:, 1], ent_map,
                                     './data/' + Config.language + '/test_scores.npy')
        test_scores = torch.tensor(test_scores).to(device)
        sims += test_scores
    else:
        test_scores = None
    print('testing set：')
    test_hits1, test_hits10, test_mrr = get_hits_ma(sims, test_ILL, device)
    # test_hits1, test_hits10, test_mrr = get_hits_vec(ent_vecs, test_ILL, device)
    msg = 'Hits@1:%.4f, Hits@10:%.4f, MRR:%.4f\n' % (test_hits1, test_hits10, test_mrr)
    print(msg)
    print()
    del sims
    if sub_graph_size is not None:
        del sims2, neighbor

    if test_hits1 > max_hits1:
        eval_res = (epoch, test_hits1, test_hits10, test_mrr)
        max_hits1 = test_hits1
    true_alignment = set()
    true_l = set()
    true_r = set()
    for i, j in test_ILL:
        true_alignment.add((i, j))
        true_l.add(i)
        true_r.add(j)

    for l, r in asl:
        l_t = ent_vecs[l]
        r_t = ent_vecs[r]
        m = (l_t + r_t) / 2
        ent_vecs[l] = m
        ent_vecs[r] = m

    with torch.no_grad():
        for epoch in range(epochs):
            print('----------------------------The ' + str(epoch + 1) + ' epoch begin----------------------------')
            norm_emb = F.normalize(ent_vecs, 2, -1)

            l_cands = list(ents1 - set([i for i, j in asl]))
            r_cands = list(ents2 - set([j for i, j in asl]))

            if sub_graph_size is not None:
                aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(device),
                                                            torch.Size([bsz, bsz]))
                aggregations = torch.matmul(aggregations_edge, ent_vecs)
                neighbor = (ent_vecs + aggregations) / (sub_graph_size + 1).unsqueeze(-1)
                neighbor = F.normalize(neighbor, p=2, dim=-1)
            else:
                neighbor = None

            if ent_map:
                if epoch == 0:
                    unseed_scores = get_edit_score(bsz, l_cands, r_cands, ent_map,
                                                   './data/' + Config.language + '/unseed_scores.npy')
                else:
                    unseed_scores = get_edit_score(bsz, l_cands, r_cands, ent_map)
                unseed_scores = torch.tensor(unseed_scores).to(device)
            else:
                unseed_scores = None

            hard_match, sims = get_explicit_alignments(norm_emb, l_cands, r_cands, s_hard, neighbor, unseed_scores, asl)
            asl.extend(hard_match.tolist())
            print('hard_match increase', len(hard_match))
            cnt = 0
            true_cnt = 0
            for i, j in hard_match:
                if i in true_l or j in true_r:
                    cnt += 1
                if (i, j) in true_alignment:
                    true_cnt += 1
            if cnt != 0:
                print("hard_match precision in test_ILL:", true_cnt / cnt)
                print("hard_match recall in test_ILL:", true_cnt / len(true_alignment))

            print('\nw/o propagation and using one-to-one match:')
            test_hits1 = eval_alignment(asl, test_ILL, ent_vecs, device, neighbor, test_scores, True)
            print('Hits@1:', test_hits1)

            sims = torch.tensor(sims).to(device)
            weights = model.preprocess(asl, sims, norm_emb)
            ent_vecs = model.update(norm_emb, edges, weights)

            if sub_graph_size is not None:
                aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(device),
                                                            torch.Size([bsz, bsz]))
                aggregations = torch.matmul(aggregations_edge, ent_vecs)
                neighbor = (ent_vecs + aggregations) / (sub_graph_size + 1).unsqueeze(-1)
                neighbor = F.normalize(neighbor, p=2, dim=-1)
            else:
                neighbor = None

            print('\ntesting set：')
            test_hits1 = eval_alignment(asl, test_ILL, ent_vecs, device, neighbor, test_scores, True)
            print('Hits@1:', test_hits1)
            if test_hits1 > max_hits1:
                eval_res = (epoch + 1, test_hits1)
                max_hits1 = test_hits1
            print('\nbest performance：')
            print('epoch:', eval_res[0], ' Hits@1:', eval_res[1])
            print('----------------------------The ' + str(epoch + 1) + ' epoch over----------------------------')
            print()
            del unseed_scores
            del neighbor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print('best performance：')
    print('epoch:', eval_res[0], ' Hits@1:', eval_res[1])
    print('----------------------------')
    print()
    return ent_vecs, asl
