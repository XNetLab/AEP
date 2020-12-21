import torch
import torch.nn.functional as F

from include.Config import Config
from include.Load import get_edit_score
from include.Util import get_explicit_alignments


def get_hits_vec(ent_vecs, test_pair, device, top_k=(1, 10)):
    """
    Input:
        ent_vecs: the entity embeddings
        test_pair: the testing set
    Output: hits1,hits10,mrr
    """
    with torch.no_grad():
        test_left = torch.LongTensor(test_pair[:, 0].squeeze())
        test_right = torch.LongTensor(test_pair[:, 1].squeeze())
        test_left_vec = ent_vecs[test_left]
        test_right_vec = ent_vecs[test_right]

        he = F.normalize(test_left_vec, p=2, dim=-1)
        norm_e_em = F.normalize(test_right_vec, p=2, dim=-1)
        sim = torch.matmul(he, norm_e_em.t())
        cos_distance = 1 - sim
        top_lr = [0] * len(top_k)
        mrr_sum_l = 0
        ranks = torch.argsort(cos_distance)
        rank_indexs = torch.where(ranks == torch.arange(len(test_pair)).unsqueeze(1).to(device))[
            1].cpu().numpy().tolist()
        for i in range(cos_distance.shape[0]):
            rank_index = rank_indexs[i]
            mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1

        # msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (
        #     top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
        # print(msg)
        return top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair)


def get_hits_ma(sim, test_pair, device, top_k=(1, 10)):
    """
    Input:
        sim: the similarity of entity embeddings
        test_pair: the testing set
    Output: hits1,hits10,mrr
    """
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    cos_distance = 1 - sim
    ranks = torch.argsort(cos_distance)
    rank_indexs = torch.where(ranks == torch.arange(len(test_pair)).unsqueeze(1).to(device))[
        1].cpu().numpy().tolist()
    for i in range(cos_distance.shape[0]):
        rank_index = rank_indexs[i]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    # msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair),
    #                                                  mrr_sum_l / len(test_pair))
    # print(msg)

    return top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair)


def eval_alignment(hard_match, test_pair, ent_vecs, device, neighbor=None, edit_scores=None, ont_to_one_match=True):
    """
    Input:
        hard_match: the set of hard_match, including the aligned entity pairs from the model prediction at first
        test_pair: the testing set
        ent_vecs: the entity embeddings
        neighbor: the neighbors id of entities
        edit_scores: the edit distance matrix of test_pair
        ont_to_one_match: using one-to-one alignment or one-to-many alignment
    Output: hits1,hits10,mrr
    """
    total = len(test_pair)
    hits1_cnt = 0
    true_alignments = set()
    pred_alignments = set()
    pred_l = set()
    pred_r = set()
    for i, j in hard_match:
        pred_alignments.add((i, j))
        pred_l.add(i)
        pred_r.add(j)

    # all entities which are not in hard_match set
    unknow_l = list()
    unknow_r = list()
    # the pos of entities of the unknow set in the matrix of edit_scores
    row_id = []
    col_id = []

    for n in range(len(test_pair)):
        i, j = test_pair[n]
        true_alignments.add((i, j))
        if (i, j) in pred_alignments:
            hits1_cnt += 1
        if i not in pred_l:
            unknow_l.append(i)
            row_id.append(n)
        if j not in pred_r:
            unknow_r.append(j)
            col_id.append(n)

    if len(unknow_l) == 0 and len(unknow_r) == 0:
        return hits1_cnt / total

    if ont_to_one_match:
        # using assigning algorithm for alignment
        norm_emb = F.normalize(ent_vecs, p=2, dim=-1)
        if edit_scores is not None:
            unknow_scores = edit_scores[row_id][:, col_id]
        else:
            unknow_scores = None
        unknow_set, _ = get_explicit_alignments(norm_emb, unknow_l, unknow_r, -1, neighbor, unknow_scores,
                                                allMatch=True)
        for i, j in unknow_set:
            if (i, j) in true_alignments:
                hits1_cnt += 1
    else:
        # each entity takes the entity with the largest similarity for alignment
        l_cands = torch.LongTensor(unknow_l).to(device)
        r_cands = torch.LongTensor(unknow_r).to(device)
        l_cands_vec = ent_vecs[l_cands]
        r_cands_vec = ent_vecs[r_cands]
        he = F.normalize(l_cands_vec, p=2, dim=-1)
        norm_e_em = F.normalize(r_cands_vec, p=2, dim=-1)
        sim = torch.matmul(he, norm_e_em.t())

        if neighbor is not None:
            sim += torch.matmul(neighbor[l_cands], neighbor[r_cands].t())
        if edit_scores is not None:
            sim += edit_scores
        cos_distance = 1 - sim
        change = False
        if len(unknow_l) > len(unknow_r):
            cos_distance = cos_distance.t()
            change = True
        ranks = torch.argmin(cos_distance, -1).cpu().detach().numpy()
        for i in range(ranks.shape[0]):
            if change:
                r = unknow_r[i]
                l = unknow_l[ranks[i]]
            else:
                l = unknow_l[i]
                r = unknow_r[ranks[i]]
            if (l, r) in true_alignments:
                hits1_cnt += 1

    hits1 = hits1_cnt / total
    return hits1


def stable_match(ent_vecs, test_ILL, edges, sub_graph_size, ent_map, device):
    norm_emb = F.normalize(ent_vecs, 2, -1)
    true_alignments = set()
    for i, j in test_ILL:
        true_alignments.add((i, j))

    bsz = ent_vecs.shape[0]
    if sub_graph_size is not None:
        aggregations_edge = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]).to(device),
                                                    torch.Size([bsz, bsz]))
        aggregations = torch.matmul(aggregations_edge, ent_vecs)
        neighbor_s = (ent_vecs + aggregations) / (sub_graph_size + 1).unsqueeze(-1)
        neighbor_s = F.normalize(neighbor_s, p=2, dim=-1)
    else:
        neighbor_s = None
    if ent_map is not None:
        unseed_scores_s = get_edit_score(bsz, set(test_ILL[:, 0]), set(test_ILL[:, 1]), ent_map,
                                         './data/' + Config.language + '/test_scores.npy')
        unseed_scores_s = torch.tensor(unseed_scores_s).to(device)
    else:
        unseed_scores_s = None
    hits1_cnt = 0
    unknow_set, _ = get_explicit_alignments(norm_emb, test_ILL[:, 0], test_ILL[:, 1], -1, neighbor_s, unseed_scores_s,
                                            allMatch=True)
    for i, j in unknow_set:
        if (i, j) in true_alignments:
            hits1_cnt += 1
    print(hits1_cnt / len(test_ILL))
