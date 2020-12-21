import random
import torch
import numpy as np
import torch.nn.functional as F


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def nearest_neighbor_sampling(emb, left, right, K):
    neg_left = []
    distance = pairwise_distances(emb[right], emb[right])
    for idx in range(right.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_left.append(right[indices[1: K + 1]])
    neg_left = torch.cat(tuple(neg_left), dim=0)
    neg_right = []
    distance = pairwise_distances(emb[left], emb[left])
    for idx in range(left.shape[0]):
        _, indices = torch.sort(distance[idx, :], descending=False)
        neg_right.append(left[indices[1: K + 1]])
    neg_right = torch.cat(tuple(neg_right), dim=0)
    return neg_left, neg_right


def get_kg_info(KG):
    """
    Output: h2t is a dict where key is entity's id and value is the tail entities' id of the key entity
            t2h is a dict where key is entity's id and value is the head entities' id of the key entity
            ent_pair2r is a dict where key is a entity pair and value is relations' id of the entity pair
    """
    h2t = dict()
    t2h = dict()
    ent_pair2r = dict()
    rs = set()
    for h, r, t in KG:
        rs.add(r)
        ts = h2t.get(h, set())
        ts.add(t)
        h2t[h] = ts

        hs = t2h.get(t, set())
        hs.add(h)
        t2h[t] = hs

        rels = ent_pair2r.get((h, t), set())
        rels.add(r)
        ent_pair2r[(h, t)] = rels
    return h2t, t2h, ent_pair2r


def extract_subgraph(h2t, t2h, ent_pair2r, ent_num, rel_num, keep_self=False):
    """
    Output: sub_graphs is a dict where key is entity's id and value is its info including neighbors and sub-graph of it
    """
    sub_graphs = dict()
    max_neighbor = 0
    max_rel = 0
    for ent in range(ent_num):
        hs = t2h.get(ent, set())
        ts = h2t.get(ent, set())
        if len(hs) == 0 and len(ts) == 0:
            continue
        if ent in hs or ent in ts:
            hs.discard(ent)
            ts.discard(ent)

        ents = set()
        ents.update(ts)
        ents.update(hs)

        neighbors = list(ents)
        random.shuffle(neighbors)
        if keep_self:
            neighbors = [ent] + neighbors
        max_neighbor = max(max_neighbor, len(neighbors))

        m = [[[rel_num] for _ in range(len(neighbors))] for _ in range(len(neighbors))]
        for h in range(len(neighbors)):
            for t in range(len(neighbors)):
                r = ent_pair2r.get((neighbors[h], neighbors[t]), None)
                if r:
                    m[h][t] = list(r)
                    random.shuffle(m[h][t])
                    max_rel = max(max_rel, len(r))

        info = dict()
        info['neighbors'] = neighbors
        info['graph'] = m
        sub_graphs[ent] = info
    return sub_graphs, max_neighbor, max_rel


def get_node_embedding_by_sum(word_emb, word_idx, ent_map):
    """
    Input:
        word_emb: word embeddings
        word_idx: the dict which records the mapping from word to id
        ent_map: the dict which records the mapping from entity id to entity name
    Output: entity embeddings
    """
    node_vecs = []
    for id in range(len(ent_map)):
        node_vec = []
        name = ent_map[id]
        for word in name.split():
            word_vec = word_emb[word_idx[word]]
            node_vec.append(word_vec)
        node_vec = np.sum(np.array(node_vec), 0)
        node_vecs.append(node_vec)
    node_vecs = torch.tensor(node_vecs)
    return node_vecs


def get_edges(sub_graphs):
    """
    Input: sub_graphs is a dict where key is entity's id and value is its info including neighbors and sub-graph of it
    Output: the edges in the KGs
    """
    edges = [[], []]
    ent_num = len(sub_graphs)
    for ent in range(ent_num):
        if ent not in sub_graphs:
            continue
        neighbors = sub_graphs[ent]['neighbors']
        neighbor_num = len(neighbors)

        for neighbor_id in range(neighbor_num):
            edges[0].append(ent)
            edges[1].append(neighbors[neighbor_id])

    return edges


def get_ents(triples):
    ents = set()
    for h, r, t in triples:
        ents.add(h)
        ents.add(t)
    return list(ents)


def get_sims(ent_vecs, test_pairs):
    test_left = torch.LongTensor(test_pairs[:, 0].squeeze())  # 实体对中的实体1
    test_right = torch.LongTensor(test_pairs[:, 1].squeeze())
    test_left_vec = ent_vecs[test_left]
    test_right_vec = ent_vecs[test_right]

    he = F.normalize(test_left_vec, p=2, dim=-1)
    norm_e_em = F.normalize(test_right_vec, p=2, dim=-1)
    sim = torch.matmul(he, norm_e_em.t())
    return sim


def get_explicit_alignments(norm_emb, l_cands, r_cands, threshold=-1, neighbor=None, unseed_scores=None, asl=None,
                            allMatch=False):
    bsz = norm_emb.shape[0]
    sims = [-1 for _ in range(bsz)]
    match_set = dict()
    with torch.no_grad():
        left_vecs = norm_emb[l_cands]
        right_vecs = norm_emb[r_cands]

        sim_l2r = torch.matmul(left_vecs, right_vecs.t())
        cnt = 1
        if neighbor is not None:
            sim_l2r += torch.matmul(neighbor[l_cands], neighbor[r_cands].t())
            cnt += 1
        if unseed_scores is not None:
            sim_l2r += unseed_scores
            cnt += 1
        sim_l2r /= cnt
        sim_r2l = sim_l2r.t()

        left_first = True
        if len(l_cands) > len(r_cands):
            left_first = False
            l_cands, r_cands = r_cands, l_cands
            sim_l2r, sim_r2l = sim_r2l, sim_l2r

        index_l = torch.argsort(sim_l2r, -1, True).cpu().detach().numpy()
        index_r = torch.argsort(sim_r2l, -1, True).cpu().detach().numpy()
        sim_r2l = sim_r2l.cpu().detach().numpy()

        cans_nums = [0 for _ in range(len(l_cands))]

        for i, l in enumerate(l_cands):
            candidate_l = i
            while candidate_l is not None:
                cans_num = cans_nums[candidate_l]
                candidate_r = index_l[candidate_l][cans_num]
                sim = sim_r2l[candidate_r][candidate_l]
                if candidate_r not in match_set:
                    match_set[candidate_r] = (candidate_l, sim)
                    candidate_l = None
                else:
                    exist_candidate, exist_sim = match_set[candidate_r][0], match_set[candidate_r][1]
                    if sim > exist_sim:
                        match_set[candidate_r] = (candidate_l, sim)
                        cans_nums[exist_candidate] += 1
                        candidate_l = exist_candidate
                    else:
                        cans_nums[candidate_l] += 1
    # if allMatch is True, return all aligned entities (similarly to the stable matching)
    # if allMatch is False, return the explict alignment (the entity pairs have reciprocal maximum similarity)
    if allMatch:
        match = []
        sims = []
        for ent in match_set:
            r = ent
            l = match_set[r][0]
            sims.append(match_set[r][1])
            if left_first:
                match.append((l_cands[l], r_cands[r]))
            else:
                match.append((r_cands[r], l_cands[l]))
        return match, sims

    if asl is not None:
        for left, right in asl:
            sims[left] = 1
            sims[right] = 1

    hard_match = []
    for ent in match_set:
        r = ent
        l = match_set[r][0]
        sim = match_set[r][1]
        if index_l[l][0] == r and index_r[r][0] == l:
            if sim > threshold:
                if left_first:
                    hard_match.append((l_cands[l], r_cands[r]))
                else:
                    hard_match.append((r_cands[r], l_cands[l]))
                sims[l_cands[l]] = 1
                sims[r_cands[r]] = 1

            else:
                sims[l_cands[l]] = sim
                sims[r_cands[r]] = sim

    return np.array(hard_match), sims
