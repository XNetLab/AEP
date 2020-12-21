import os
import Levenshtein
import numpy as np


def loadfile(fn, num=1):
    # load a file and return a list of tuple containing $num integers in each line
    print('loading a file ' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line.strip().split('\t')
            if num > 1:
                x = []
                for i in range(num):
                    x.append(int(th[i]))
                ret.append(x)
            else:
                ret.append(th[0])
    return ret


def load_word_embedding(emb_path, emb_dim):
    """
	Input:
	    emb_path: the path to word embeddings
	    emb_dim: the dim of embeddings
	Output: word_idx is a dict where key is word and value is id
	"""

    print('loading ' + emb_path)
    word_idx = {}
    with open(emb_path, 'r', encoding='utf-8') as f:
        vecs = []
        for line in f:
            line = line.strip()
            if len(line.split(" ")) == 2:
                continue
            info = line.split(' ')
            word = info[0]
            vec = [float(v) for v in info[1:]]
            if len(vec) != emb_dim:
                continue
            vecs.append(vec)
            word_idx[word] = len(word_idx.keys())

    emb = np.array(vecs)
    return emb, word_idx


def load_id_map(feature_paths):
    """
	Input: feature_paths is a list including paths of two KGs' feature
	Output: res is a dict where key is id and value is feature
    """
    res = {}
    for path in feature_paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('\t')
                id = int(info[0])
                if len(info) != 2:
                    feature = "**UNK**"
                else:
                    feature = info[1].lower()
                res[id] = feature
    return res


def get_edit_score(ent_num, ents1, ents2, ent_map, path=None):
    """
    Input:
        ent_num: total number of entities
        ents1: the set of entities in the KG1
        ents2: the set of entities in the KG2
        ent_map: a dict which records the mapping from id to name
        path: if path exists, it means the loading path; otherwise it means the saving path
    Output: a matrix of edit distance about entities
    """
    if path is None or not os.path.exists(path):
        unseed_name_l = []
        unseed_name_r = []
        for i in range(ent_num):
            if i in ents1:
                unseed_name_l.append(ent_map[i])
            elif i in ents2:
                unseed_name_r.append(ent_map[i])

        unseed_scores = []
        for item in range(len(unseed_name_l)):
            name1 = unseed_name_l[item]
            scores = []
            for item2 in range(len(unseed_name_r)):
                name2 = unseed_name_r[item2]
                scores.append(Levenshtein.ratio(name1, name2))
            unseed_scores.append(scores)
        unseed_scores = np.array(unseed_scores)
        if path is not None:
            np.save(path, unseed_scores)
    else:
        unseed_scores = np.load(path)

    return unseed_scores
