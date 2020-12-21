from gensim.models import FastText
from include.Load import *
from include.Util import *
from include.Eval import *

# train word embeddings and return word embeddings( or node embeddings)
def train_emb(KG, need_node_vecs=False):
    ents = load_id_map([Config.ent_name_path1, Config.ent_name_path2])
    sentences = []
    for h, r, t in KG:
        h_ent = ents[h]
        t_ent = ents[t]
        sentences.append(h_ent.split() + t_ent.split())
        # sentences.append([h_ent, t_ent])

    # train word embeddings
    model = FastText(sentences, size=Config.dim, window=2, negative=5, min_count=1, iter=10,
                     min_n=1, max_n=3, word_ngrams=1, workers=-1)
    word_vecs = {}
    for id in ents:
        name = ents[id]
        for word in name.split():
            vec = model[word]
            word_vecs[word] = vec

    print('start to save word embeddings')
    with open(Config.word_emb_path, "w", encoding="utf-8") as f:
        for word in word_vecs:
            f.write(word + " ")
            word_vec = word_vecs[word]
            for i in range(len(word_vecs[word])):
                f.write(str(word_vec[i]))
                if i != len(word_vec) - 1:
                    f.write(' ')
            f.write("\n")

    # generate node_vecs by summation
    if need_node_vecs:
        node_vecs = []
        for id in range(len(ents)):
            vec = []
            name = ents[id]
            for word in name.split():
                word_vec = word_vecs[word]
                vec.append(word_vec)
            vec = np.sum(np.array(vec), 0)
            node_vecs.append(vec)
        node_vecs = torch.tensor(node_vecs)
        return node_vecs

    return word_vecs


def eval_emb(test_ILL, node_vecs=None):
    if node_vecs is None:
        emb_dim = Config.dim
        word_emb, word_idx = load_word_embedding(Config.word_emb_path, emb_dim)
        ent_map = load_id_map([Config.ent_name_path1, Config.ent_name_path2])
        node_vecs = get_node_embedding_by_sum(word_emb,word_idx,ent_map)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    top1, top10, mrr = get_hits_vec(node_vecs.to(device), test_ILL, device)
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top1, top10, mrr)
    print(msg)


if __name__ == '__main__':
    # load
    ent_num = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    rel_num = len(set(loadfile(Config.r1, 1)) | set(loadfile(Config.r2, 1)))
    print(ent_num)
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    ILL = np.array(ILL)

    test_ILL = np.array(ILL[:10500])
    train_ILL = np.array(ILL[10500:])
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    KG = KG1 + KG2

    print('start to train word embeddings')
    node_vecs = train_emb(KG, True)
    print('start to eval hits1 based on word embeddings')
    eval_emb(test_ILL, node_vecs)
