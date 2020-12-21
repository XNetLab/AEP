import os

from include.AlignmentModel import propagation
from include.Config import Config
from include.Load import *
from include.AttentionGCN import AttentionGCN, training_for_gcn
from include.Util import *
from include.Eval import *
import gc

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # load graph info
    ent_num = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))  # 实体数量
    rel_num = len(set(loadfile(Config.r1, 1)) | set(loadfile(Config.r2, 1)))  # 实体数量
    print(ent_num)
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    KG = KG1 + KG2

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    test_end_index = int(Config.test_ratio * illL)
    test_ILL = np.array(ILL[:test_end_index])
    train_ILL = np.array(ILL[test_end_index:])
    ILL = np.array(ILL)

    h2t, t2h, ent_pair2r = get_kg_info(KG)
    sub_graphs, _, _ = extract_subgraph(h2t, t2h, ent_pair2r, ent_num, rel_num)
    edges = get_edges(sub_graphs)
    edges = torch.tensor(edges).to(device)
    ents1 = get_ents(KG1)
    ents2 = get_ents(KG2)
    ents1 = set(ents1)
    ents2 = set(ents2)
    sub_graph_size = []
    for ent in range(ent_num):
        sub_graph_size.append(len(sub_graphs[ent]['neighbors']) + 1)
    sub_graph_size = torch.tensor(sub_graph_size).to(device)

    # build entity embeddings based on word embeddings
    ent_map = load_id_map([Config.ent_name_path1, Config.ent_name_path2])
    if os.path.exists(Config.word_emb_path):
        word_emb, word_idx = load_word_embedding(Config.word_emb_path, Config.dim)
        node_vecs = get_node_embedding_by_sum(word_emb, word_idx, ent_map)
    else:
        from WordEmbedding import train_emb

        node_vecs = train_emb(KG, need_node_vecs=True)

    if not Config.use_neighbor_view:
        sub_graph_size = None
    if not Config.use_edit_view:
        ent_map = None

    # train model including AttentionGCN, GCN or GAT
    model = AttentionGCN(ent_num, Config.dim, Config.layers, Config.n_heads, Config.dropout, device,
                         node_vecs).to(device)
    ent_vecs = training_for_gcn(model, Config.epochs, device,
                                edges, train_ILL, test_ILL, ents1, ents2,
                                Config.unlabel_weight, Config.unlabel_k, Config.bootstrap_threshold,
                                sub_graph_size, ent_map,
                                Config.lr, Config.weight_decay, Config.gamma, Config.k, Config.dist,
                                Config.bootstrap_interval, Config.eval_interval, Config.neg_interval,
                                Config.early_stopping, Config.save_path).detach()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # propagation alignment
    model.load_state_dict(torch.load(Config.save_path))
    model.eval()
    ent_vecs = model(edges)

    sub_graph_size2 = sub_graph_size
    ent_map2 = ent_map

    ent_vecs2 = propagation(ent_vecs, Config.s_hard, Config.alpha, Config.propagation_layer, device,
                            edges, ents1, ents2,
                            train_ILL, test_ILL,
                            Config.propagation_epochs, sub_graph_size2, ent_map2)
