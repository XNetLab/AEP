class Config:
    language = 'zh_en'  # zh_en | ja_en | fr_en en_fr_15k_V1 en_de_15k_V1 dbp_wd_15k_V1 dbp_yg_15k_V1 fb_dbp
    e1 = 'data/' + language + '/ent_ids_1'
    e2 = 'data/' + language + '/ent_ids_2'
    ent_name_path1 = "data/" + language + "/id_features_1"
    ent_name_path2 = "data/" + language + "/id_features_2"
    r1 = 'data/' + language + '/rel_ids_1'
    r2 = 'data/' + language + '/rel_ids_2'

    ill = 'data/' + language + '/ref_ent_ids'
    test_ratio = 0.7
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'

    # save_path = './data/' + language + '/model.pth'
    save_path = './data/' + language + '/model'

    dropout = 0.1
    lr = 1e-3
    weight_decay = 1e-5

    dim = 300
    word_emb_path = "./data/" + language + "/word_vec_" + str(dim) + ".txt"
    n_heads = [4, 4]
    layers = 2
    epochs = 3000

    gamma = 3  # margin based loss, 3.0
    k = 50  # number of negative samples for each positive one
    dist = 2

    unlabel_weight = 1
    unlabel_k = 30
    bootstrap_interval = 10
    bootstrap_threshold = -1

    neg_interval = 10
    eval_interval = 50
    early_stopping = 200

    alpha = 0.7
    s_hard = 0.7
    propagation_layer = 1
    propagation_epochs = 1

    use_neighbor_view = True
    use_edit_view = True
