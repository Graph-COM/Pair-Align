import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import random
import scipy.sparse as sp
import os
import sys
from torch_geometric.io import read_txt_array

sys.path.append('..')
from data_process import pre_arxiv

def edges_to_adj(edges, num_node):
    edge_source = [int(i[0]) for i in edges]
    edge_target = [int(i[1]) for i in edges]
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index

def edgeidx_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    return adj

def CSBM(setting, SIGMA):
    d = 3
    num_nodes = 6000
    MU = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if setting == "cond1":
        py = [1/3, 1/3, 1/3]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
    elif setting == "cond2":
        py = [1/3, 1/3, 1/3]
        B = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    elif setting == "card1":
        py = [1/3, 1/3, 1/3]
        B = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "card2":
        py = [1/3, 1/3, 1/3]
        B = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
        B = [[i/4 for i in row] for row in B]
    elif setting == "css1":
        py = [1/3, 1/3, 1/3]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "css2":
        py = [1/3, 1/3, 1/3]
        B = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "gss1":
        py = [1/2, 1/4, 1/4]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
        B = [[i/2 for i in row] for row in B]
    elif setting == "gss2":
        py = [0.1, 0.3, 0.6]
        B = [[0.15, 0.075, 0.075], [0.075, 0.15, 0.075], [0.075, 0.075, 0.15]]
        B = [[i/2 for i in row] for row in B]
    else:
        py = [1/3, 1/3, 1/3]
        B = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
    
    N = [int(num_nodes * i) for i in py]
    B = [[i*0.1 for i in row] for row in B]

    G = nx.stochastic_block_model(N, B)
    edge_list = list(G.edges)
    
    MU_0 = MU[0]
    MU_1 = MU[1]
    MU_2 = MU[2]
    C0 = np.random.multivariate_normal(mean=MU_0, cov=np.eye(d) * SIGMA**2, size=N[0])
    C1 = np.random.multivariate_normal(mean=MU_1, cov=np.eye(d) * SIGMA**2, size=N[1])
    C2 = np.random.multivariate_normal(mean=MU_2, cov=np.eye(d) * SIGMA**2, size=N[2])

    num_nodes = np.sum(N)
    print(num_nodes)
    node_idx = np.arange(num_nodes)
    features = np.zeros((num_nodes, C1.shape[1]))
    label = np.zeros((num_nodes))

    c0_idx = node_idx[list(G.graph['partition'][0])]
    c1_idx = node_idx[list(G.graph['partition'][1])]
    c2_idx = node_idx[list(G.graph['partition'][2])]

    features[c0_idx] = C0
    features[c1_idx] = C1
    features[c2_idx] = C2

    label[c1_idx] = 1
    label[c2_idx] = 2

    random.shuffle(c0_idx)
    random.shuffle(c1_idx)
    random.shuffle(c2_idx)

    features = torch.FloatTensor(features)
    label = torch.LongTensor(label)
    idx_source_train = np.concatenate((c0_idx[:int(0.6 * len(c0_idx))],
                                 c1_idx[:int(0.6 * len(c1_idx))], c2_idx[:int(0.6 * len(c2_idx))]))
    idx_source_valid = np.concatenate((c0_idx[int(0.6 * len(c0_idx)): int(0.8 * len(c0_idx))],
                                      c1_idx[int(0.6 * len(c1_idx)) : int(0.8 * len(c1_idx))], c2_idx[int(0.6 * len(c2_idx)): int(0.8 * len(c2_idx))]))
    idx_source_test = np.concatenate((c0_idx[int(0.8 * len(c0_idx)):],
                                c1_idx[int(0.8 * len(c1_idx)):], c2_idx[int(0.8 * len(c2_idx)):]))
    idx_target_valid = np.concatenate((c0_idx[:int(0.2 * len(c0_idx))],
                                       c1_idx[:int(0.2 * len(c1_idx))], c2_idx[:int(0.2 * len(c2_idx))]))
    idx_target_test = np.concatenate((c0_idx[int(0.2 * len(c0_idx)):],
                                c1_idx[int(0.2 * len(c1_idx)):], c2_idx[int(0.2 * len(c2_idx)):]))
    num_nodes = len(label)
    adj, edge_index = edges_to_adj(edge_list, num_nodes)

    graph = Data(x=features, edge_index=edge_index, y=label)
    graph.source_training_mask = idx_source_train
    graph.source_validation_mask = idx_source_valid
    graph.source_testing_mask = idx_source_test
    graph.target_validation_mask = idx_target_valid
    graph.target_testing_mask = idx_target_test
    graph.source_mask = np.arange(graph.num_nodes)
    graph.target_mask = np.arange(graph.num_nodes)

    graph.adj = adj
    graph.num_classes = 3
    graph.edge_weight = torch.ones(graph.num_edges)
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class
    print("done")
    return graph

def prepare_dblp_acm(raw_dir, name):
    docs_path = os.path.join(raw_dir, name, 'raw/{}_docs.txt'.format(name))
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)
    x = torch.from_numpy(x).to(torch.float)

    edge_path = os.path.join(raw_dir, name, 'raw/{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

    num_node = x.size(0)
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    label_path = os.path.join(raw_dir, name, 'raw/{}_labels.txt'.format(name))
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)

    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)
    y = torch.from_numpy(y).to(torch.int64)
    graph = Data(edge_index=edge_index, x=x, y=y)
    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = y
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class

    return graph

def prepare_arxiv(root, years):
    dataset = pre_arxiv.load_nc_dataset(root, 'ogb-arxiv', years)
    idx = (dataset.test_mask == True).nonzero().view(-1).numpy()
    np.random.shuffle(idx)
    num_training = idx.shape[0]
    adj = edgeidx_to_adj(dataset.graph['edge_index'][0], dataset.graph['edge_index'][1], dataset.graph['num_nodes'])
    edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph = Data(edge_index=edge_index, x=dataset.graph['node_feat'], y=dataset.label.view(-1))
    graph.adj = adj
    graph.source_training_mask = idx[0:int(0.6*num_training)]
    graph.source_validation_mask = idx[int(0.6*num_training):int(0.8*num_training)]
    graph.source_testing_mask = idx[int(0.8*num_training):]
    graph.target_validation_mask = idx[0:int(0.2*num_training)]
    graph.target_testing_mask = idx[int(0.2*num_training):]
    graph.source_mask = idx
    graph.target_mask = idx
    graph.edge_weight = torch.ones(graph.num_edges)
    graph.num_classes = dataset.num_classes
    if torch.unique(graph.y).size(0) < graph.num_classes:
        print("miss classes")
        #return 
    graph.y_hat = graph.y 
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class
    return graph

def prepare_MAG(dir, name):
    graph = torch.load(os.path.join(dir, '{}_labels_20.pt'.format(name)))
    adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
    graph.adj = adj
    graph.edge_index = torch.from_numpy(np.array([adj.nonzero()[0], adj.nonzero()[1]])).long()
    graph.num_classes = torch.max(graph.y) + 1
    
    idx = np.arange(graph.num_nodes)
    np.random.shuffle(idx)
    idx_len = idx.shape[0]
    graph.source_training_mask = idx[0:int(0.6*idx_len)]
    graph.source_validation_mask = idx[int(0.6*idx_len):int(0.8*idx_len)]
    graph.source_testing_mask = idx[int(0.8*idx_len):]
    graph.target_validation_mask = idx[0:int(0.2*idx_len)]
    graph.target_testing_mask = idx[int(0.2*idx_len):]
    graph.source_mask = idx
    graph.target_mask = idx

    graph.edge_weight = torch.ones(graph.num_edges)

    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y[i], graph.y[j]] = 1
    graph.edge_class = edge_class

    return graph
