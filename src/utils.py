import torch.optim as optim
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
import scipy.sparse as sp

import torch
import numpy as np
import torch.nn.functional as F
import sys

def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay', default=50)
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio', default=0.8)
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.', default=0)


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 500, 700, 900],
                                                   gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer



class Logger(object):
    def __init__(self, dir):
        self.terminal = sys.stdout
        self.log = open(f"{dir}/log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
def BCE_loss(pred, label):
    pred = pred.view(-1)
    label = label.view(-1).type(torch.float32)
    loss = nn.BCELoss()
    return loss(pred, label)

def CE_loss(pred, label):
    label = label.type(torch.int64)
    loss = nn.CrossEntropyLoss()
    return loss(pred, label)

def CE_srcweight_loss(pred, label):
    label = label.type(torch.int64)
    y_onehot = F.one_hot(label).float()
    p_y = torch.sum(y_onehot, 0) / len(y_onehot)
    class_weights = torch.tensor(1.0 / p_y, dtype=torch.float, requires_grad=False)
    loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
    return torch.mean(loss(pred, label))

def CE_loss_weight(pred, label, label_weight, weight_src):
    class_num = len(label.unique())
    label = label.type(torch.int64)
    y_onehot = F.one_hot(label).float()
    p_y = torch.sum(y_onehot, 0) / len(y_onehot)
    class_weights = torch.tensor(1.0 / p_y, dtype=torch.float, requires_grad=False)
    if weight_src:
        loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
    weight = torch.mm(y_onehot, label_weight.view(-1, 1).float())
    #loss = nn.CrossEntropyLoss(weight= label_weight)
    return torch.mean(loss(pred, label).view(-1, 1) *weight)/ class_num

def get_scores(pred, label, phase, dataset):
    auc_score = get_auc_score(pred, label)
    acc_score = get_acc_score(pred, label, dataset)
    acc_list, acc_avg_score = get_acc_scores_class(pred, label, dataset)
    f1_score = get_f1_score(pred, label)
    print(phase + ": auc score: " + str(auc_score) + "; acc score: " + str(acc_score) + "; acc class score: " + str(acc_avg_score) +"; f1 score: " + str(f1_score)); 
    print("---------------------------------------------------------------------------")
    return acc_score, auc_score, f1_score, acc_avg_score, acc_list

def get_acc_scores_class(pred, label, dataset):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)

    matrix = confusion_matrix(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    acc_list = matrix.diagonal()/matrix.sum(axis=1)
    if dataset == "MAG":
        acc_score = np.mean(acc_list[:-1])
    else:
        acc_score = np.mean(acc_list)
    return acc_list, acc_score


def get_acc_score(pred, label, dataset):
    if dataset == "MAG":
        idx = torch.nonzero(label < len(torch.unique(label)) - 1)
    else:
        idx = torch.arange(label.size(0))
    label = label[idx]
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    pred_label = pred_label[idx]
    acc_score = accuracy_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    
    return acc_score

def get_f1_score(pred, label):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    if torch.max(label) > 1:
        f1 = f1_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy(), average = "macro")
    else:
        f1 = f1_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    return f1

def get_auc_score(pred, label):
    try:
        pred = F.softmax(pred, dim=1)
        #print(torch.unique(label).size(0))
        if torch.unique(label).size(0) > 2:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class='ovr')
        else:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
        return auc_score
    except ValueError:
        return None
        pass

def plot_loss(epochs_train, loss_src_train, loss_src_valid, loss_src_test, loss_tgt_valid, loss_tgt_test, filename):
    plt.plot(epochs_train, loss_src_train, label='source_train')
    plt.plot(epochs_train, loss_src_valid, label='source_valid')
    plt.plot(epochs_train, loss_src_test, label='source_test')
    plt.plot(epochs_train, loss_tgt_valid, label='target_valid')
    plt.plot(epochs_train, loss_tgt_test, label='target_test')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('classification loss')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_dc_loss(epochs_train, dc_loss_src, dc_loss_tgt, filename):
    plt.plot(epochs_train, dc_loss_src, label='source')
    plt.plot(epochs_train, dc_loss_tgt, label='target')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('domain classifier loss')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_weight_diff(epochs, weight_diff, titlename, filename):
    plt.plot(epochs, weight_diff, label=titlename)
    plt.xlabel('Epochs')
    plt.ylabel('weight_diff with truth')
    plt.title(titlename)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def cal_edge_prob_same(src_graph, tgt_graph):
    # here src_graph and tgt_graph is the same
    graph = src_graph
    adj = graph.adj.T
    num_nodes = graph.num_nodes
    num_nodes_src = len(graph.source_mask)
    num_nodes_tgt = len(graph.target_mask)
    num_class = src_graph.num_classes

    graph_label_one_hot = sp.csr_matrix((np.ones(num_nodes), (np.arange(num_nodes), graph.y.cpu().numpy())),
                                      shape=(num_nodes, num_class))
    src_label_one_hot = sp.csr_matrix((np.ones(num_nodes_src), (graph.source_mask, graph.y[graph.source_mask].cpu().numpy())),
                                     shape=(num_nodes, num_class))
    tgt_label_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (graph.target_mask, graph.y[graph.target_mask].cpu().numpy())),
                                     shape=(num_nodes, num_class))

    src_node_num = src_label_one_hot.sum(axis=0).T * num_nodes
    tgt_node_sum = tgt_label_one_hot.sum(axis=0).T * num_nodes
    #print(src_node_num)
    #print(tgt_pred_node_sum)
    #print(tgt_node_sum)

    src_num_edge = (src_label_one_hot.T * adj * graph_label_one_hot)
    tgt_num_edge = (tgt_label_one_hot.T * adj * graph_label_one_hot)
    #print(src_num_edge)
    #print(tgt_pred_num_edge)
    #print(tgt_true_num_edge)

    src_edge_prob = src_num_edge / src_node_num
    tgt_edge_prob = tgt_num_edge / tgt_node_sum

    src_edge_prob = torch.from_numpy(np.array(src_edge_prob))
    tgt_edge_prob = torch.from_numpy(np.array(tgt_edge_prob))

    return src_edge_prob, tgt_edge_prob


def cal_edge_prob_sep(src_graph, tgt_graph):
    src_adj = src_graph.adj.T
    tgt_adj = tgt_graph.adj.T
    num_nodes_src = src_graph.num_nodes
    num_nodes_tgt = tgt_graph.num_nodes
    src_label = src_graph.y
    tgt_label = tgt_graph.y
    num_class = src_graph.num_classes

    src_label_one_hot = sp.csr_matrix((np.ones(num_nodes_src), (np.arange(num_nodes_src), src_label.cpu().numpy())),
                                      shape=(src_graph.num_nodes, num_class))
    tgt_label_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (np.arange(num_nodes_tgt), tgt_label.cpu().numpy())),
                                      shape=(tgt_graph.num_nodes, num_class))

    src_node_num = src_label_one_hot.sum(axis=0).T * num_nodes_src
    tgt_node_sum = tgt_label_one_hot.sum(axis=0).T * num_nodes_tgt
    #print(tgt_pred_node_sum)
    #print(tgt_node_sum)

    src_num_edge = (src_label_one_hot.T * src_adj * src_label_one_hot)
    tgt_num_edge = (tgt_label_one_hot.T * tgt_adj * tgt_label_one_hot)
    #print(src_num_edge)
    #print(tgt_pred_num_edge)
    #print(tgt_true_num_edge)

    src_edge_prob = src_num_edge / src_node_num
    tgt_edge_prob = tgt_num_edge / tgt_node_sum

    src_edge_prob = torch.from_numpy(np.array(src_edge_prob))
    tgt_edge_prob = torch.from_numpy(np.array(tgt_edge_prob))

    return src_edge_prob, tgt_edge_prob

def cal_edge_prob_multi(src_graphs, tgt_graphs):
    num_class = src_graphs[0].num_classes
    num_graphs = len(src_graphs)
    src_edge_prob = torch.zeros(num_class, num_class)
    tgt_edge_prob = torch.zeros(num_class, num_class)

    for (src_graph, tgt_graph) in zip(src_graphs, tgt_graphs):
        src_edge_prob_i, tgt_edge_prob_i = cal_edge_prob_sep(src_graph, tgt_graph)
        src_edge_prob = torch.add(src_edge_prob, src_edge_prob_i)
        tgt_edge_prob = torch.add(tgt_edge_prob, tgt_edge_prob_i)
    
    src_edge_prob = torch.div(src_edge_prob, num_graphs)
    tgt_edge_prob = torch.div(tgt_edge_prob, num_graphs)
    return src_edge_prob, tgt_edge_prob

def calc_true_ew(src_graph, tgt_graph, args):
    dataset = args.dataset
    if dataset == 'CSBM' or dataset == 'DBLP_ACM' or dataset == "MAG":
        src_edge_prob, tgt_edge_prob = cal_edge_prob_sep(src_graph, tgt_graph)
    elif dataset == 'cora' or dataset == 'arxiv':
        src_edge_prob, tgt_edge_prob = cal_edge_prob_sep(src_graph, tgt_graph)
    else:
        if isinstance(src_graph, list):
            src_edge_prob, tgt_edge_prob = cal_edge_prob_multi(src_graph, tgt_graph)
        else:
            src_edge_prob, tgt_edge_prob = cal_edge_prob_sep(src_graph, tgt_graph)

    #edge_weight = tgt_edge_prob / src_edge_prob
    edge_weight = (tgt_edge_prob + args.gamma_reg) / (src_edge_prob + args.gamma_reg)
    print(edge_weight.min())
    print(edge_weight.max())

    # edge_weight[edge_weight == float('inf')] = 1
    # edge_weight = torch.nan_to_num(edge_weight, nan = 1)
    
    src_graph.true_ew = edge_weight.cpu().detach().numpy()
    src_graph.ew = np.ones_like(src_graph.true_ew)
    print("true ew: " + str(src_graph.true_ew))

def calc_true_lw(src_graph, tgt_graph):
    label_onehot_src = F.one_hot(src_graph.y[src_graph.source_mask])
    label_onehot_tgt = F.one_hot(tgt_graph.y[tgt_graph.target_mask])
    num_nodes_src = torch.sum(label_onehot_src, 0)
    label_dis_src = num_nodes_src/ src_graph.source_mask.size

    num_nodes_tgt = torch.sum(label_onehot_tgt, 0)
    label_dis_tgt = num_nodes_tgt/ tgt_graph.target_mask.size

    label_weight = label_dis_tgt / label_dis_src
    src_graph.true_lw = label_weight.cpu().detach().numpy()
    src_graph.lw = np.ones_like(src_graph.true_lw)
    print("true lw: " + str(src_graph.true_lw))
    
def calc_true_beta(src_graph, tgt_graph, args):
    num_classes = src_graph.num_classes
    edge_class_src = np.reshape(src_graph.edge_class, (src_graph.num_edges, num_classes**2))
    p_edge_src = edge_class_src.sum(axis=0) / src_graph.num_edges

    edge_class_tgt = np.reshape(tgt_graph.edge_class, (tgt_graph.num_edges, num_classes**2))
    p_edge_tgt = edge_class_tgt.sum(axis=0) / tgt_graph.num_edges

    beta = (p_edge_tgt) / (p_edge_src)
    beta[beta == float('inf')] = 1
    beta = np.nan_to_num(beta, nan = 1)
    
    src_graph.true_beta = beta
    print("true beta: " + str(beta))
    print("max: " + str(np.max(beta)) + "; min: " + str(np.min(beta)))

def edge_class(graph):
    edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
    for idx in range(graph.num_edges):
        i = graph.edge_index[0][idx]
        j = graph.edge_index[1][idx]
        edge_class[idx, graph.y_hat[i], graph.y_hat[j]] = 1
    return edge_class.reshape(graph.num_edges, graph.num_classes **2)

def calc_beta(src_graph, tgt_graph, src_edge_class, tgt_edge_class):
    num_classes = src_graph.num_classes
    edge_class_src = np.reshape(src_edge_class, (src_graph.num_edges, num_classes**2))
    p_edge_src = edge_class_src.sum(axis=0) / src_graph.num_edges

    edge_class_tgt = np.reshape(tgt_edge_class, (tgt_graph.num_edges, num_classes**2))
    p_edge_tgt = edge_class_tgt.sum(axis=0) / tgt_graph.num_edges

    beta = p_edge_tgt / p_edge_src
    print("p_edge_tgt: " + str(p_edge_tgt.reshape(num_classes, num_classes)))
    beta[beta == float('inf')] = 1
    beta = np.nan_to_num(beta, nan = 1)
    return beta
    
