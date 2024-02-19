import json
import torch
import argparse
import torch.nn.functional as F
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score
import sys
import pickle

import models, utils, calc_rw
sys.path.append('..')
import data_process.pre_datasets as pre_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')

    parser.add_argument('--cls_layers', type=int, help='Number of classification layers', default=2)
    parser.add_argument('--gnn_layers', type=int, help='Number of GNN layers', default=3)
    parser.add_argument('--dc_layers', type=int, help='domain classifier layers', default=3)
    parser.add_argument('--gnn_dim', type=int, help='hidden dimension for GNN', default=300)
    parser.add_argument('--cls_dim', type=int, help='hidden dimension for classification layer', default=300)
    parser.add_argument('--dc_dim', type=int, help='hidden dimension for domain classifier', default=300)
    parser.add_argument('--backbone', type=str, help='backbone for GNN', default="GS")
    parser.add_argument('--pooling', type=str, help='pooling method in the GNN conv: add or mean', default="mean")
    parser.add_argument('--bn', type=bool, help='if batch norm in GNN', default=False)
    parser.add_argument('--valid_data', type=str,help='specify the validation dataset to select model', default="tgt")
    parser.add_argument('--valid_metric', type=str,help='specify the validation metric to select model', default="acc")
    parser.add_argument('--gnn_dc', type=bool,help='if we want to add adversarial alignment for the GNN', default=False)

    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=400)
    parser.add_argument('--opt', type=str, help='optimizer', default='adam')
    parser.add_argument('--opt_scheduler', type=str, help='optimizer scheduler', default='step')
    parser.add_argument('--opt_decay_step', type=int, default=50)
    parser.add_argument('--opt_decay_rate', type=int, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.003)
    parser.add_argument("--alphatimes", type=float, help='constant in front of the alpha for dc', default=1)
    parser.add_argument("--alphamax", type=float, help='max of alpha for dc', default=1)
    parser.add_argument("--dc_coeff", type=float, help='coeff in front of dc loss', default=1)

    parser.add_argument('--edge_rw', type=bool, help='if we want to reweight the edge source graph', default=False)
    parser.add_argument('--label_rw', type=bool, help='if we want to reweight the label', default=False)
    parser.add_argument('--weight_CE_src', type=bool, help='if we want to reweight the loss by src class', default=False)
    parser.add_argument('--ew_start', type=int, help='starting epoch for edge reweighting', default=0)
    parser.add_argument('--lw_start', type=int, help='starting epoch for label reweighting', default=0)
    parser.add_argument('--ew_freq', type=int, help='frequency for edge reweighting', default=10)
    parser.add_argument('--lw_freq', type=int, help='frequency for label reweighting', default=10)
    parser.add_argument("--rw_lmda", type=float, help='lambda to control the rw', default=1)
    parser.add_argument("--gamma_reg", type=float, help='mimic the variance of the edges to normalize the weight', default=1e-4)
    parser.add_argument('--ew_type', type=str, help='if we want to use the true edge weight', default="pseudobeta")
    parser.add_argument("--ls_lambda", type=float, help='lambda to regularize the distance to 1 in w optimization', default=25)
    parser.add_argument("--lw_lambda", type=float, help='lambda to regularize the distance to 1 in beta optimization', default=0.01)

    parser.add_argument("--dir_name", type=str, help='specify the directory name', default="debug")
    parser.add_argument("--dataset", type=str, help='specify the dataset', default='DBLP_ACM')
    parser.add_argument('--sigma', type=float, help='sigma(std) in the stochastic block model', default=0.3)
    parser.add_argument('--CSBM_set', type=str, help='setting in the synthetic dataset', default="cond1")
    parser.add_argument("--src_name", type=str, help='specify for the source dataset name',default='acm')
    parser.add_argument("--tgt_name", type=str, help='specify for the target dataset name',default='dblp')
    parser.add_argument('--start_year', type=int, help='training year start for arxiv', default=2012)
    parser.add_argument('--end_year', type=int, help='training year end for arxiv', default=2014)
    parser.add_argument('--src_end_year', type=int, help='training year end for arxiv', default=2014)
    parser.add_argument("--train_sig", type=str, help='training signal for PU dataset', default='gg')
    parser.add_argument("--test_sig", type=str, help='testing signal for PU dataset', default='gg')
    parser.add_argument("--train_PU", type=int, help='training PU level for PU dataset', default=10)
    parser.add_argument("--test_PU", type=int, help='testing PU level for PU dataset', default=50)

    return parser.parse_args()

def update_ew(src_data, tgt_data, pred_src, pred_tgt, device, args):
    
    if args.ew_type == "pseudobeta":
        ew_diff, beta_diff = calc_rw.calc_edge_rw_pseudo(src_data, tgt_data, pred_src, pred_tgt, device, args)
    else:
        beta_diff = 0
        if args.ew_type == "truth":
            edge_rw = src_data.true_ew
        else:
            edge_rw , _ = calc_rw.calc_ratio_weight(src_data, src_data.true_beta, args.gamma_reg)
        edge_weight = np.matmul(src_data.edge_class.reshape(src_data.num_edges, src_data.num_classes**2), edge_rw.reshape(src_data.num_classes**2))
        src_data.edge_weight = torch.from_numpy(edge_weight).float().to(device)
        src_data.ew = edge_rw
        ew_diff = np.abs(edge_rw - src_data.true_ew.reshape(src_data.num_classes, src_data.num_classes)).sum()

        # print("ew: "+ str(edge_rw))
        # print("max: " + str(np.max(edge_rw)) + "; min: " + str(np.min(edge_rw)))
    
    return 

def update_lw(src_data, tgt_data, pred_src, pred_tgt, args):
    yhat_tgt = torch.mean(F.softmax(pred_tgt,dim=1)[tgt_data.target_mask], dim=0)
    y_src_onehot = F.one_hot(src_data.y[src_data.source_mask]).float()
    y_src = torch.mean(y_src_onehot, dim=0)
    y_src_pred = F.softmax(pred_src,dim=1)[src_data.source_mask]
    cov_mat = torch.mm(y_src_pred.T, y_src_onehot) / len(src_data.source_mask)
    
    # print("cov mat rank: "  + str(torch.linalg.matrix_rank(cov_mat)))
    # print("cov mat condition number: " + str(torch.linalg.cond(cov_mat)))
    # print()

    label_weight = calc_rw.calc_label_rw(y_src, yhat_tgt, cov_mat, args.lw_lambda)

    return label_weight

def train_one_epoch(model, GNN_dc, src_data, tgt_data, args, opt, scheduler, epoch):
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)

    # check if the edge weights are all 1
    rw_done = 1
    all_1 = torch.ones(src_data.num_edges).to(device)
    if torch.equal(src_data.edge_weight, all_1):
        rw_done = 0

    GNN_embed_src, pred_src = model.forward(src_data, src_data.x)
    GNN_embed_tgt, pred_tgt = model.forward(tgt_data, tgt_data.x)

    # foward to the domain classifier after the GNN
    #p = (epoch + 1) / args.pre_epochs
    #alpha = min((args.alphatimes * (2. / (1. + np.exp(-10 * p)) - 1)), args.alphamax)
    alpha = min(((epoch + 1) / args.epochs), args.alphamax)
    #alpha = 0

    # alpha = float(2.0 * (1 - 0) / (1.0 + np.exp(-10*epoch / args.epochs)) - (1- 0) + 0)
    # alpha = min(alpha, args.alphamax)
    dc_pred_src = GNN_dc.forward(GNN_embed_src, alpha)
    dc_pred_tgt = GNN_dc.forward(GNN_embed_tgt, alpha)

    mask_src = src_data.source_training_mask
    label_src = src_data.y[mask_src]
    pred_src = pred_src[mask_src]

    dc_pred_src = dc_pred_src[src_data.source_mask]
    dc_pred_tgt = dc_pred_tgt[tgt_data.target_mask]
    dc_pred = torch.concat((dc_pred_src, dc_pred_tgt))
    dc_label = torch.concat((torch.zeros_like(dc_pred_src), torch.ones_like(dc_pred_tgt)))
    dc_loss = utils.BCE_loss(dc_pred, dc_label)
    dc_pred_label = dc_pred.clone()
    dc_pred_label[dc_pred_label >= 0.5] = 1
    dc_pred_label[dc_pred_label < 0.5] = 0

    dc_acc = accuracy_score(dc_label.cpu().detach().numpy(), dc_pred_label.cpu().detach().numpy())

    if args.label_rw:
        cls_loss_src = utils.CE_loss_weight(pred_src, label_src, torch.from_numpy(src_data.lw).to(device), args.weight_CE_src)
    else:
        if args.weight_CE_src:
            cls_loss_src = utils.CE_srcweight_loss(pred_src, label_src)
        else:
            cls_loss_src = utils.CE_loss(pred_src, label_src)

    loss = cls_loss_src
    
    # calculate for the weighted BCE loss for the dc
    lw_sample = torch.matmul(F.one_hot(src_data.y[src_data.source_mask]).float(), torch.from_numpy(src_data.lw).float().to(device))
    weighted_nll_source = - lw_sample.reshape(-1, 1) * torch.log(dc_pred_src + 1e-12)
    nll_target = - torch.log(1 - dc_pred_tgt + 1e-12)
    weighted_dc_loss = (torch.mean(weighted_nll_source) + torch.mean(nll_target))/2

    if args.gnn_dc:
        loss += args.dc_coeff * weighted_dc_loss

    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    GNN_embed_src, pred_src_gnn = model.forward(src_data, src_data.x)
    GNN_embed_tgt, pred_tgt_gnn = model.forward(tgt_data, tgt_data.x)

    pred_src = pred_src_gnn
    pred_tgt = pred_tgt_gnn

    # update edge weight
    pred_tgt_label = pred_tgt.argmax(dim=1)
    pred_tgt_label = pred_tgt_label.to(device)
    tgt_data.y_hat = pred_tgt_label
    if args.edge_rw and (epoch+1) >= args.ew_start and (epoch+1) % args.ew_freq == 0:
        update_ew(src_data, tgt_data, pred_src, pred_tgt, device, args)

    # update label weight
    label_weight = update_lw(src_data, tgt_data, pred_src, pred_tgt, args)
    if (epoch+1) >= args.lw_start and (epoch+1) % args.lw_freq == 0:
        src_data.lw = label_weight
    
    return rw_done


def train(source_dataset, target_dataset, args):

    if isinstance(source_dataset, list):
        input_dim = source_dataset[0].num_node_features
        num_classes = source_dataset[0].num_classes
    else:
        input_dim = source_dataset.num_node_features
        num_classes = source_dataset.num_classes
    
    # initialize models
    model = models.GNN(input_dim, num_classes, args)
    model = model.to(device)
    GNN_dc = models.GNN_dc(args.gnn_dim, 1, args)
    GNN_dc = GNN_dc.to(device)

    # build optimizers
    scheduler, opt = utils.build_optimizer(args, list(model.parameters()) + list(GNN_dc.parameters()))

    epochs_train = []
    loss_src_train = []
    loss_src_valid = []
    loss_src_test = []
    loss_tgt_valid = []
    loss_tgt_test = []
    best_valid_score = 0
    best_test_score = 0
    best_epoch = 0
    best_report = {}
    rw_rec = 0

    utils.calc_true_ew(source_dataset, target_dataset, args)
    utils.calc_true_lw(source_dataset, target_dataset)
    utils.calc_true_beta(source_dataset, target_dataset, args)

    for epoch in range(args.epochs):
        model.train()
        rw_done = train_one_epoch(model, GNN_dc, source_dataset, target_dataset, args, opt, scheduler, epoch)
        rw_rec += rw_done

        if (epoch+1) % 1 == 0:
            print("epoch " + str(epoch+1))
            epochs_train.append(epoch)

            source_embed, target_embed, inter_report, loss_dict = evaluate(source_dataset, target_dataset, model, args)

            loss_src_train.append(loss_dict['loss_src_train'])
            loss_src_valid.append(loss_dict['loss_src_valid'])
            loss_src_test.append(loss_dict['loss_src_test'])
            loss_tgt_valid.append(loss_dict['loss_tgt_valid'])
            loss_tgt_test.append(loss_dict['loss_tgt_test'])

            valid_name = args.valid_metric + '_' + args.valid_data + "_valid"
            test_name = args.valid_metric + '_' + "tgt_test"
            valid_score = inter_report[valid_name]
            
            if valid_score > best_valid_score:
                best_epoch = epoch + 1
                best_valid_score = valid_score
                best_test_score = inter_report[test_name]
                best_report = inter_report

    print("rw times: " + str(rw_rec))
    print("best_epoch: " + str(best_epoch))
    print("best_valid_score: " + str(best_valid_score))
    print("best_test_score: " + str(best_test_score))

    return best_report


def evaluate(source_dataset, target_dataset, model, args):
    
    loss_src_train, GNN_embed_src_train, acc_src_train, auc_src_train, f1_src_train, acc_cls_src_train = evaluate_dataset(source_dataset, model, "src_train", args.dataset)
    loss_src_valid, GNN_embed_src_valid, acc_src_valid, auc_src_valid, f1_src_valid, acc_cls_src_valid = evaluate_dataset(source_dataset, model, "src_valid", args.dataset)
    loss_src_test, GNN_embed_src_test, acc_src_test, auc_src_test, f1_src_test, acc_cls_src_test = evaluate_dataset(source_dataset, model, "src_test", args.dataset)
    loss_tgt_valid, GNN_embed_tgt_valid, acc_tgt_valid, auc_tgt_valid, f1_tgt_valid, acc_cls_tgt_valid = evaluate_dataset(target_dataset, model, "tgt_valid", args.dataset)
    loss_tgt_test, GNN_embed_tgt_test, acc_tgt_test, auc_tgt_test, f1_tgt_test, acc_cls_tgt_test = evaluate_dataset(target_dataset, model, "tgt_test", args.dataset)
    
    if args.valid_metric == "auc":
        defalt_metric = auc_tgt_valid
    elif args.valid_metric == "f1":
        defalt_metric = f1_tgt_valid
    elif args.valid_metric == "acccls":
        defalt_metric = acc_cls_tgt_valid
    else:
        defalt_metric = acc_tgt_valid

    report = {'acc_tgt_valid': acc_tgt_valid, 'acccls_tgt_valid': acc_cls_tgt_valid, 'auc_tgt_valid': auc_tgt_valid, 'f1_tgt_valid': f1_tgt_valid,
                    'acc_tgt_test': acc_tgt_test, 'acccls_tgt_test': acc_cls_tgt_test, 'auc_tgt_test': auc_tgt_test, 'f1_tgt_test': f1_tgt_test,
                    'acc_src_train': acc_src_train, 'acccls_src_train': acc_cls_src_train, 'auc_src_train': auc_src_train, 'f1_src_train': f1_src_train,
                    'acc_src_valid': acc_src_valid, 'acccls_src_valid': acc_cls_src_valid, 'auc_src_valid': auc_src_valid, 'f1_src_valid': f1_src_valid,
                    'acc_src_test': acc_src_test, 'acccls_src_test': acc_cls_src_test, 'auc_src_test': auc_src_test, 'f1_src_test': f1_src_test,
                    'default': defalt_metric} 
    loss_dict = {'loss_src_train': loss_src_train, 'loss_src_valid': loss_src_valid,
                    'loss_src_test': loss_src_test, 'loss_tgt_valid': loss_tgt_valid,
                    'loss_tgt_test': loss_tgt_test}

    source_embed = [[GNN_embed_src_train, GNN_embed_src_valid, GNN_embed_src_test]]
    target_embed = [[GNN_embed_tgt_valid, GNN_embed_tgt_test]]

    return source_embed, target_embed, report, loss_dict

def evaluate_dataset(data, model, phase, dataset):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        if phase == "src_train":
            mask = data.source_training_mask
        elif phase == "src_valid":
            mask = data.source_validation_mask
        elif phase == "src_test":
            mask = data.source_testing_mask
        elif phase == "tgt_valid":
            mask = data.target_validation_mask
        else:
            mask = data.target_testing_mask

        label = data.y
        GNN_embed, pred = model.forward(data, data.x)

        GNN_embed = GNN_embed[mask]
        pred = pred[mask]
        label = label[mask]

        loss = utils.CE_loss(pred, label).item()
        acc, auc, f1, acc_avg_class, acc_list = utils.get_scores(pred, label, phase, dataset)

    return loss, GNN_embed, acc, auc, f1, acc_avg_class

def get_avg_std_report(reports):
    all_keys = {k: [] for k in reports[0]}
    avg_report, avg_std_report = {}, {}
    for report in reports:
        for k in report:
            if report[k]:
                all_keys[k].append(report[k])
            else:
                all_keys[k].append(0)
    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.5f} +/- {np.std(v):.5f}' for k, v in all_keys.items()}
    return avg_report, avg_std_report

def main():
    torch.set_num_threads(10)
    args = arg_parse()

    if args.dataset == "Pileup":
        src_name = str(args.train_sig) + str(args.train_PU)
        tgt_name = str(args.test_sig) + str(args.test_PU)
    elif args.dataset == "Arxiv":
        src_name = "src" + str(args.src_end_year)
        tgt_name = "tgt" + str(args.start_year) + "_" + str(args.end_year)
    else:
        src_name = args.src_name
        tgt_name = args.tgt_name
        
    args.dir_name = args.dataset + "_" + src_name + "_" + tgt_name + "_EW" + str(args.edge_rw) + "_" + args.ew_type + \
        "_gammareg" + str(args.gamma_reg) + "_LW" + str(args.label_rw) + "_CEweightsrc" + str(args.weight_CE_src) + 'lslmda_' + str(args.ls_lambda) +\
        "_lr" + str(args.lr) + "_do" + str(args.dropout) + "_" + args.backbone + "_valid" + args.valid_metric + "_" + args.pooling
    
    directory = args.dir_name
    parent_dir = "../logs"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)
    if isdir == False:
        os.mkdir(path)
    sys.stdout = utils.Logger(path)
    print(args)

    #all_seeds = np.random.choice(range(5), num_seeds, replace=False)
    #all_seeds = [1, 3, 5, 4, 8]
    all_seeds = [1, 5, 8]

    reports = []
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args.dataset == "CSBM":
            source_dataset = pre_datasets.CSBM("src", args.sigma)
            target_dataset = pre_datasets.CSBM(args.CSBM_set, args.sigma)
            print("CSBM")
            print(source_dataset.num_edges)
        elif args.dataset == "DBLP_ACM":
            source_dataset = pre_datasets.prepare_dblp_acm("../data_files", args.src_name)
            target_dataset = pre_datasets.prepare_dblp_acm("../data_files", args.tgt_name)
            print("DBLP_ACM")
            print(source_dataset.num_edges)
        elif args.dataset == "Arxiv":
            source_dataset = pre_datasets.prepare_arxiv("../data_files", [1950, args.src_end_year])
            target_dataset = pre_datasets.prepare_arxiv("../data_files", [args.start_year, args.end_year])
            print("arxiv")
            print(source_dataset.num_edges)
        elif args.dataset == "Twitch":
            source_dataset = pre_datasets.prepare_Twitch("../data_files", args.src_name)
            target_dataset = pre_datasets.prepare_Twitch("../data_files", args.tgt_name)
        elif args.dataset == "WebKB":
            source_dataset = pre_datasets.prepare_WebKB("../data_files", args.src_name)
            target_dataset = pre_datasets.prepare_WebKB("../data_files", args.tgt_name)
        elif args.dataset == "MAG":
            source_dataset = pre_datasets.prepare_MAG("../data_files/ogbn_mag", args.src_name)
            target_dataset = pre_datasets.prepare_MAG("../data_files/ogbn_mag", args.tgt_name)
        else:
            with open(f"../data_files/pileup/test_{args.train_sig}_PU{args.train_PU}_graph" , "rb") as fp:
                source_dataset = pickle.load(fp)
            
            with open(f"../data_files/pileup/test_{args.test_sig}_PU{args.test_PU}_graph" , "rb") as fp:
                target_dataset = pickle.load(fp)
            print("Pileup")
        
        report_dict = train(source_dataset, target_dataset, args)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print('-' * 80), print('-' * 80)
    
    avg_report, avg_std_report = get_avg_std_report(reports)
    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)
    print("ggg")

if __name__ == '__main__':
    main()
