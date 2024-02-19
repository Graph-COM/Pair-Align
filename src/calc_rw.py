import torch
import cvxpy as cp
import numpy as np

def calc_ratio_weight(src_graph, kmm_weight, gamma_reg):
    p_eij_src = (np.sum(src_graph.edge_class.reshape(src_graph.num_edges, src_graph.num_classes**2), axis=0) / src_graph.num_edges)
    p_eij_tgt = p_eij_src * kmm_weight.reshape(-1)
    p_eij_src_reg = p_eij_src + gamma_reg
    p_eij_tgt_reg = p_eij_tgt + gamma_reg

    p_ei_src_reg = np.sum(p_eij_src_reg.reshape(src_graph.num_classes, src_graph.num_classes), axis=1)
    p_ei_tgt_reg = np.sum(p_eij_tgt_reg.reshape(src_graph.num_classes, src_graph.num_classes), axis=1)
    #print(p_ei_src)
    #print(p_ei_tgt)
    #print("cond_ratio:")
    #print((p_eij_src.reshape(src_graph.num_classes, src_graph.num_classes).T / p_ei_src).T)
    #print((p_eij_tgt.reshape(src_graph.num_classes, src_graph.num_classes).T / p_ei_tgt).T)
    gamma = ((p_eij_tgt_reg.reshape(src_graph.num_classes, src_graph.num_classes).T / p_ei_tgt_reg).T) / ((p_eij_src_reg.reshape(src_graph.num_classes, src_graph.num_classes).T / p_ei_src_reg).T)

    return gamma, p_ei_tgt_reg

def LS_optimization(cov, muhat_tgt, mu_src, lambda_reg):
    mu_src = mu_src.cpu().detach().numpy().reshape(-1).astype(np.double)
    muhat_tgt = muhat_tgt.cpu().detach().numpy().reshape(-1).astype(np.double)
    cov = cov.cpu().detach().numpy().astype(np.double)

    print("rank of Cov in edge: " + str(np.linalg.matrix_rank(cov)))
    print("condition number of Cov in edge: " + str(np.linalg.cond(cov)))
    x0 = np.ones(cov.shape[1])  
    x = cp.Variable(cov.shape[1])

    objective = cp.Minimize(cp.norm(cov @ x - muhat_tgt, 2)**2 + lambda_reg * cp.norm(x - x0, 2)**2)
    constraints = [mu_src @ x == 1 , 0 <= x]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    x_value = x.value
   
    print("edge_weight: "+ str(x_value))
    return x_value

def calc_edge_rw_pseudo(src_graph, tgt_graph, yhat_src, yhat_tgt, device, args):
    true_ew = src_graph.true_ew
    true_beta = src_graph.true_beta
    num_classes = src_graph.num_classes

    src_i_src = zip(np.arange(src_graph.num_edges), src_graph.edge_index[0])
    src_i_tgt = zip(np.arange(src_graph.num_edges), src_graph.edge_index[1])
    src_v = torch.ones(src_graph.num_edges, dtype=torch.float32)
    src_edgehat_src = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*src_i_src)), src_v, (src_graph.num_edges, src_graph.num_nodes)).to(device), yhat_src)
    src_edgehat_tgt = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*src_i_tgt)), src_v, (src_graph.num_edges, src_graph.num_nodes)).to(device), yhat_src)
    src_edgehat = torch.einsum("bi,bj->bij",src_edgehat_src, src_edgehat_tgt).view(-1, num_classes**2)

    tgt_i_src = zip(np.arange(tgt_graph.num_edges), tgt_graph.edge_index[0])
    tgt_i_tgt = zip(np.arange(tgt_graph.num_edges), tgt_graph.edge_index[1])
    tgt_v = torch.ones(tgt_graph.num_edges, dtype=torch.float32)
    tgt_edgehat_src = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*tgt_i_src)), tgt_v, (tgt_graph.num_edges, tgt_graph.num_nodes)).to(device), yhat_tgt)
    tgt_edgehat_tgt = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*tgt_i_tgt)), tgt_v, (tgt_graph.num_edges, tgt_graph.num_nodes)).to(device), yhat_tgt)
    tgt_edgehat = torch.einsum("bi,bj->bij",tgt_edgehat_src, tgt_edgehat_tgt).view(-1, num_classes**2)
    
    src_edgeclass = torch.tensor(src_graph.edge_class.reshape(-1, src_graph.num_classes**2)).float().to(device)
    C_hat = (src_edgehat.T @ src_edgeclass) / src_graph.num_edges
    muhat_tgt = torch.mean(tgt_edgehat, dim = 0).view(num_classes**2, 1)
    mu_src = torch.mean(src_edgeclass, dim = 0).view(num_classes**2, 1)
    
    w = LS_optimization(C_hat, muhat_tgt, mu_src, args.ls_lambda)
    w = np.reshape(w, (src_graph.num_classes, src_graph.num_classes))
    # beta = true_beta
    print("w: "+ str(np.reshape(w, (src_graph.num_classes, src_graph.num_classes))))
    print("max: " + str(np.max(w)) + "; min: " + str(np.min(w)))
    beta_diff = np.abs(w - true_beta.reshape(src_graph.num_classes, src_graph.num_classes)).sum()

    gamma, p_ei_tgt = calc_ratio_weight(src_graph, w, args.gamma_reg)
    print("gamma: "+ str(gamma))
    edge_rw = gamma
    edge_rw[edge_rw == float('inf')] = 1
    edge_rw = np.nan_to_num(edge_rw, nan = 1)
    print("calc_ew: " + str(edge_rw))
    print("max: " + str(np.max(edge_rw)) + "; min: " + str(np.min(edge_rw)))
    edge_weight = np.matmul(src_graph.edge_class.reshape(src_graph.num_edges, src_graph.num_classes**2), edge_rw.reshape(src_graph.num_classes**2))
    src_graph.edge_weight = torch.from_numpy(edge_weight).float().to(device)
    src_graph.ew = edge_rw
    #return edge_weight

    diff = np.abs(edge_rw - true_ew.reshape(src_graph.num_classes, src_graph.num_classes)).sum()
    return diff, beta_diff

def calc_label_rw(y_src, y_hat_tgt, cov, lambda_reg):
    y_src = y_src.cpu().detach().numpy().astype(np.double)
    y_hat_tgt = y_hat_tgt.cpu().detach().numpy().astype(np.double)
    cov = cov.cpu().detach().numpy().astype(np.double)
    
    x0 = np.ones(cov.shape[1])
    #lambda_reg = 0.005  
    x = cp.Variable(cov.shape[1])

    objective = cp.Minimize(cp.norm(cov @ x - y_hat_tgt, 2)**2 + lambda_reg * cp.norm(x - x0, 2)**2)
    constraints = [y_src @ x == 1, 0 <= x]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    x_value = x.value
    print("label_weight: "+ str(x_value))
    return x_value