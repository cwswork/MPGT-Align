import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        position embedding size：pos_enc_dim=15
    """
    # Laplacian

    # adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    # DGLGraph.adj_external 以外部格式返回邻接矩阵，csr是返回给定格式的 scipy 稀疏矩阵。
    A = g.adj_external(scipy_fmt='csr')
    # dgl.backend.asnumpy() 转化为numpy格式，numpy.clip()用于保留数组中在间隔范围内的值。
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    # L = sp.eye(g.number_of_nodes())
    # L = L - N * A * N

    # Eigenvectors with scipy  https://blog.csdn.net/panbaoran913/article/details/111249500
    # scipy.sparse.linalg.eigs, 求平方矩阵A的k个特征值和特征向量, ndarray
    # k所需的特征值和特征向量的数量。 k必须小于N-1。不可能计算矩阵的所有特征向量。
    # ‘which’表示偏移的特征值w’i, ‘SR’:smallest real part  || tol：float, 可选参数,特征值的相对精度(停止标准)默认值0表示机器精度。
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return lap_pos_enc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    将scipy稀疏矩阵转换为torch稀疏张量。  """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix

def re_features11(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], K+1, features.shape[1])

    for i in range(features.shape[0]):
        nodes_features[i, 0, :] = features[i]
    x = features + torch.zeros_like(features)

    for i in range(K):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, i + 1, :] = x[index]
    #nodes_features = nodes_features.squeeze()

    return nodes_features

def re_features(adj, features, K):
    #传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(K+1, features.shape[0], features.shape[1])

    # for i in range(features.shape[0]):
    #     nodes_features[i, 0, :] = features[i]
    nodes_features[0] = features
    x = features + torch.zeros_like(features)

    for i in range(K):
        x = torch.matmul(adj, x)
        # for index in range(features.shape[0]):
        #     nodes_features[index, i + 1, :] = x[index]
        nodes_features[i+1] = x #torch.unsqueeze(x, dim=1)

    nodes_features = nodes_features.transpose(0,1) # <hop+1, N, input_dim> -> <N, hop+1, input_dim>
    #nodes_features = nodes_features.squeeze()

    return nodes_features # <N, hop+1, input_dim>


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix




