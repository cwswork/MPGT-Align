import math
import os
import random
from os.path import join

import dgl
import gensim
import scipy.sparse as sp
import numpy as np
import torch
import pickle
import torch.utils.data as Data

from align import model_util, dgl_utils
from autil import fileUtil

class load_KGs_data(object):
    def __init__(self, myconfig):
        # Load Datasets
        ent_ids_1 = fileUtil.load_ids(join(myconfig.Data_Dir, 'ent_ids_1'))
        ent_ids_2 = fileUtil.load_ids(join(myconfig.Data_Dir, 'ent_ids_2'))
        KG1_E = len(ent_ids_1)
        KG2_E = len(ent_ids_2)
        self.KG_E = KG1_E + KG2_E
        #self.ent_ids_list = ent_ids_1 + ent_ids_2
        # rel_triples1 = fileUtil.load_triples_id(join(myconfig.Data_Dir, 'triples_1'))
        # rel_triples2 = fileUtil.load_triples_id(join(myconfig.Data_Dir, 'triples_2'))
        # rel_triples = rel_triples1 + rel_triples2
        myconfig.myprint("Num of KG1 entitys:" + str(KG1_E))
        myconfig.myprint("Num of KG2 entitys:" + str(KG2_E))
        # myconfig.myprint("Num of KG1 rel_triples:" + str(len(rel_triples1)))
        # myconfig.myprint("Num of KG2 rel_triples:" + str(len(rel_triples2)))

        ### ent name embedding
        embed_dict1 = fileUtil.loadpickle(myconfig.Data_Dir + myconfig.name_embed + "_1.pkl")
        embed_dict2 = fileUtil.loadpickle(myconfig.Data_Dir + myconfig.name_embed + "_2.pkl")
        embed_dict1.update(embed_dict2)
        name_embed = []
        for i in range(self.KG_E):
            name_embed.append(embed_dict1[i][0])
        ent_embed = torch.FloatTensor(name_embed)
        ### longterm embedding
        if myconfig.longterm_emb!='':
            Longterm_embed = fileUtil.loadpickle(myconfig.Data_Dir + myconfig.longterm_emb)
            ent_embed = torch.cat((ent_embed, Longterm_embed), dim=-1)
        self.ent_embed = ent_embed
        self.input_dim = self.ent_embed.shape[1]

        path_adj_list = list()
        # # <hop+1, N, input_dim>
        for i in range(myconfig.hops):
            path_adj_tensor = fileUtil.loadpickle('{}/path12/path{}_adj_tensor'.format(myconfig.Data_Dir, i+1))
            path_adj_list.append(path_adj_tensor)

        # 0-hops embedd
        path_embed = np.zeros((myconfig.hops +1 , self.KG_E, self.input_dim)) # <hop+1, E, d>
        path_embed = torch.FloatTensor(path_embed)
        path_embed[0] = self.ent_embed
        # n-hops embedd
        for i in range(myconfig.hops):
            embedd = torch.matmul(path_adj_list[i], self.ent_embed) # A*X
            path_embed[i+1] = embedd
        self.path_embed = path_embed.transpose(0, 1) # <hop+1, E, d> -> <E, hop+1, d>

        ### Val and Test data
        val_path = myconfig.Data_Dir + 'link/valid_links_id'    ##'valid.ref'
        test_path = myconfig.Data_Dir + 'link/test_links_id'    ## 'test.ref'

        # val_ids_list, val_ar = get_links_loader(val_path)
        # test_ids_list, self.test_link = get_links_loader(test_path)
        self.train_set = train_batchData(self.path_embed, myconfig.batch_size)
        self.val_set, val_ar = val_batchData(self.path_embed, val_path, myconfig.batch_size)
        self.test_set, self.test_link = val_batchData(self.path_embed, test_path, myconfig.batch_size)

        tt_link = torch.LongTensor(self.test_link)
        self.kg_embed = [self.ent_embed[tt_link[:, 0],:], self.ent_embed[tt_link[:, 1],:] ]


def get_adj__(KG_E, rel_triples):
    du = [1] * KG_E  # # du[e] is the number of occurrences of entity e in the triples
    for (h, r, t) in rel_triples:
        du[h] += 1
        du[t] += 1

    head, end, eweight = [], [], []
    for (h, r, t) in rel_triples:
        ee = 1 / math.sqrt(du[h]) / math.sqrt(du[t])
        head.append(h)
        end.append(t)
        eweight.append(ee)

    # self-node
    for h in range(KG_E):
        du[h] += 1
        ee = 1 / math.sqrt(du[h]) / math.sqrt(du[h])
        head.append(h)
        end.append(h)
        eweight.append(ee)

    adj = sp.coo_matrix((eweight, (head, end)), shape=(KG_E, KG_E))
    # 将scipy稀疏矩阵转换为torch稀疏张量。
    adj = dgl_utils.sparse_mx_to_torch_sparse_tensor(adj)

    return adj



def val_batchData(ent_embed, link_file, batch_size):
    link_ls = fileUtil.read_links_ids(link_file)
    link_ar = np.array(link_ls)

    link_ids_list = link_ar[:, 0].tolist() + link_ar[:, 1].tolist()
    link_ids = torch.LongTensor(link_ids_list)
    val_embed = ent_embed[link_ids, :]
    batch_embed_list = []
    begid, alllen =0, len(val_embed)
    while begid < alllen:
        if begid + batch_size < alllen:
            batch_embed_list.append(val_embed[begid:begid+batch_size])
        else:
            batch_embed_list.append(val_embed[begid:])
        begid = begid+batch_size

    return batch_embed_list, link_ar

def train_batchData(ent_embed, batch_size):
    batch_embed_list = []
    begid, alllen =0, len(ent_embed)
    while begid < alllen:
        if begid + batch_size < alllen:
            batch_embed_list.append(ent_embed[begid:begid+batch_size])
        else:
            batch_embed_list.append(ent_embed[begid:])
        begid = begid + batch_size

    return batch_embed_list


#######################################

def get_adj_array__(sparse_mx, isnormalized=True):
    if isnormalized:
        sparse_mx = model_util.normalize_adj(sparse_mx).tocoo().astype(np.float32)
    else:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

    row, col, weight = sparse_mx.row, sparse_mx.col, sparse_mx.data

    #adj_arr = np.vstack((row, col, weight))
    edge_index = torch.LongTensor(np.vstack((row, col)))
    edge_weight = torch.FloatTensor(weight)
    return edge_index, edge_weight


def get_links_loader__(link_file):
    link_ls = fileUtil.read_links_ids(link_file)
    link_ar = np.array(link_ls)
    link_ids_list = link_ar[:,0].tolist() + link_ar[:,1].tolist()

    # link_left = link_ar[:,0].tolist()
    # link_right = link_ar[:,1].tolist()
    # mat_csr = adj_array_csr[link_left, :]
    # mat_csr = mat_csr[:, link_right]

    return link_ids_list, link_ar # , link_ls

def task_divide(idx, batch_size):
    ''' 划分成N个任务 '''
    total = len(idx)
    if total <= batch_size:
        return [idx]
    else:
        tasks = []
        beg =0
        while(beg<total):
            end = beg + batch_size
            if end<total:
                tasks.append(idx[beg:end])
            else:
                tasks.append(idx[beg:])

            beg = end

        return tasks

