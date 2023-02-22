import dgl
import dgl.function as fn
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from dgl import AddSelfLoop
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from tqdm import tqdm

from dgl import AddSelfLoop

def get_data_dgl(x, y, batch_size, top_k, embs, uttr_embs, num_clusters, shuffle):
    null_cluster = 2 * num_clusters
    embs_dim = len(embs[0])
    uttr_embs_dim = len(uttr_embs[0][0])
    node_dict = {}
    edge_dict = {}
    
    for ntype in range(null_cluster + 2):
        node_dict[str(ntype)] = len(node_dict)

    edge_dict['user'] = 0
    edge_dict['system'] = 1
    edge_dict['null'] = 2
    edge_dict['self'] = 3
    
    data_len = len(x)
    indexes = np.arange(data_len)

    if shuffle == 1:
        np.random.shuffle(indexes)
        
    x_ = np.concatenate((x[indexes], 
                         np.full((batch_size - data_len % batch_size, top_k), null_cluster + 1)), 
                         axis = 0)
    y_ = np.concatenate((y[indexes], 
                         np.full((batch_size - data_len % batch_size), num_clusters)), 
                         axis = 0)

    uttr_embs_ = np.concatenate((uttr_embs[indexes], 
                                 np.zeros((batch_size - data_len % batch_size, 
                                          top_k, uttr_embs_dim))), axis = 0)
    batches_x = np.reshape(x_, (len(x_) // batch_size, batch_size, top_k))
    batches_y = np.reshape(y_, (len(y_) // batch_size, batch_size))
    batches_embs = np.reshape(uttr_embs_, (len(uttr_embs_) // batch_size, 
                                           batch_size, top_k, uttr_embs_dim))
    num_batches = len(batches_x)
    matrices = []
    all_node_features = []
    all_labels = []
    all_batches = []
    
    for ind_batch in tqdm(range(num_batches)):
        batch_x = batches_x[ind_batch]
        batch_embs = batches_embs[ind_batch]
        labels = batches_y[ind_batch]
        vertex_pairs = []

        len_batch = len(batch_x)
    
        data = {}
        node_types_dict = {}
        vtypes = []
        node_types_dict = node_types_dict.fromkeys(np.arange(2 * num_clusters + 2), 0)
        
        node_types_embs = dict([(str(i), []) for i in range(2 * num_clusters + 2)])

        for ind_graph in range(len_batch):
            graph = batch_x[ind_graph]
            graph_embs = batch_embs[ind_graph]
            
            for i in range(top_k - 1):
                
                if graph[i] == null_cluster or graph[i] == null_cluster + 1:
                    type_edge = 'null'
                elif graph[i] > num_clusters:
                    type_edge = 'user'
                else:
                    type_edge = 'system'
                
                triplet = (str(graph[i]), type_edge, str(graph[i + 1]))
                if triplet not in data:
                    data[triplet] = []

                j1 = node_types_dict[graph[i]]
                vertex_pairs.append((graph[i], j1))
                node_types_dict[graph[i]] += 1
                node_types_embs[str(graph[i])].append(np.concatenate((embs[graph[i]], \
                                                                      graph_embs[i])))
                
                j2 = node_types_dict[graph[i + 1]]
                data[triplet].append(torch.tensor([j1, j2]))
                vtypes.append(str(graph[i]))
                
            j1 = node_types_dict[graph[top_k - 1]]
            vtypes.append(str(graph[top_k - 1]))
            vertex_pairs.append((graph[top_k - 1], j1))
            node_types_dict[graph[top_k - 1]] += 1
            node_types_embs[str(graph[top_k - 1])].append(np.concatenate((embs[graph[top_k - 1]],\
                                                                          graph_embs[top_k - 1])))
        g = dgl.heterograph(data)
        
        transform = AddSelfLoop(new_etypes=True)
        g = transform(g)
        
        node_features = {}
        for i in node_types_dict.keys():
            if node_types_dict[i] != 0:
                node_features[str(i)] = torch.tensor(np.array(node_types_embs[str(i)])).float()

        for etype in g.canonical_etypes:
            g.edges[etype].data["id"] = (
                torch.ones(g.number_of_edges(etype), dtype=torch.long)
                * edge_dict[etype[1]])
            
        for ntype in g.ntypes:
            emb = nn.Parameter(
                torch.Tensor(g.number_of_nodes(ntype), 256), requires_grad=False
            )
            emb = nn.init.xavier_uniform_(emb)
#             print(np.array([embs[int(ntype)] for _ in range(g.number_of_nodes(ntype))]).shape)
#             print(np.array(node_types_embs[str(ntype)]).shape)
#             g.nodes[ntype].data["inp"] = torch.tensor(np.array(node_types_embs[str(ntype)])).float()
            g.nodes[ntype].data["inp"] = emb
        g.ndata['h'] = node_features
        
        all_batches.append((g, vertex_pairs, labels, data.keys(), np.unique(np.array(vtypes))))
    return all_batches
