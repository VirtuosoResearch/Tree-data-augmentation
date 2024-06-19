import torch
import torch.nn.functional as F
import numpy as np
import random
import torch_geometric.utils as geo_utils

def drop_nodes(data, ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num*ratio)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if n not in idx_drop]

    edge_index, edge_attr = geo_utils.subgraph(idx_nondrop, data.edge_index, data.edge_attr, num_nodes=node_num)

    # # data.x = data.x[idx_nondrop]
    # edge_index = data.edge_index.numpy()

    # adj = torch.zeros((node_num, node_num))
    # adj[edge_index[0], edge_index[1]] = 1
    # adj[idx_drop, :] = 0
    # adj[:, idx_drop] = 0
    # edge_index = adj.nonzero().t()

    data.edge_index = edge_index    
    data.edge_attr = edge_attr

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def add_nodes(data, ratio):
    ''' Add nodes to the graph; extend the node indexes; and add random edges to the graph'''
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    add_num = int(node_num*ratio)

    if add_num < 1: 
        return data
    
    idx_add = np.arange(node_num, node_num+add_num)
    avg_degree = min(int(2*edge_num/node_num), node_num)

    new_edges = []
    for idx in idx_add:
        new_edges.append(
            np.array([np.ones(avg_degree)*idx, np.random.choice(node_num, size=avg_degree, replace=False)]).transpose(1, 0)
            ) 
    new_edges = np.concatenate(new_edges, axis=0)

    edge_index = data.edge_index.transpose(0, 1).numpy()
    edge_index = np.concatenate((edge_index, new_edges), axis=0)
    data.edge_index = torch.LongTensor(edge_index).transpose_(0, 1)

    if data.edge_attr is not None:
        add_edge_attr = torch.zeros(new_edges.shape[0], 2)
        add_edge_attr[:,0] = 4
        add_edge_attr= add_edge_attr.type(torch.LongTensor)
        data.edge_attr = torch.cat((data.edge_attr, add_edge_attr), dim=0)
        
        new_nodex_features = torch.zeros(add_num, 2)
        new_nodex_features[:,0] = torch.randint(0, 120, (add_num, ))
        new_nodex_features = new_nodex_features.type(torch.LongTensor)
        data.x = torch.cat((data.x, new_nodex_features), dim=0)
    else:
        new_nodex_features = torch.randn(add_num, data.x.size()[1])
        data.x = torch.cat((data.x, new_nodex_features), dim=0)

    return data
    

def permute_edges(data, ratio):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num*ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    mask = np.random.choice(edge_num, edge_num-permute_num, replace=False)
    edge_index = edge_index[mask]
    data.edge_index = torch.LongTensor(edge_index).transpose_(0, 1)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]

    return data

def add_edges(data, ratio):
    
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    add_num = int(edge_num*ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (add_num, 2))
    edge_index = np.concatenate((edge_index, idx_add), axis=0)
    data.edge_index = torch.LongTensor(edge_index).transpose_(0, 1)
    
    if data.edge_attr is not None:
        add_edge_attr = torch.zeros(add_num, 2)
        add_edge_attr[:,0] = 4
        add_edge_attr = add_edge_attr.type(torch.LongTensor)
        data.edge_attr = torch.cat((data.edge_attr, add_edge_attr), dim=0)
        data.edge_index, data.edge_attr = geo_utils.coalesce(data.edge_index, data.edge_attr, node_num, reduce='min')
    else:
        data.edge_index = geo_utils.coalesce(data.edge_index, None, node_num)

    return data

def drop_edges(data, ratio):
    
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(edge_num*ratio)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    mask = np.random.choice(edge_num, edge_num-drop_num, replace=False)
    edge_index = edge_index[mask]
    data.edge_index = torch.LongTensor(edge_index).transpose_(0, 1)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]
    
    return data


def subgraph(data, ratio):
    # ratio: the ratio of dropped subgraphs

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1-ratio))

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    edge_index, edge_attr = geo_utils.subgraph(idx_nondrop, data.edge_index, data.edge_attr, num_nodes=node_num)

    # # data.x = data.x[idx_nondrop]
    # edge_index = data.edge_index.numpy()

    # adj = torch.zeros((node_num, node_num))
    # adj[edge_index[0], edge_index[1]] = 1
    # adj[idx_drop, :] = 0
    # adj[:, idx_drop] = 0
    # edge_index = adj.nonzero().t()

    data.edge_index = edge_index
    data.edge_attr = edge_attr

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def mask_nodes(data, ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * ratio)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = 0

    return data

def mask_channels(data, ratio):

    node_num, feat_dim = data.x.size()
    mask_num = int(feat_dim * ratio)

    idx_mask = np.random.choice(feat_dim, mask_num, replace=False)
    data.x[:, idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(node_num, mask_num)), dtype=torch.float32)

    return data

def gaussain_noise(data, ratio):
    
    node_num, feat_dim = data.x.size()
    noise_num = int(node_num * ratio)

    idx_noise = np.random.choice(node_num, noise_num, replace=False)
    data.x[idx_noise] = data.x[idx_noise] + torch.tensor(np.random.normal(loc=0, scale=0.1, size=(noise_num, feat_dim)), dtype=torch.float32)

    return data

class DropNodes():

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = drop_nodes(data, self.p)
        return data
    
class PermuteEdges():

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = permute_edges(data, self.p)
        return data
    
class Subgraph():
    
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = subgraph(data, self.p)
        return data
    
class MaskNodes():
    
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = mask_nodes(data, self.p)
        return data
    
class Identity():

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        return data
    

class AddNodes():
    
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = add_nodes(data, self.p)
        return data
    
class AddEdges():
    
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = add_edges(data, self.p)
        return data

class DropEdges():
    
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = drop_edges(data, self.p)
        return data
    
class MaskChannels():

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = mask_channels(data, self.p)
        return data
    
class GaussianNoise():

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, data):
        data = gaussain_noise(data, self.p)
        return data