import os.path as osp
import re

import torch
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from datasets.feature_expansion import FeatureExpander
from datasets.tu_dataset import TUDatasetExt


def get_tu_dataset(name, feat_str="deg+odeg100", root=None, augment_func=None):
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0

    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        group_degree=groupd).transform

    
    dataset = TUDatasetExt(
        root, name, pre_transform=pre_transform, transform=augment_func,
        use_node_attr=True, processed_filename="data_%s.pt" % feat_str)

    dataset.data.edge_attr = None

    return dataset
