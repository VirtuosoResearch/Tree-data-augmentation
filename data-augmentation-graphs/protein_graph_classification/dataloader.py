import os
import json
import torch
from torch_geometric.data import Dataset, Data
from itertools import combinations

node_label_ids = {
    "ASP": 0, "ALA": 1, 
    "GLY": 2, "GLN": 3, 
    "CYS": 4, "VAL": 5, 
    "GLU": 6, "LYS": 7, 
    "PRO": 8, "THR": 9, 
    "LEU": 10, "ASN": 11, 
    "SER": 12, "PHE": 13, 
    "ILE": 14, "TRP": 15, 
    "HIS": 16, "TYR": 17, 
    "MET": 18, "ARG": 19}

class ProteinDataset(Dataset):

    def __init__(self, root = None, transform = None, pre_transform = None, pre_filter = None,
                 class_idxes = -1):
        super().__init__(root, transform, pre_transform, pre_filter)
        if class_idxes == -1:
            self.class_idxes = list(range(10463)) # all classes
        else:
            self.class_idxes = class_idxes # class_idxes has to be list 

    @property
    def raw_file_names(self):
        '''
        A list of files in the raw_dir which needs to be found in order to skip the download
        '''
        return []

    @property
    def processed_file_names(self):
        '''
        A list of files in the processed_dir which needs to be found in order to skip the processing.
        '''
        
        return [f"data_{i}.pt" for i in range(20504)]

    def download(self):
        ''' No need to download '''
        return
    
    def process(self):
        def generate_node_list(nodes_string):
            nodes = nodes_string.split()
            nodes = [int(node) for node in nodes]
            edges = list(combinations(nodes, 2))
            return edges

        def to_list(node_attr_dict):
            node_attrs = [0]*len(node_attr_dict)
            for key, val in node_attr_dict.items():
                node_attrs[key-1] = val

            return node_attrs
        
        def transform_node_labels(node_labels, node_label_ids):
            node_ys = torch.zeros(len(node_labels))
            for i, label in enumerate(node_labels):
                node_ys[i] = node_label_ids[label]
            return node_ys.type(torch.LongTensor)
        
        def transform_graph_labels(graph_labels, graph_label_ids):
            y = torch.zeros(len(graph_label_ids))
            for label in graph_labels:
                y[graph_label_ids[label]] = 1
            return y

        # Read data into huge `Data` list.
        data_list = []
        graph_label_ids = {}; fucntion_count = 0
        node_label_set = set()
        for filename in os.listdir(self.raw_dir):
            edge_list = []
            node_labels = []
            graph_labels = []
            with open(os.path.join(self.raw_dir, filename), "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line[0] >= "0" and line[0] <= "9":
                        tmp_edges = generate_node_list(line)
                        edge_list += tmp_edges
                    elif line.startswith("{"):
                        node_labels.append(eval(line))
                    elif line.startswith("["):
                        tmp_funcs = eval(line)
                        graph_labels += tmp_funcs
                        for func in tmp_funcs:
                            if func not in graph_label_ids:
                                graph_label_ids[func] = fucntion_count
                                fucntion_count += 1

            edge_list = list(set(edge_list))
            node_amino_labels = to_list(node_labels[0])
            node_uncertainties = to_list(node_labels[1])

            node_label_set = node_label_set.union(set(node_amino_labels))

            edge_index = torch.LongTensor(edge_list).T-1
            data = Data(edge_index=edge_index)
            data.graph_labels = graph_labels
            data.node_labels = node_amino_labels
            node_uncertainties = torch.Tensor([float(val) for val in node_uncertainties])
            data.node_uncertainties = node_uncertainties
            data_list.append(data)

        node_label_ids = {}; node_label_count = 0
        for label in node_label_set:
            node_label_ids[label] = node_label_count
            node_label_count += 1

        with open(os.path.join(self.processed_dir, "node_label_ids.json"), "w") as outfile:
            json.dump(node_label_ids, outfile)

        with open(os.path.join(self.processed_dir, "graph_label_ids.json"), "w") as outfile:
            json.dump(graph_label_ids, outfile)

        for data in data_list:
            data.x = transform_node_labels(data.node_labels, node_label_ids)
            data.y = transform_graph_labels(data.graph_labels, graph_label_ids)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        idx = 0
        for data in data_list:
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        data.y = data.y[self.class_idxes]
        return data