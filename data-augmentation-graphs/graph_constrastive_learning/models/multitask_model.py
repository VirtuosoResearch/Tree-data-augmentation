from models import ResGCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcn_conv import GCNConv
from models.gin_model import GNN_graphpred

class MultitaskResGCN(ResGCN):

    def __init__(self, in_channels, hidden_channels, out_channels, num_feat_layers=1, num_conv_layers=3, num_fc_layers=2, gfn=False, 
                 collapse=False, residual=False, res_branch="BNConvReLU", global_pool="sum", dropout=0, edge_norm=True, use_xg=False, xg_size=0,
                 tasks=[]):
        super().__init__(in_channels[0], hidden_channels, out_channels[0], num_feat_layers, num_conv_layers, num_fc_layers, gfn, 
                         collapse, residual, res_branch, global_pool, dropout, edge_norm, use_xg, xg_size)
        
        self.task_to_labels = dict([(tasks[i], out_channels[i]) for i in range(len(out_channels))])
        self.classifiers = {}
        self.proj_heads = {} 
        self.input_bns = {}
        self.input_convs = {}

        for i, task in enumerate(tasks):
            self.input_bns[task] = nn.BatchNorm1d(in_channels[i])
            self.input_convs[task] = GCNConv(in_channels[i], hidden_channels, gfn=True)
            self.classifiers[task] = nn.Linear(hidden_channels, self.task_to_labels[task])
            self.proj_heads[task] =  nn.Sequential(nn.Linear(hidden_channels, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))
        self.task_head_list = nn.ModuleList(self.proj_heads.values())
        self.task_classifier_list = nn.ModuleList(self.classifiers.values())
        self.task_input_bn_list = nn.ModuleList(self.input_bns.values())
        self.task_input_conv_list = nn.ModuleList(self.input_convs.values())

    def forward(self, task_name, data, return_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.res_branch == "BNConvReLU":
            return self.forward_BNConvReLU(task_name, x, edge_index, batch, xg, return_features)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_BNConvReLU(self, task_name,  x, edge_index, batch, xg=None, return_features=False):
        x = self.input_bns[task_name](x)
        x = F.relu(self.input_convs[task_name](x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if return_features:
            return x
        x = self.classifiers[task_name](x)
        return F.log_softmax(x, dim=-1)

    def forward_cl(self, task_name, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        return self.forward_BNConvReLU_cl(task_name, x, edge_index, batch, xg)

    def forward_BNConvReLU_cl(self, task_name, x, edge_index, batch, xg=None):
        x = self.input_bns[task_name](x)
        x = F.relu(self.input_convs[task_name](x, edge_index))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_heads[task_name](x)
        return x
    


class MultitaskGNN(GNN_graphpred):

    def __init__(self, input_dim, num_layer, emb_dim, num_tasks, 
        JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin", residual=False,
        tasks = [], in_channels = [], out_channels = []):
        super().__init__(input_dim, num_layer, emb_dim, num_tasks, 
        JK, drop_ratio, graph_pooling, gnn_type, residual)

        self.input_lins = {}
        self.classifiers = {}
        self.proj_heads = {} 

        for i, task in enumerate(tasks):
            task_in_channel = in_channels[i]; task_out_channel = out_channels[i]
            if task_in_channel != 2:
                self.input_lins[task] = nn.Linear(task_in_channel, emb_dim)
            self.classifiers[task] = nn.Linear(emb_dim, task_out_channel)
            self.proj_heads[task] = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        self.task_head_list = nn.ModuleList(self.proj_heads.values())
        self.task_classifier_list = nn.ModuleList(self.classifiers.values())
        self.task_input_lin_list = nn.ModuleList(self.input_lins.values())

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.x_mlp.reset_parameters()
        self.gnn.reset_parameters()
        if self.graph_pooling == "attention" or self.graph_pooling[:-1] == "set2set":
            self.pool.reset_parameters()
        self.graph_pred_linear.reset_parameters()
        for module in self.proj_head:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        
        for module in self.input_lins.values():
            module.reset_parameters()
        for module in self.classifiers.values():
            module.reset_parameters()
        for module in self.proj_heads.values():
            for m in module:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    def forward(self, task_name, data, return_features = False, return_softmax=True):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x.shape[1] == 2:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            x = self.input_lins[task_name](x)
        
        node_representation = self.gnn(x, edge_index, edge_attr)

        x = self.pool(node_representation, batch)

        if return_features:
            return x

        output = self.classifiers[task_name](x)
        if return_softmax:
            return F.log_softmax(output, dim = 1)
        else:
            return output 
        
    def forward_cl(self, task_name, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if x.shape[1] == 2:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            x = self.input_lins[task_name](x)

        node_representation = self.gnn(x, edge_index, edge_attr)

        x = self.pool(node_representation, batch)

        output = self.proj_heads[task_name](x)
        
        return output 