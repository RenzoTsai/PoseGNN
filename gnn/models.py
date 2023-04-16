import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

dgl.load_backend('pytorch')
from dgl.nn.pytorch import conv as dgl_conv

from dgl.nn.pytorch.conv import GINConv, GCN2Conv, GATConv
from dgl.nn.pytorch.glob import SumPooling


class GATLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GATLayer, self).__init__()
        self.output_dim = output_dim
        self.W = nn.Linear(input_dim, output_dim, bias=True)
        self.A = nn.Linear(2 * output_dim, 1, bias=True)
        self.dropout = nn.Dropout(0.6)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.A.weight, gain=nn.init.calculate_gain('relu'))

    def message_func(self, edges):
        Wh_j = edges.src['Wh']
        Wh_i = edges.dst['Wh']
        e_ij = self.A(torch.cat([Wh_i, Wh_j], dim=1))
        e_ij = self.relu(e_ij)
        return {'Wh_j': Wh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        e_ij = nodes.mailbox['e_ij']
        alpha_ij = self.softmax(e_ij)
        Wh_j = nodes.mailbox['Wh_j']
        h = torch.sum(alpha_ij * Wh_j, dim=1)
        return {'h': h}

    def forward(self, g, h):
        g.ndata['h'] = h
        Wh = self.W(h)
        g.ndata['Wh'] = Wh
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        return self.dropout(h)


class MultiHeadGATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, combination='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.combination = combination
        self.heads = nn.ModuleList([GATLayer(input_dim, output_dim) for _ in range(num_heads)])

    def forward(self, g, h):
        head_outs = [head(g, h) for head in self.heads]
        if self.combination == 'cat':
            return torch.cat(head_outs, dim=1)
        elif self.combination == 'avg':
            return torch.mean(torch.stack(head_outs), dim=0)


class GAT_Net(nn.Module):
    def __init__(self, in_feats, n_classes, n_hidden=10, n_layers=2, n_heads=4, dropout=0.6):
        super(GAT_Net, self).__init__()
        input_dim = in_feats
        hidden_dim = n_hidden
        output_dim = n_classes
        num_heads = n_heads
        combination_method = 'cat'
        L = n_layers
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, n_classes)

        self.layer_list = [MultiHeadGATLayer(input_dim, hidden_dim, num_heads, combination=combination_method)]
        for i in range(1, L - 1):
            self.layer_list.append(MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                                     combination=combination_method))
        self.layer_list.append(MultiHeadGATLayer(hidden_dim * num_heads, output_dim, num_heads,
                                                 combination='avg'))
        self.MultiHeadGATlayers = nn.ModuleList(self.layer_list)

    def forward(self, g, h):
        for layer in self.MultiHeadGATlayers[:-1]:
            h = self.dropout(h)
            h = layer(g, h)
            h = self.elu(h)
        if len(self.MultiHeadGATlayers) > 1:
            h = self.dropout(h)
            h = self.MultiHeadGATlayers[-1](g, h)
            h = self.softmax(h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.fc(hg)
        # return h


class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        """ Ref https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/_legacy/basic_apps/BasicTasks_pytorch.ipynb """
        super(GraphSAGEModel, self).__init__()
        self.activation_fn = activation
        self.n_hidden_layers = n_layers
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(
            dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=self.activation_fn))
        # hidden layers
        for i in range(self.n_hidden_layers):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout,
                                                 activation=self.activation_fn))
        # output layer
        self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        # Node level embeddings
        h = features
        for idx, layer in enumerate(self.layers):
            h = layer(g, h)
            if idx <= self.n_hidden_layers:
                h = self.batch_norms[idx](h)
                h = self.activation_fn(h)

        # Graph level embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        return F.softmax(hg, dim=1)


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    """https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py"""

    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(in_feats, n_hidden, bias=False))
        self.linears.append(nn.Linear(n_hidden, out_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((n_hidden))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GINModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_feats, n_hidden, n_hidden)
            else:
                mlp = MLP(n_hidden, n_hidden, n_hidden)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(in_feats, out_dim))
            else:
                self.linear_prediction.append(nn.Linear(n_hidden, out_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer



