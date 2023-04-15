import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph

dgl.load_backend('pytorch')
from dgl.nn.pytorch import conv as dgl_conv

import numpy as np


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
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(dgl_conv.SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(dgl_conv.SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(dgl_conv.SAGEConv(n_hidden, out_dim, aggregator_type, feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        # sum up the node embeddings and output a single graph-level prediction
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return hg


class GraphClassificationModel(nn.Module):
    def __init__(self, in_feats, n_hidden, out_dim, n_layers, activation, dropout, aggregator_type):
        super(GraphClassificationModel, self).__init__()
        self.sage = GraphSAGEModel(in_feats, n_hidden, out_dim, n_layers, activation, dropout, aggregator_type)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, g, features):
        h = self.sage(g, features)
        h = dgl.mean_nodes(h, 'h')
        out = self.fc(h)
        return out.squeeze()
