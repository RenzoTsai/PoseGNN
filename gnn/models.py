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
        # Node level embeddings
        h = features
        for layer in self.layers:
            h = layer(g, h)

        # Graph level embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        # Apply softmax activation to output layer
        hg = F.softmax(hg, dim=1)
        return hg
