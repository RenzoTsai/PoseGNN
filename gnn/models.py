import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

dgl.load_backend('pytorch')
from dgl.nn.pytorch import conv as dgl_conv

from dgl.nn.pytorch.conv import GINConv, GCN2Conv, GATConv
from dgl.nn.pytorch.glob import SumPooling


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_feats * num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, num_heads=1)
        self.bn2 = nn.BatchNorm1d(out_feats)

    def forward(self, g, h):
        h = h
        h = self.conv1(g, h).flatten(1)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.conv2(g, h).mean(1)
        h = self.bn2(h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return hg


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



