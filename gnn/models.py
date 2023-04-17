import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

dgl.load_backend('pytorch')
from dgl.nn.pytorch import conv as dgl_conv

from dgl.nn.pytorch.conv import GINConv, GCN2Conv, GATConv
from dgl.nn.pytorch.glob import SumPooling


class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, n_layers):
        super(GATModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.n_layers = n_layers

        self.conv_layers.append(GATConv(in_feats, hidden_feats, num_heads=num_heads, residual=True))
        self.bn_layers.append(nn.BatchNorm1d(hidden_feats * num_heads))

        for i in range(1, n_layers - 1):
            self.conv_layers.append(GATConv(hidden_feats * num_heads, hidden_feats, num_heads=num_heads, residual=True))
            self.bn_layers.append(nn.BatchNorm1d(hidden_feats * num_heads))

        self.conv_layers.append(GATConv(hidden_feats * num_heads, out_feats, num_heads=6))
        self.bn_layers.append(nn.BatchNorm1d(out_feats))

    def forward(self, g, h):
        h = h

        for i in range(self.n_layers - 1):
            h = self.conv_layers[i](g, h).flatten(1)
            h = self.bn_layers[i](h)
            h = F.elu(h)

        h = self.conv_layers[self.n_layers - 1](g, h).mean(1)
        h = self.bn_layers[self.n_layers - 1](h)

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
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(in_feats, n_hidden, bias=False))
        self.linears.append(nn.Linear(n_hidden, out_dim, bias=False))
        self.xaiver_init()
        self.batch_norm = nn.BatchNorm1d(n_hidden)

    def xaiver_init(self):
        for layer in self.linears:
            nn.init.xavier_uniform_(layer.weight)

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
        )

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
