import os

from HandGestureDataset import HandGestureDataset

os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import torch

import dgl
from dgl.data import DGLDataset

# Define the node list - edges between joints which are stored as n * [u, v]
NODE_LIST = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
    [1, 0], [2, 1], [3, 2], [4, 3],  # Reversed Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
    [5, 0], [6, 5], [7, 6], [8, 7],  # Reversed Index finger
    [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
    [9, 0], [10, 9], [11, 10], [12, 11],  # Reversed Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
    [13, 0], [14, 13], [15, 14], [16, 15],  # Reversed Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20],  # Little finger
    [17, 0], [18, 17], [19, 18], [20, 19],  # Reversed Little finger
]


# Dummy labels for demonstration purposes
LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_dataset():
    """
    Dataset:
        number of graphs / images: 2088
        number of columns: 2 - nodes (21) and labels (1)?
        number of nodes/vertices on each hand: 21
        number of features on each node: 3 (x, y, z)
    """
    asl_dataset = HandGestureDataset()
    asl_dataset = asl_dataset.load_dataset()
    return asl_dataset


class HandGestureGraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='HandGestureGraphDataset')

        # Initialize the graphs and labels for dataset
        self.graphs = None
        self.labels = None
        self.num_classes = None
        self.num_node_feature_dim = 3

        # Process the dataset
        self.process()

    def process(self):
        # Load the dataset
        asl_dataset = load_dataset()
        self.graphs, self.labels = [], []

        # Edges - constant edges for all graphs
        src, dst = zip(*NODE_LIST)
        edges_src = torch.from_numpy(np.asarray(src))
        edges_dst = torch.from_numpy(np.asarray(dst))

        # Graph labels
        unique_labels = np.unique(LABELS)
        self.num_classes = len(unique_labels)

        # For each graphs
        for i, (joints_data, label) in enumerate(asl_dataset):
            # Create a graph
            g = dgl.graph((edges_src, edges_dst), num_nodes=len(joints_data))
            # Add node features
            g.ndata['feat'] = torch.from_numpy(np.array(joints_data).astype(np.float32))

            # Add graph labels
            # Convert label to one-hot encoding
            label_index = np.where(unique_labels == label)[0][0]
            label_one_hot = np.zeros(len(unique_labels))
            label_one_hot[label_index] = 1

            # Add self-loop edges
            g = dgl.add_self_loop(g)

            self.graphs.append(g)
            self.labels.append(label_one_hot)

        # Convert the labels to a tensor for saving
        self.labels = torch.from_numpy(np.array(self.labels))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
