import os
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import pickle
import torch

import dgl
from dgl.data import DGLDataset

# Define the node list - edges between joints which are stored as n * [u, v]
NODE_LIST = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
    [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20]  # Little finger
]

# Dummy labels for demonstration purposes
LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
          'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def load_dataset():
    """
    Dataset:
        number of graphs / images: 1619
        number of columns: 2 - nodes (21) and labels (1)?
        number of nodes/vertices on each hand: 21
        number of features on each node: 3 (x, y, z)
    """

    # Get the path to the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Load the dataset
    pickle_path = os.path.join(project_dir, 'dataset', 'asl_dataset.pickle')

    with open(pickle_path, 'rb') as f:
        asl_dataset = pickle.load(f)
    return asl_dataset


class HandGestureDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='HandGestureDataset')

        # Initialize the graphs and labels for dataset
        self.graphs = None
        self.labels = None
        self.num_classes = None

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
            g.ndata['feat'] = torch.from_numpy(np.array(joints_data))

            # Add graph labels
            # Convert label to one-hot encoding
            label_index = np.where(unique_labels == label)[0][0]
            label_one_hot = np.zeros(len(unique_labels))
            label_one_hot[label_index] = 1

            self.graphs.append(g)
            self.labels.append(label_one_hot)

        # Convert the labels to a tensor for saving
        self.labels = torch.from_numpy(np.array(self.labels))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
