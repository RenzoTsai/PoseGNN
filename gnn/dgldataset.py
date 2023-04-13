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


class HandGraph(dgl.DGLGraph):
    def __init__(self, joints, node_list):
        """
        Define a DGL graph for each hand skeleton
        """
        super().__init__()

        # Add nodes for each joint
        if node_list is None:
            node_list = [0, 1]
        self.add_nodes(len(joints))

        # Add edges for each connection between joints
        src, dst = zip(*node_list)
        self.add_edges(src, dst)

        # Set node features to be the joint positions
        self.ndata['pos'] = torch.from_numpy(np.array(joints))
        # self.ndata['pos'] = torch.tensor(joints)


class HandDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='HandDataset')

        self.graphs = []

    def process(self):
        # Load the dataset
        asl_dataset = self.load_dataset()

        # Get the unique labels
        unique_labels = np.unique(LABELS)

        # Create a graph for each hand skeleton
        graphs = []
        for hand_data in asl_dataset:
            joints = hand_data[0]
            label = hand_data[1]

            # Convert label to one-hot encoding
            label_one_hot = np.zeros(len(unique_labels))
            label_index = np.where(unique_labels == label)[0][0]
            label_one_hot[label_index] = 1

            # # Create a graph
            # graph = HandGraph(joints, np.array(NODE_LIST))
            # graph.ndata['label'] = torch.from_numpy(label_one_hot)
            #
            # # Add the graph to the list
            # self.graphs.append(graph)

            # Create a graph
            graph = dgl.DGLGraph()
            num_nodes = torch.from_numpy(np.array(joints)).shape[0]
            graph.add_nodes(num_nodes)
            graph.ndata['h'] = torch.from_numpy(np.array(joints))
            # graph.ndata['label'] = torch.from_numpy(label_one_hot)

            # Keep track of node IDs and corresponding labels
            node_ids = []
            labels = []
            for i in range(num_nodes):
                if label[i] != "":
                    node_ids.append(i)
                    labels.append(label_one_hot)

            # Add edges to the graph
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        graph.add_edge(i, j)

            # Set node labels for the nodes with labels
            graph.ndata['label'][node_ids] = torch.from_numpy(np.array(labels))

            graphs.append(graph)

        self.graphs = graphs


    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    @staticmethod
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
