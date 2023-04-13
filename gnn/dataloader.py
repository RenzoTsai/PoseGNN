import pickle
import os


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
