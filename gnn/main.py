import numpy as np
import torch

from dataloader import *
from dgldataset import DGLDataset, HandDataset, HandGestureDataset

if __name__ == '__main__':
    hands_dataset = HandGestureDataset()
    # hands_dataset.process()
    # print(hands_dataset.graphs)
