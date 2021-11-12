import numpy as np


class Clique:
    # xi is the partition input parameter
    xi: int = 1

    # tau is the density threshold value
    tau: int = 0.2

    # pruning value
    isPruning: bool = False

    # data is the dataset input
    data = []

    numbers_of_features: int = 2
    numbers_of_data_points: int = 2

    def __init__(self, xi, tau, pruning, data):
        self.xi = xi
        self.tau = tau
        self.isPruning = pruning
        self.data = data
        self.numbers_of_features = np.shape(data)[1]
        self.numbers_of_features = np.shape(data)[0]
    # runes clique algorithm
    def process(self):
        pass
