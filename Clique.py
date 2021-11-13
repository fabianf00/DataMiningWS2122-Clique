import numpy as np


class Clique:
    # xi is the partition input parameter
    xi: int = 1

    # tau is the density threshold value
    tau: int = 0.2

    # pruning value
    pruning: bool = False

    # data is the preprocessed dataset input
    data = []

    numbers_of_features: int = 0
    numbers_of_data_points: int = 0

    intervals: list = []

    def __init__(self, xi, tau, pruning, data):
        self.xi = xi
        self.tau = tau
        self.pruning = pruning
        self.numbers_of_features = np.shape(data)[1]
        self.numbers_of_data_points = np.shape(data)[0]
        self.data = data.copy()
        for feature in range(self.numbers_of_features):
            self.data[:, feature] -= min(self.data[:, feature])
            max_value = max(self.data[:, feature]) + 1e-6 # added 1e-6 because clustering only considers [0,max_value)
            self.intervals.append((max_value / self.xi))

    # runes clique algorithm
    def process(self):
        dense_units = self.generate_one_Dimensional_Units()
        dimension = 2
        while dimension < self.numbers_of_features and len(dense_units) > 0:
            dense_units = self.generate_n_dimensional_dense_units(dense_units, dimension)
        return dense_units

    def get_unit_ID(self, feature, element):
        return int(element // self.intervals[feature])

    def generate_one_Dimensional_Units(self):
        subspaces = np.zeros((self.xi, self.numbers_of_features))

        for feature in range(self.numbers_of_features):
            for element in self.data[:, feature]:
                index = self.get_unit_ID(feature, element)
                subspaces[index, feature] += 1

        one_dim_dense_units = []

        for f in range(self.numbers_of_features):
            for unit in range(self.xi):
                if subspaces[unit, f] > self.tau * self.numbers_of_data_points:
                    dense_unit = dict({f: unit})
                    one_dim_dense_units.append(dense_unit)
        return one_dim_dense_units