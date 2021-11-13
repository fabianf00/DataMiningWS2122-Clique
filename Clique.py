import numpy as np


class Clique:
    # xi is the partition input parameter
    xi: int = 1

    # tau is the density threshold value
    tau: int = 0.2

    # pruning value
    isPruning: bool = False

    # data is the preprocessed dataset input
    data = []

    numbers_of_features: int = 2
    numbers_of_data_points: int = 2

    intervals: list = []

    def __init__(self, xi, tau, pruning, data):
        self.xi = xi
        self.tau = tau
        self.isPruning = pruning
        self.numbers_of_features = np.shape(data)[1]
        self.numbers_of_data_points = np.shape(data)[0]
        self.data = data.copy()
        for feature in range(self.numbers_of_features):
            self.data[:, feature] -= min(self.data[:, feature])
            max_value = max(self.data[:, feature])
            self.intervals.append((max_value / self.xi))

    # runes clique algorithm
    def process(self):
        dense_units = self.generate_one_Dimensional_Units()

    def get_unit_ID(self, feature, element):
        return (element - self.minValue[feature]) // self.intervals[feature]

    def generate_one_Dimensional_Units(self):
        subspaces = np.zeros(self.xsi, self.number_features)

        for feature in range(self.number_features):
            for element in self.data[:, feature]:
                index = self.get_unit_ID(feature, element)
                subspaces[index, feature] += 1

        one_dim_dense_units = []

        for f in range(self.number_features):
            for unit in range(self.xsi):
                if subspaces[unit, f] > self.tau * self.numbers_of_data_points:
                    dense_unit = dict({f: unit})
                    one_dim_dense_units.append(dense_unit)
        return one_dim_dense_units
