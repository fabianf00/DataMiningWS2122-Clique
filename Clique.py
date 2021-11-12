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

    minValues: list = []
    intervals: list = []

    def __init__(self, xi, tau, pruning, data):
        self.xi = xi
        self.tau = tau
        self.isPruning = pruning
        self.data = data
        self.numbers_of_features = np.shape(data)[1]
        self.numbers_of_data_points = np.shape(data)[0]
        for feature in range(self.number_features):
            self.minValues.append(min(data[:, feature]))
            maxValue = max(data[:, feature])
            self.intervals.append((maxValue - self.minValues[feature])//self.xi)
    # runes clique algorithm
    def process(self):
        dense_units = self.generate_oneDimensionalUnits()
        pass

    def get_unit_ID(self, feature, element):
        return (element - self.minValue[feature])//self.intervals[feature]

    def generate_oneDimensionalUnits(self):
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
