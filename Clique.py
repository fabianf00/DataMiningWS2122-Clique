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
            max_value = max(self.data[:, feature]) + 1e-6  # added 1e-6 because clustering only considers [0,max_value)
            self.intervals.append((max_value / self.xi))

    # runes clique algorithm
    def process(self):
        dense_units = self.generate_one_Dimensional_Units()
        dimension = 2
        print(dense_units)
        while dimension <= self.numbers_of_features and len(dense_units) > 0:
            dense_units = self.generate_n_dimensional_dense_units(dense_units, dimension)
            dimension += 1

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

    def generate_n_dimensional_dense_units(self, previous_dense_units, dimension):
        candidates = self.join_dense_units(previous_dense_units, dimension)
        print("Number of Candidates before Prunning: ", len(candidates))
        if self.pruning:
            self.prune(candidates, previous_dense_units)
            print("Number of Candidates after Prunning: ", len(candidates))
        return candidates

    def join_dense_units(self, previous_dense_units, dimension):
        candidates = []
        for dense_unit in previous_dense_units:
            for dense_unit2 in previous_dense_units:
                joined_dense_unit = dense_unit.copy()
                joined_dense_unit.update(dense_unit2)
                if len(joined_dense_unit.keys()) == dimension and joined_dense_unit not in candidates:
                    candidates.append(joined_dense_unit)
        return candidates

    def prune(self, candidates, previous_dense_units):
        for candidate in candidates:
            if not self.subdimensions_included(candidate, previous_dense_units):
                candidates.remove(candidate)

    def subdimensions_included(self, candidate, previous_dense_units):
        for feature in candidate.keys():
            subspace_candidate = candidate.copy()
            subspace_candidate.pop(feature)
            if subspace_candidate not in previous_dense_units:
                return False
        return True
