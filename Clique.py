import numpy as np


class Clique:
    # xi is the partition input parameter
    xi: int = 1

    # tau is the density threshold value
    tau: int = 0.2

    # pruning value
    pruning: bool = True

    # data is the preprocessed dataset input
    data = []

    numbers_of_features: int = 0
    numbers_of_data_points: int = 0

    intervals: list = []

    clusters_subspaces = dict()

    def __init__(self, xi, tau, data, pruning=True):
        self.xi = xi
        self.tau = tau
        self.pruning = pruning
        self.numbers_of_features = np.shape(data)[1]
        self.numbers_of_data_points = np.shape(data)[0]
        self.data = data.copy()
        for feature in range(self.numbers_of_features):
            self.data[:, feature] -= min(self.data[:, feature])  # preprocess data: minimum value is adjusted to 0
            max_value = max(self.data[:, feature]) + 1e-6  # added 1e-6 because clustering only considers [0,max_value)
            self.intervals.append((max_value / self.xi))

    # runes clique algorithm
    def process(self):
        dense_units = self.generate_one_Dimensional_Units()
        self.clusters_subspaces.update(self.find_all_clusters(dense_units))
        dimension = 2
        while dimension <= self.numbers_of_features and len(dense_units) > 0:
            dense_units = self.generate_n_dimensional_dense_units(dense_units, dimension)
            self.clusters_subspaces.update(self.find_all_clusters(dense_units))
            dimension += 1

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
                if subspaces[unit, f] >= self.tau * self.numbers_of_data_points:
                    dense_unit = dict({f: unit})
                    one_dim_dense_units.append(dense_unit)
        return one_dim_dense_units

    def generate_n_dimensional_dense_units(self, previous_dense_units, dimension):
        candidates = self.join_dense_units(previous_dense_units, dimension)
        dense_units = []
        # print("Number of Candidates before Prunning: ", len(candidates))
        if self.pruning:
            self.prune(candidates, previous_dense_units)
            # print("Number of Candidates after Prunning: ", len(candidates))

        subdim_projection = np.zeros(len(candidates))
        for datapoint in self.data:
            for i in range(len(candidates)):
                if self.is_in_unit(datapoint, candidates[i]):
                    subdim_projection[i] += 1

        for i in range(len(subdim_projection)):
            if subdim_projection[i] >= self.tau * self.numbers_of_data_points:
                dense_units.append(candidates[i])
        return dense_units

    def join_dense_units(self, previous_dense_units, dimension):
        candidates = []
        for i in range(len(previous_dense_units)):
            for j in range(i, len(previous_dense_units)):
                joined_dense_unit = previous_dense_units[i].copy()
                joined_dense_unit.update(previous_dense_units[j])
                if len(joined_dense_unit.keys()) == dimension and joined_dense_unit not in candidates:
                    candidates.append(joined_dense_unit)
        return candidates

    def prune(self, candidates, previous_dense_units):
        for candidate in candidates.copy():
            if not self.dense_unit_included_in_subspaces(candidate, previous_dense_units):
                candidates.remove(candidate)

    def dense_unit_included_in_subspaces(self, candidate, previous_dense_units):
        for feature in candidate.keys():
            subspace_candidate = candidate.copy()
            subspace_candidate.pop(feature)
            if subspace_candidate not in previous_dense_units:
                return False
        return True

    def is_in_unit(self, datapoint, unit):
        for feature_id, unit_id in unit.items():
            if unit_id != self.get_unit_ID(feature_id, datapoint[feature_id]):
                return False
        return True

    def find_all_clusters(self, dense_units):
        set_of_subspaces = set()
        clusters_by_subspaces = dict()
        clusters_by_subspaces_dense_unit = dict()
        for dense_unit in dense_units:
            set_of_subspaces.add(frozenset(dense_unit.keys()))
        for subspace in set_of_subspaces:
            dense_units_in_subspace = []
            for dense_unit in dense_units:
                if dense_unit.keys() == subspace:
                    dense_units_in_subspace.append(dense_unit)

            clusters_in_subspace_with_dense_units = self.generateClusters(dense_units_in_subspace)
            clusters_in_subspace_with_points = self.get_clusters_with_point_ids(clusters_in_subspace_with_dense_units)
            clusters_by_subspaces.update({subspace: clusters_in_subspace_with_points})
            clusters_by_subspaces_dense_unit.update({subspace: clusters_in_subspace_with_dense_units})

        return clusters_by_subspaces

    def generateClusters(self, dense_units):
        graph_matrix = self.generate_graph_adjacency_matrix(dense_units)
        clusterlist = self.get_dense_unit_clusters(dense_units, graph_matrix)
        # print(clusterlist)
        return clusterlist

    def generate_graph_adjacency_matrix(self, dense_units):
        graph_matrix = np.zeros((len(dense_units), len(dense_units)))
        for i in range(len(dense_units)):
            for j in range(i, len(dense_units)):
                connected = self.isadjacent(dense_units[i], dense_units[j])
                graph_matrix[i, j] = connected
                graph_matrix[j, i] = connected
        return graph_matrix

    def isadjacent(self, unit1, unit2):
        if unit1 == unit2:
            return 0
        distance = 0
        for feature in unit1.keys():
            distance += abs(unit1[feature] - unit2[feature])  # Manhattan Distance
            if distance > 1:
                return 0
        return 1

    def get_dense_unit_clusters(self, dense_units, graph_matrix):
        clusterlist = []
        unvisited = dense_units.copy()
        while len(unvisited) != 0:
            cluster = self.bfs(unvisited[0], dense_units, graph_matrix)
            for dense_unit in cluster:
                unvisited.remove(dense_unit)
            clusterlist.append(cluster)
        return clusterlist

    def bfs(self, starting_unit, dense_units, graph_matrix):
        connected_dense_units = [starting_unit]
        for i in range(len(dense_units)):
            if i >= len(connected_dense_units):
                break
            idx = dense_units.index(connected_dense_units[i])
            for j in range(len(dense_units)):
                if graph_matrix[idx, j] == 1:
                    dense_unit = dense_units[j]
                    if dense_unit not in connected_dense_units:
                        connected_dense_units.append(dense_unit)

        return connected_dense_units

    def get_clusters_with_point_ids(self, clusters_in_subspace_with_dense_units):
        clusterlist_with_pointids = []
        for cluster in clusters_in_subspace_with_dense_units:
            pointids_in_cluster = []
            for i in range(len(self.data)):
                for dense_unit in cluster:
                    if self.is_in_unit(self.data[i], dense_unit):
                        pointids_in_cluster.append(i)
                        break
            clusterlist_with_pointids.append(pointids_in_cluster)
        return clusterlist_with_pointids
