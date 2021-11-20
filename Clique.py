import numpy as np


class Clique:
    xi: int = 1
    tau: float = 0.2
    pruning: bool = True
    data = []

    numbers_of_features: int = 0
    numbers_of_data_points: int = 0

    intervals = dict()
    clusters_of_all_subspaces = dict()

    def __init__(self, xi, tau, data, pruning=True):
        self.xi = xi
        self.tau = tau
        self.pruning = pruning
        self.numbers_of_features = np.shape(data)[1]
        self.numbers_of_data_points = np.shape(data)[0]
        self.data = data.copy()
        self.__preprocess_data(data)

    def __del__(self):
        self.intervals.clear()
        for subspace, clusters in self.clusters_of_all_subspaces.items():
            for cluster in clusters:
                cluster.clear()
            clusters.clear()
        self.clusters_of_all_subspaces.clear()

    # runes clique algorithm
    def process(self):
        print("Starting Calculation of one dimensional clusters")
        dense_units = self.__find_one_dimensional_dense_units()
        self.clusters_of_all_subspaces.update(self.__find_all_clusters(dense_units))
        print("Finished calculating one dimensional clusters")
        dimension = 2
        while dimension <= self.numbers_of_features and len(dense_units) > 0:
            print("Starting Calculation of", dimension, " dimensional clusters")
            dense_units = self.__find_n_dimensional_dense_units(dense_units, dimension)
            self.clusters_of_all_subspaces.update(self.__find_all_clusters(dense_units))
            print("Finished calculating", dimension, "dimensional clusters")
            dimension += 1

    def __get_unit_ID(self, feature, element):
        return int(element // self.intervals[feature])

    def __is_in_unit(self, datapoint, unit):
        for feature_id, unit_id in unit.items():
            if unit_id != self.__get_unit_ID(feature_id, datapoint[feature_id]):
                return False
        return True

    # data is shifted by minimum value therefore no minimum value is needed to calculate the unit
    # the scanned space is increased by 1e-6 because only points in the interval [0,max_value)
    # the highest value of each feature would not be considered otherwise
    def __preprocess_data(self, data):
        for feature in range(self.numbers_of_features):
            self.data[:, feature] -= min(data[:, feature])
            max_value = max(self.data[:, feature]) + 1e-6
            self.intervals.update({feature: (max_value / self.xi)})

    def get_all_clusters(self):
        return self.clusters_of_all_subspaces

    # Section FINDING DENSE UNITS
    def __find_one_dimensional_dense_units(self):
        subspaces = np.zeros((self.xi, self.numbers_of_features))

        for feature in range(self.numbers_of_features):
            for element in self.data[:, feature]:
                index = self.__get_unit_ID(feature, element)
                subspaces[index, feature] += 1

        one_dim_dense_units = []

        for f in range(self.numbers_of_features):
            for unit in range(self.xi):
                if subspaces[unit, f] >= self.tau * self.numbers_of_data_points:
                    dense_unit = dict({f: unit})
                    one_dim_dense_units.append(dense_unit)
        return one_dim_dense_units

    def __find_n_dimensional_dense_units(self, previous_dense_units, dimension):
        candidates = self.__join_dense_units(previous_dense_units, dimension)
        dense_units = []
        if self.pruning and dimension > 2:
            self.__prune(candidates, previous_dense_units)

        subspaces = np.zeros(len(candidates))
        for datapoint in self.data:
            for i in range(len(candidates)):
                if self.__is_in_unit(datapoint, candidates[i]):
                    subspaces[i] += 1

        for i in range(len(subspaces)):
            if subspaces[i] >= self.tau * self.numbers_of_data_points:
                dense_units.append(candidates[i])
        return dense_units

    def __join_dense_units(self, previous_dense_units, dimension):
        candidates = []

        for i in range(len(previous_dense_units)):
            for j in range(i + 1, len(previous_dense_units)):
                joined_dense_unit = previous_dense_units[i].copy()
                joined_dense_unit.update(previous_dense_units[j])
                if len(joined_dense_unit.keys()) == dimension and joined_dense_unit not in candidates:
                    candidates.append(joined_dense_unit)
        return candidates

    def __prune(self, candidates, previous_dense_units):
        for candidate in candidates.copy():
            if not self.__dense_unit_included_in_all_lower_subspaces(candidate, previous_dense_units):
                candidates.remove(candidate)

    def __dense_unit_included_in_all_lower_subspaces(self, candidate, previous_dense_units):
        for feature in candidate.keys():
            subspace_candidate_projection = candidate.copy()
            subspace_candidate_projection.pop(feature)
            if subspace_candidate_projection not in previous_dense_units:
                return False
        return True

    # Section FINDING CLUSTERS
    def __find_all_clusters(self, dense_units):
        set_of_subspaces = set()
        clusters_by_subspaces = dict()

        for dense_unit in dense_units:
            set_of_subspaces.add(frozenset(dense_unit.keys()))
        for subspace in set_of_subspaces:
            dense_units_in_subspace = []
            for dense_unit in dense_units:
                if dense_unit.keys() == subspace:
                    dense_units_in_subspace.append(dense_unit)

            dense_units_clusters_in_subspace = self.__find_clusters_in_subspace(dense_units_in_subspace)
            points_clusters_in_subspace = self.__get_clusters_with_point_ids(dense_units_clusters_in_subspace)
            clusters_by_subspaces.update({subspace: points_clusters_in_subspace})

        return clusters_by_subspaces

    def __find_clusters_in_subspace(self, dense_units):
        graph_matrix = self.__generate_graph_adjacency_matrix(dense_units)
        cluster_list = self.__get_dense_unit_clusters(dense_units, graph_matrix)
        return cluster_list

    def __generate_graph_adjacency_matrix(self, dense_units):
        graph_matrix = np.zeros((len(dense_units), len(dense_units)))
        for i in range(len(dense_units)):
            for j in range(i, len(dense_units)):
                connected = self.__is_adjacent(dense_units[i], dense_units[j])
                graph_matrix[i, j] = connected
                graph_matrix[j, i] = connected
        return graph_matrix

    def __is_adjacent(self, unit1, unit2):
        if unit1 == unit2:
            return 0
        distance = 0
        for feature in unit1.keys():
            distance += abs(unit1[feature] - unit2[feature])  # Manhattan Distance
            if distance > 1:
                return 0
        return 1

    def __get_dense_unit_clusters(self, dense_units, graph_matrix):
        cluster_list = []
        unvisited = dense_units.copy()
        while len(unvisited) != 0:
            cluster = self.__bfs(unvisited[0], dense_units, graph_matrix)
            for dense_unit in cluster:
                unvisited.remove(dense_unit)
            cluster_list.append(cluster)
        return cluster_list

    def __bfs(self, starting_unit, dense_units, graph_matrix):
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

    def __get_clusters_with_point_ids(self, clusters_in_subspace_with_dense_units):
        cluster_list_with_point_ids = []
        for cluster in clusters_in_subspace_with_dense_units:
            point_ids_in_cluster = []
            for i in range(len(self.data)):
                for dense_unit in cluster:
                    if self.__is_in_unit(self.data[i], dense_unit):
                        point_ids_in_cluster.append(i)
                        break
            cluster_list_with_point_ids.append(point_ids_in_cluster)
        return cluster_list_with_point_ids

    # Section LABELING
    def get_all_labels(self):
        labels = dict()
        for key in self.clusters_of_all_subspaces.keys():
            labels.update({key: self.get_labels_for_subspace(key)})
        return labels

    # the element subspace should be a iterable type (e.g. List, Tuple)
    def get_labels_for_subspace(self, subspace):
        subspace_key = frozenset(subspace)
        labels = np.full(self.numbers_of_data_points, -1)  # label of -1 is a noise point
        cluster_list = self.clusters_of_all_subspaces[subspace_key]

        for cluster_index, cluster_points in enumerate(cluster_list):
            labels[cluster_points] = cluster_index
        return labels
