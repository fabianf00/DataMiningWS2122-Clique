import numpy as np
from Clique import Clique

if __name__ == '__main__':
    data = np.array([[1, 2, 3],
                     [5, 7, 0],
                     [0, 0, 0.5],
                     [0.1, 0.1, 0.4],
                     [1.1, 5, 8],
                     [0.33, 5.43, -1]
                     ])
    clique = Clique(10, 0.1, data)
    clusters = clique.process()
    print(clusters)
    print(clique.get_labels_for_subspace((1, 2)))
