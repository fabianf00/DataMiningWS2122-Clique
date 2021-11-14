import numpy as np
from Clique import Clique

if __name__ == '__main__':
    data = np.array([[1, 2, 3],
                     [5, 7, 0],
                     [0, 0, 0.5],
                     [1.1, 5, 8],
                     [0.33, 5.43, -1]
                     ])
    clique = Clique(10, 0.1, data)
    dense_units = clique.process()
    print(dense_units)
