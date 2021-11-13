import numpy as np
from Clique import Clique

if __name__ == '__main__':
    data = np.array([[1, 2, 3, 4],
                     [5, 7, 0, 1],
                     [0, 0, 0.5, 10]])
    clique = Clique(2, 0.1, True, data)
    dense_units = clique.process()
    print(dense_units)
