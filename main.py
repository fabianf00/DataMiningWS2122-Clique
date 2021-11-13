import numpy as np
import Clique

if __name__ == '__main__':
    data = np.array([[1, 2, 3, 4],
                     [5, 7, 0, 1],
                     [0, 0, 0.5, 10]])
    clique = Clique(1, 0.1, True, data)