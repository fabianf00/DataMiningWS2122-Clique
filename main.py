import numpy as np
from Clique import Clique
import random
import time

if __name__ == '__main__':
    """
    list_of_data = []
    for i in range(3600):
        value = random.randint(-2, random.randint(0, 25))
        list_of_data.append(value)
    data = np.array(list_of_data).reshape((1200, 3))
    """
    data = np.array([[0,0],
                     [1,1],
                     [2,2],
                     [3,3],
                     [4,4],
                     [5,5],
                     [6,6]])
    start = time.time()
    print("Started processing")
    clique = Clique(3, 0.1, data)
    clique.process()
    print("Finished processing")
    print("Start Labeling")
    labels_for_subspace = clique.get_all_labels()
    print("Finished Labeling")

    end = time.time()
    print(end - start, "seconds for execution")
    print(labels_for_subspace)