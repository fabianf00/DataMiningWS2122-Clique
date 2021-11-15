import numpy as np
from Clique import Clique
import random
import time

if __name__ == '__main__':
    list_of_data = []
    for i in range(20000):
        value = random.randint(-2, 9)
        list_of_data.append(value)
    data = np.array(list_of_data).reshape((1000, 20))
    start = time.time()
    clique = Clique(3, 0.1, data, False)
    clique.process()
    print("Finished processing")
    print("Start Labeling")
    labels_for_subspace = clique.get_all_labels()
    print("Finished Labeling")

    end = time.time()
    print(end - start, "seconds for execution")