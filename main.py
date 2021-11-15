import pandas as pd
from Clique import Clique
import time


def get_data_from_file(file):
    df = pd.read_csv(file)
    numpy_data = df.to_numpy()
    return numpy_data


if __name__ == '__main__':
    data = get_data_from_file("segmentation data.csv")

    start = time.time()
    print("Started processing")
    clique = Clique(3, 0.3, data)
    clique.process()
    print("Finished processing")
    print("Start Labeling")
    labels_for_subspace = clique.get_all_labels()
    print("Finished Labeling")

    end = time.time()
    """
    for subspace, labels in labels_for_subspace.items():
        print("Subspaces:", list(subspace))
        print(labels)
    """
    print(end - start, "seconds for execution")
