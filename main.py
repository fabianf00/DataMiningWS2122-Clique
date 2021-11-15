import pandas as pd
from Clique import Clique
import time


def get_data_from_file(file):
    df = pd.read_csv(file)
    numpy_data = df.to_numpy()
    return numpy_data


def save_labels_for_subspaces(all_labels, output_file):
    with open(output_file, 'w') as f:
        for subspace, labels in all_labels.items():
            f.write(str(list(subspace)))
            f.write('\n')
            f.write('[')
            for label in labels:
                f.write(str(label) + ", ")
            f.write(']')
            f.write('\n')


if __name__ == '__main__':
    data = get_data_from_file("Clustering.csv")

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

    save_labels_for_subspaces(labels_for_subspace, "output.txt")
