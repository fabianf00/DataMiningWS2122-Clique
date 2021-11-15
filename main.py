import pandas as pd
from Clique import Clique
import time


def get_data_from_file(file):
    df = pd.read_csv(file)
    numpy_data = df.to_numpy()
    return numpy_data


def save_labels_for_subspaces(all_labels, output_file, input_file, xi, tau):
    with open(output_file, 'w') as f:
        f.write("Data from:" + input_file + '\n')
        f.write("xi: " + str(xi) + "\n" + "tau: " + str(tau) + "\n\n")
        for subspace, labels in all_labels.items():
            f.write(str(list(subspace)))
            f.write('\n')
            f.write('[')
            for label in labels:
                f.write(str(label) + ", ")
            f.write(']')
            f.write('\n')


def main(input_file, output_file, xi, tau):
    data = get_data_from_file(input_file)

    start = time.time()
    print("Started processing")
    clique = Clique(xi, tau, data)
    clique.process()
    print("Finished processing")
    print("Start Labeling")
    labels_for_subspace = clique.get_all_labels()
    print("Finished Labeling")
    end = time.time()
    print(end - start, "seconds for execution")

    save_labels_for_subspaces(labels_for_subspace, output_file, input_file, xi, tau)


if __name__ == '__main__':
    main("Clustering.csv", "Output_clustering_file.txt", 3, 0.1)
    main("wine-clustering.csv", "output_wine.txt", 3, 0.1)
    main("segmentation data.csv", "Output_seg.txt", 4, 0.05)

