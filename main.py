import pandas as pd
from Clique import Clique
import time
import sys
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

def save_labels_for_subspaces(all_labels, output_file, input_file, xi, tau, execution_time, max_nmi_score,
                              avg_nmi_score):
    with open(output_file, 'w') as f:
        f.write("Data from: " + input_file + '\n')
        f.write("xi: " + str(xi) + "\n" + "tau: " + str(tau) + "\n")
        f.write("execution time: " + str(execution_time) + " s \n")
        f.write("Maximum nmi_score: " + str(max_nmi_score) + "\n")
        f.write("Average nmi_score: " + str(avg_nmi_score) + "\n\n\n")
        for subspace, labels in all_labels.items():
            f.write("subspace: " + str(list(subspace)))
            f.write('\n')
            f.write('[')
            first = True
            for label in labels:
                if first:
                    first = False
                    f.write(str(label))
                else:
                    f.write("," + str(label))
            f.write(']')
            f.write('\n\n')


if __name__ == "__main__":

    input_file = "RNAseq_801x25.csv"
    start_range = 0
    end_range = 10
    label_file = "labels.csv"
    xi = 3
    tau = 0.1
    sep = " "
    output_file = "Output files/output_10d_3_01.txt"

    df = pd.read_csv(input_file, sep=' ', header=None)

    true_labels = pd.read_csv(label_file, sep=",")["Class"].to_numpy()
    data = df.to_numpy()[:, start_range:end_range]
    start = time.time()

    print("Running Clique with xi= ", xi, "tau =", tau)
    clique = Clique(xi, tau, data)
    clique.process()
    labels_for_subspace = clique.get_all_labels()

    end = time.time()
    print(end - start, "seconds for execution")

    score = []
    max_subspace = []
    for subspace, labels in labels_for_subspace.items():
        score.append(normalized_mutual_info_score(true_labels, labels))
        print(list(subspace), " nmi score:", normalized_mutual_info_score(true_labels, labels), "Number of found "
                                                                                                "Clusters: ",
              max(labels) + 1)
        print("number of noise points:", np.count_nonzero(labels == -1))
        if normalized_mutual_info_score(true_labels, labels) == max(score):
            max_subspace = subspace
    print(max(score), max_subspace)
    print(sum(score) / len(score))
    print(end - start, "seconds for execution")

    save_labels_for_subspaces(labels_for_subspace, output_file, input_file, xi, tau, end - start, max(score),
                              sum(score) / len(score))