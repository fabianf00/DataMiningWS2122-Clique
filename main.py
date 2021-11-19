import pandas as pd
from Clique import Clique
import time
import sys
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import pickle


def load_data(filename):
    with open(filename, "rb") as file:
        serialized_data = file.read()
        label_dict = pickle.loads(serialized_data)
    return label_dict


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


def evaluate_clustering_performance(labels_for_all_subspace, true_labels):
    score = []
    max_subspace = []
    for subspace, labels in labels_for_all_subspace.items():
        score.append(normalized_mutual_info_score(labels, true_labels))
        print(list(subspace), " nmi score:", normalized_mutual_info_score(labels, labels), "Number of found "
                                                                                           "Clusters: ",
              max(labels) + 1)
        print("number of noise points:", np.count_nonzero(labels == -1))
        if normalized_mutual_info_score(labels, true_labels) == max(score):
            max_subspace = subspace
    print("Maximum NMI score:", max(score), "for subspace:", list(max_subspace))
    print("Average NMI score:", sum(score) / len(score))
    return max(score), sum(score) / len(score)


if __name__ == "__main__":

    calculate_clusters = True
    input_file = "RNAseq_801x25.csv"
    number_of_dimensions = 10
    label_file = "labels.csv"
    xi = 3
    tau = 0.1
    sep = " "

    if len(sys.argv) > 3:
        number_of_dimensions = int(sys.argv[1])
        xi = int(sys.argv[2])
        tau = float(sys.argv[3])

    output_file = "output_files/output_" + str(number_of_dimensions) + "d_" + str(xi) + "_" + "".join(
        str(tau).split(".")) + ".txt"
    labels_dict_file = "saved_labels_dict/labels_" + str(number_of_dimensions) + "d_" + str(xi) + "_" + "".join(
        str(tau).split(".")) + ".txt"

    if len(sys.argv) == 3 and sys.argv[1] == "load":
        calculate_clusters = False
        labels_dict_file = "saved_labels_dict/" + sys.argv[2]
        labels_for_subspace = load_data(labels_dict_file)

    true_labels = pd.read_csv(label_file, sep=",")["Class"].to_numpy()

    if calculate_clusters:
        df = pd.read_csv(input_file, sep=' ', header=None)
        data = df.to_numpy()[:, :number_of_dimensions]
        start = time.time()

        print("Running Clique with xi =", xi, "tau =", tau)
        clique = Clique(xi, tau, data)
        clique.process()
        labels_for_subspace = clique.get_all_labels()

        end = time.time()
        print(end - start, "seconds for execution")

    max_nmi, average_nmi = evaluate_clustering_performance(labels_for_subspace, true_labels)

    if calculate_clusters:
        save_labels_for_subspaces(labels_for_subspace, output_file, input_file, xi, tau, end - start, max_nmi,
                                  average_nmi)
        serialized_data = pickle.dumps(labels_for_subspace)

        with open(labels_dict_file, "wb") as file:
            file.write(serialized_data)
