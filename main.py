import pandas as pd
from Clique import Clique
import time
import sys
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import matplotlib.pyplot as plt


def get_data_from_file(file, separator=','):
    df = pd.read_csv(file, sep=separator)
    numpy_data = df.to_numpy()
    print(numpy_data)
    return numpy_data


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


def runOurClique(_data, _xi, _tau, _input_file, _output_file):
    print("Data shape:\n", _data.shape)
    print("Xi: ", _xi, " Tau: ", _tau)
    import time

    # import our Clique implementation
    from Clique import Clique
    # import functions from our implementation
    from main import save_labels_for_subspaces

    start = time.time()

    clique = Clique(_xi, _tau, _data)
    clique.process()
    labels_for_subspace = clique.get_all_labels()
    clique.get_all_clusters()

    end = time.time()
    print(end - start, "seconds for execution")

    save_labels_for_subspaces(labels_for_subspace, _output_file, _input_file, _xi, _tau,end - start,1,1)


df = pd.read_csv("data.csv", sep=',')
df = df.drop(columns=["Unnamed: 0"])
print(df, "\n")
df = df.iloc[0:100, 0:10]
gene_data = df.to_numpy()

runOurClique(gene_data, 3, 0.1, "data.csv", "testing.txt")

if False:
    if len(sys.argv) > 7:
        input_file = sys.argv[1]
        start_range = int(sys.argv[2])
        end_range = int(sys.argv[3])
        label_file = sys.argv[4]
        xi = int(sys.argv[5])
        tau = float(sys.argv[6])
        sep = sys.argv[7]
        output_file = sys.argv[8]
    else:
        input_file = "RNAseq_801x25.csv"
        input_file = "data.csv"
        start_range = 0
        end_range = 10
        label_file = "labels.csv"
        xi = 3
        tau = 0.1
        sep = ","
        output_file = "output_testing.txt"

    df = pd.read_csv(input_file, sep=',')
    # removes the C from the labels and converts them to numbers
    print(df, "\n")
    df = df.drop(columns=["Unnamed: 0"])
    print(df, "\n")
    df = df.iloc[0:100, 0:10]
    print(df, "\n")

    data = df.to_numpy()
    # print(gene_data)
    print(df)
    # data = df.to_numpy()[:, :10]
    # true_labels = pd.read_csv(label_file, sep=",")["Class"].to_numpy()
    # data = df.to_numpy()[0:100, start_range+1:end_range+1]
    # print(data)
    # print(true_labels)
    start = time.time()

    print("Running Clique with xi= ", xi, "tau =", tau)
    clique = Clique(xi, tau, data)
    clique.process()
    labels_for_subspace = clique.get_all_labels()

    end = time.time()
    print(end - start, "seconds for execution")

    score = [1]
    # for subspace, labels in labels_for_subspace.items():
    #     score.append(normalized_mutual_info_score(true_labels, labels))
    #     print(list(subspace), " nmi score:", normalized_mutual_info_score(true_labels, labels), "Number of found "
    #                                                                                             "Clusters: ",
    #           max(labels) + 1)
    #     # print(list(subspace), "Number of found ", "Clusters: ", max(labels) + 1)
    #     print("number of noise points:", np.count_nonzero(labels == -1))
    # print(max(score))
    # print(sum(score)/len(score))
    print(end - start, "seconds for execution")

    save_labels_for_subspaces(labels_for_subspace, output_file, input_file, xi, tau, end - start, max(score),
                              sum(score) / len(score))

    # labels = labels_for_subspace[frozenset([0, 1])]
    # plt.scatter(data[labels[:] == -1, 0], data[labels[:] == -1, 1], color="black")
    # plt.scatter(data[labels[:] == 0, 0], data[labels[:] == 0, 1], color="blue")
    # plt.scatter(data[labels[:] == 1, 0], data[labels[:] == 1, 1], color="orange")
    # plt.scatter(data[labels[:] == 2, 0], data[labels[:] == 2, 1], color="green")
    # plt.scatter(data[labels[:] == 3, 0], data[labels[:] == 3, 1], color="red")
    # plt.scatter(data[labels[:] == 4, 0], data[labels[:] == 4, 1], color="purple")
    # plt.scatter(data[labels[:] == 5, 0], data[labels[:] == 5, 1], color="brown")
    # plt.show()
