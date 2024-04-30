import numpy
import utils
import plot
from PCA import PCA

def main():
    # Load data
    test_path = "data/Test.txt"
    train_path = "data/Train.txt"
    train_data, train_labels = utils.load(train_path)

    # Z-normalization of data
    train_data_Z, _, _ = utils.Z_normalize(train_data)

    # Plot histograms of Z-normalized features in training data
    plot.plot_features(train_data_Z, train_labels)

    # Plot heatmap of training data
    plot.heatmap(train_data_Z, train_labels) 

    PCA2 = PCA(train_data_Z, 2)
    plot.scatter(0, 1, "Feature1", "Feature2", PCA2, train_labels, ["False", "True"])

    
main()