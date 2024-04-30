import numpy
import utils

test_path = "data/Test.txt"
train_path = "data/Train.txt"
test_data, test_labels = utils.load(test_path)
train_data, train_labels = utils.load(train_path)

max_features_val = numpy.amax(train_data, axis = 1)
min_features_val = numpy.amin(train_data, axis = 1)

for i in range(train_data.shape[0]):
    print(utils.feature_names[i], "\n", \
        min_features_val[i], "-", max_features_val[i], '\n')