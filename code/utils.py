import numpy

def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return (v.reshape(1, v.size))


def load(filename):
    # ROWS = N. OF FEATURES

    labels = []
    samples = []
    
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] != '\n':
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                data[-1] = int(data[-1].rstrip('\n'))
                samples.append(vcol(numpy.array(data[0:-1])))
                labels.append(data[-1])
    # stack samples horizontally 
    D = numpy.hstack(samples[:])
    L = numpy.array(labels)
    return D, L


def split_by_class(data, labels):
    class_0 = data[:, labels == 0]
    class_1 = data[:, labels == 1]
    return class_0, class_1


def Z_normalize(D):
    mu = D.mean(axis=1)
    sigma = D.std(axis=1)
    DZ = (D-vcol(mu))/vcol(sigma)

    return DZ, mu, sigma


def Kfold(data, labels, k):
    trainset_list = []
    evalset_list = []
    trainlabel_list = []
    evallabel_list = []
    D_s, L_s = Ksplit(data, labels, k)
    for i in range(k):
        eval_set = D_s[i]
        eval_labels = L_s[i]
        train_set = numpy.empty((data.shape[0], 0))
        train_labels = numpy.empty((0, 0))
        for j in range(k):
            if (j != i):
                # append to training set
                train_set = numpy.append(train_set, D_s[j], axis=1)
                train_labels = numpy.append(train_labels, L_s[j])
        trainset_list.append(train_set)
        trainlabel_list.append(train_labels)
        evalset_list.append(eval_set)
        evallabel_list.append(eval_labels)
    return trainset_list, trainlabel_list, evalset_list, evallabel_list
        

def Ksplit(data, labels, k, seed=0):
    data_splits = []
    label_splits = []

    nSamples = int(data.shape[1]/k)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(data.shape[1])
    for i in range(k):
        data_splits.append(data[:, idx[i*nSamples : (i+1)*nSamples]])
        label_splits.append(labels[idx[i*nSamples : (i+1)*nSamples]])
    return data_splits, label_splits


def split_single_fold(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  


def confusion_matrix(pred, actual, n_class):
    matrix = numpy.zeros((n_class, n_class))
    for i in range(len(pred)):
        matrix[int(pred[i]), int(actual[i])] += 1
    return matrix

