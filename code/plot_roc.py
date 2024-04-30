import plot
import utils
from PCA import PCA
from MVG import MVG
from GMM import GMM
from logreg import LogisticRegression
import numpy

def get_fpr_tpr(conf_matrix):
    FNR = conf_matrix[0][1]/(conf_matrix[0][1]+conf_matrix[1][1])
    TPR = 1-FNR
    FPR = conf_matrix[1][0]/(conf_matrix[0][0]+conf_matrix[1][0])
    return (FPR, TPR)


train_path = "data/Train.txt"
train_data, train_labels = utils.load(train_path)
# Load data
test_path = "data/Test.txt"
test_data, test_labels = utils.load(test_path)

# Z-normalization of data
train_data_Z, _, _ = utils.Z_normalize(train_data)
train_data_Z_PCA = PCA(train_data_Z, 7)
test_data_Z, _, _ = utils.Z_normalize(test_data)
test_data_Z_PCA = PCA(test_data_Z, 7)

priors = [0.5, 0.9, 0.1]
print("tied cov PCA m=7")
fprs = []
tprs = []
gaussian_classifier = MVG(train_data_Z_PCA, train_labels)
un_scores = gaussian_classifier.predict(test_data_Z_PCA)
scores = un_scores.flatten()
scores_sorted=numpy.sort(scores)
for t in scores_sorted:
    pred = (scores > t).astype(int)
    matrix = utils.confusion_matrix(pred, test_labels, 2)
    temp_fpr, temp_tpr = get_fpr_tpr(matrix)
    fprs.append(temp_fpr)
    tprs.append(temp_tpr)

fpr2 = []
tpr2 = []
print("logreg PCA m=7")

logreg = LogisticRegression(train_data_Z_PCA, train_labels, 0.0005, 0.5)
scores = logreg.predict(test_data_Z_PCA)
scores = scores.flatten()
scores_sorted =numpy.sort(scores)
for t in scores_sorted:
    pred = (scores > t).astype(int)
    matrix = utils.confusion_matrix(pred, test_labels, 2)
    temp_fpr, temp_tpr = get_fpr_tpr(matrix)
    fpr2.append(temp_fpr)
    tpr2.append(temp_tpr)


print("gmm pca 7")
fpr3 = []
tpr3 = []
gmm = GMM(train_data_Z_PCA, train_labels, 4)
scores = gmm.predict(test_data_Z_PCA)
scores = scores.flatten()
scores_sorted= numpy.sort(scores)
for t in scores_sorted:
    pred = (scores > t).astype(int)
    matrix = utils.confusion_matrix(pred, test_labels, 2)
    temp_fpr, temp_tpr = get_fpr_tpr(matrix)
    fpr3.append(temp_fpr)
    tpr3.append(temp_tpr)


plot.plot_ROC(fprs, tprs, fpr2, tpr2, fpr3, tpr3)

