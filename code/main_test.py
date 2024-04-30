import numpy
import utils
import plot
import DCF
from PCA import PCA
from MVG import MVG
from logreg import LogisticRegression
from SVM import SVM
from GMM import GMM


def main():
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
    dims_to_reduce = 2

    # for d in range(dims_to_reduce):
    #     if d == 0:
    #         print("No PCA")
    #     else:
    print("PCA with m=7")
    train_data_Z = train_data_Z_PCA
    test_data_Z = test_data_Z_PCA

    # MVG - all
    prints = ["Full Covariance", "Diag Covariance (Naive Bayes)", "Tied-Full Covariance", "Tied-Diag Covariance"]
    mvg_params = ["full", "diag", "tied-full", "tied-diag"]
    for type in range(len(mvg_params)):
        print("MVG", prints[type])
        gaussian_classifier = MVG(train_data_Z, train_labels, cov=mvg_params[type])
        for i in range(len(priors)):
            min_dcf = DCF.min_DCF(gaussian_classifier.predict(test_data_Z), test_labels, priors[i], 1, 1)
            print("prior=%.1f:  %.3f" %(priors[i], min_dcf))
    print()
    print()

    # LogReg
    print("LogReg - Prior 0.5")
    l = 0.0001
    piT = 0.5
    lr = LogisticRegression(train_data_Z, train_labels, l, prior=piT)
    for p in range(len(priors)):
        min_dcf = DCF.min_DCF(lr.predict(test_data_Z), test_labels, priors[p], 1, 1) 
        print("prior:", priors[p], "; pi_T for model:", piT, "; DCF:", min_dcf)
    print()
    print()
    
    # Linear SVM
    print("Linear SVM C = 5*10^-3, unbalanced")
    C = 0.005
    svm = SVM(train_data_Z, train_labels, option1='linear', option2='unbalanced',C=C)
    for i in range(len(priors)):
        minDCF = DCF.min_DCF(svm.predict(test_data_Z), test_labels, priors[i], 1, 1)
        print ("prior:", priors[i], "minDCF = ", minDCF)
    print()
    print()

    # Poly SVM
    print("Polynomial SVM C = 5*10^-5, c = 10")

    c = 10 
    C = 5*10**(-5)
    svm = SVM(train_data_Z, train_labels, option1='polynomial', C=C, c=c)
    for i in range(len(priors)):
        minDCF = DCF.min_DCF(svm.predict(test_data_Z), test_labels, priors[i], 1, 1)
        print ("prior:", priors[i], "minDCF = ", minDCF)
    print()
    print()

    # GMM
    print("GMM - 16 components")
    n_splits = 4
    gmm = GMM(train_data_Z, train_labels, n_splits, option="full")
    for i in range(len(priors)):
        min_dcf = DCF.min_DCF(gmm.predict(test_data_Z), test_labels, priors[i], 1, 1) 
        print("min DCF GMM", "with prior=%.1f:  %.3f" %(priors[i], min_dcf))





    

main()