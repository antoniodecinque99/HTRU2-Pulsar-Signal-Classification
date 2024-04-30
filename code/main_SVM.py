import utils
from PCA import PCA
from SVM import SVM
import numpy
import DCF

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
ZD, mean, standard_deviation = utils.Z_normalize(D)
C = 0.005
# print ("Start linear SVM with no rebalancing (single-fold)")


dims_to_reduce = 2
for d in range(dims_to_reduce):
    if d == 0:
        print ("No PCA")
        DZ_reduced = ZD
    else:
        print("PCA=7")
        DZ_reduced = PCA(ZD, 7)
    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
    svm = SVM(DTR_SF, LTR_SF, option1='linear', option2='unbalanced',C=C)

    for i in range(len(priors)):
        print()
        print("prior = ", priors[i])
        minDCF = DCF.min_DCF(svm.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
        print ("prior:", priors[i], "minDCF = ", minDCF)


print ("Linear SVM WITH rebalancing (single-fold)")
C = 0.0005 
for d in range(dims_to_reduce):
    if d == 0:
        print ("No PCA")
        DZ_reduced = ZD
    else:
        print("PCA=7")
        DZ_reduced = PCA(ZD, 7)
    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
    for pi in range(len(priors)):
        svm = SVM(DTR_SF, LTR_SF, option1='linear', option2='balanced',C=C, piT=priors[pi])

        for i in range(len(priors)):
            print()
            print("Working on application with prior:", priors[i])
            minDCF = DCF.min_DCF(svm.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
            print ("SVMprior=", priors[pi], "prior:", priors[i], "minDCF = ", minDCF)


print ("poly SVM (single-fold)")
c = 10 
C = 5*10**(-5)

for d in range(dims_to_reduce):
    if d == 0:
        print ("No PCA")
        DZ_reduced = ZD
    else:
        print("PCA=7")
        DZ_reduced = PCA(ZD, 7)

    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)

    for i in range(len(priors)):
        svm = SVM(DTR_SF, LTR_SF, option1='polynomial', C=C, c=c)

        print()
        print("prior =", priors[i])
        minDCF = DCF.min_DCF(svm.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
        print ("prior =", priors[i], "minDCF = ", minDCF)