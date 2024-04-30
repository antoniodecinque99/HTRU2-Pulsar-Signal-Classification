import utils
from PCA import PCA
from SVM import SVM
import DCF
import plot
import numpy

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, mu, sigma = utils.Z_normalize(D)
dims_to_reduce = 2
C_array = numpy.logspace(-4, -2, num=25)

print("Linear SVM - Unbalanced")
print("Single-fold")


for i in range(dims_to_reduce):
    minDCF_array = []

    print()
    if (i == 0):
        print("No PCA")
        DZ_reduced = DZ
    else:
        print("PCA with m =", DZ.shape[0]-i)
        DZ_reduced = PCA(DZ, DZ.shape[0]-i)
    
    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)

    for i in range(len(priors)):
        print()
        print("Working on application with prior:", priors[i])

        for curr_C in C_array:
            svm = SVM(DZ_reduced, L, option1='linear', option2='unbalanced', C=curr_C,)
            tmp = DCF.min_DCF(svm.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
            minDCF_array.append(tmp)
            print("For C = ", curr_C, "minDCF = ", tmp)
    
    if (i == 0):
        plot.plotDCF(C_array, minDCF_array, "C")
    else:
        plot.plotDCF(C_array, minDCF_array, "C", DZ.shape[0]-i)


print()
print("Linear SVM - balanced")
print("Single-fold")


for i in range(2):
    minDCF_array = []

    print()
    if (i == 0):
        print("No PCA")
        DZ_reduced = DZ
    else:
        print("PCA with m =", DZ.shape[0]-i)
        DZ_reduced = PCA(DZ, DZ.shape[0]-i)
    
    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)

    for pi in range(len(priors)):
        print()
        print("SVM MODEL piT = ", priors[p])
        for p in range(len(priors)):
            print("Working on application with prior: ", priors[pi])

            for curr_C in C_array:
                svm = SVM(DZ_reduced, L, option1='linear', option2='balanced', C=curr_C, piT=priors[pi])
                tmp = DCF.min_DCF(svm.predict(DEVAL_SF), LEVAL_SF, priors[p], 1, 1)
                minDCF_array.append(tmp)
                print("For C = ", curr_C, "piT =",priors[pi], "for prior", priors[p], "minDCF = ", tmp)
    
        if (i == 0):
            plot.plotDCF(C_array, minDCF_array, "C with prior " + repr(priors[pi]))
        else:
            plot.plotDCF(C_array, minDCF_array, "C with prior " + repr(priors[pi]), DZ.shape[0]-i)
        


