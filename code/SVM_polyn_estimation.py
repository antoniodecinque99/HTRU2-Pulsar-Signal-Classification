import utils
from PCA import PCA
from SVM import SVM
import DCF
import plot
import numpy


D, L = utils.load('data/Train.txt')
DZ, mean, sigma = utils.Z_normalize(D)

c_array = [0, 1, 5, 10]
C_array = numpy.logspace(-5, -1, num=20)

print ("Start Polynomial SVM with no re-balancing")


dims_to_reduce = 2
for d in range(dims_to_reduce):
    minDCF_array = []

    if d == 0:
        print ("No PCA")
        DZ_reduced = DZ
    else:
        print("PCA=7")
        DZ_reduced = PCA(DZ, 7)
    
    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
    
    print()
    print("prior = 0.5")

    for c_little in c_array:
        print("")
        print("Estimating c: ", c_little)

        for C_big in C_array:
            svm = SVM(DZ_reduced, L, option1='polynomial', c=c_little, C=C_big, d=2)
            tmp = DCF.min_DCF(svm.predict(DEVAL_SF), LEVAL_SF, 0.5, 1, 1)
            minDCF_array.append(tmp)
            print("For C = ", C_big, "minDCF = ", tmp)

    if (d == 0):
        plot.plotDCFpoly(C_array, minDCF_array, "C")
    else:
        plot.plotDCFpoly(C_array, minDCF_array, "C", DZ.shape[0]-d)
    