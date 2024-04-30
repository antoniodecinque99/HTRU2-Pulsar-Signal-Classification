import utils
import DCF
from logreg import LogisticRegression
import plot
from PCA import PCA
import numpy

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, mu, sigma = utils.Z_normalize(D)

lambdas = numpy.logspace(-5, 1, num=25)

dims_to_reduce = 3

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
        print("prior:", priors[i])

        for curr_lambda in lambdas:
            lr = LogisticRegression(DZ_reduced, L, curr_lambda, priors[i])
            tmp = DCF.min_DCF(lr.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
            minDCF_array.append(tmp)
            print("For lambda = ", curr_lambda, "minDCF = ", tmp)
    
    if (i == 0):
        plot.plotDCF(lambdas, minDCF_array, "lambda")
    else:
        plot.plotDCF(lambdas, minDCF_array, "lambda", DZ.shape[0]-i)

print("")