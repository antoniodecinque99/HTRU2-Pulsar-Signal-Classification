import utils
from MVG import MVG
from PCA import PCA
import DCF
import numpy

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, mu, sigma = utils.Z_normalize(D)

print("Single-fold")
prints = ["Full Covariance", "Diag Covariance (Naive Bayes)", "Tied-Full Covariance", "Tied-Diag Covariance"]
mvg_params = ["full", "diag", "tied-full", "tied-diag"]
dims_to_reduce = 5

for type in range(len(mvg_params)):
    print()
    print(prints[type])
    for i in range(dims_to_reduce):
        print()
        if (i == 0):
            print("No PCA")
            DZ_reduced = DZ
        else:
            print("PCA with m =", DZ.shape[0]-i)
            DZ_reduced = PCA(DZ, DZ.shape[0]-i)

        (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
        
        gaussian_classifier = MVG(DTR_SF, LTR_SF, cov=mvg_params[type])

        for i in range(len(priors)):
            min_dcf = DCF.min_DCF(gaussian_classifier.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1) 
            print("min DCF MVG", prints[type], "with prior=%.1f:  %.3f" %(priors[i], min_dcf))


k = 5 
print("K-fold with K=", k)

for type in range(len(mvg_params)):
    print("")
    print(prints[type])
    for i in range(5):
        print("")
        if (i == 0):
            print("No PCA")
            DZ_reduced = DZ
        else:
            print("PCA with m =", DZ.shape[0]-i)
            DZ_reduced = PCA(DZ, DZ.shape[0]-i)
            
        trs, ltrs, es, les = utils.Kfold(DZ_reduced, L, k)
        k_min_dcf = numpy.zeros((3, 1))

        for fold in range(k):

            gaussian_classifier = MVG(trs[fold], ltrs[fold], cov=mvg_params[type])
            
            for p in range(len(priors)):
                min_dcf = DCF.min_DCF(gaussian_classifier.predict(es[fold]), les[fold], priors[p], 1, 1) 
                k_min_dcf[p] += min_dcf

        k_min_dcf /= k

        print("Min DCF MVG", prints[type], "with prior=0.5", k_min_dcf[0])
        print("Min DCF MVG", prints[type], "with prior=0.9", k_min_dcf[1])
        print("Min DCF MVG", prints[type], "with prior=0.1", k_min_dcf[2])