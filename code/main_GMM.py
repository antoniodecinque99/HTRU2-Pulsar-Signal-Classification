import utils
from GMM import GMM
from PCA import PCA
import DCF
import numpy

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, mu, sigma = utils.Z_normalize(D)

dims_to_reduce = 2
options = ["full", "tied", "diag"]
n_splits = 4
print("OPTION: full")
for i in range(dims_to_reduce):
    print()
    if (i == 0):
        print("No PCA")
        DZ_reduced = DZ
    else:
        print("PCA with m =", DZ.shape[0]-i)
        DZ_reduced = PCA(DZ, DZ.shape[0]-i)

    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
    gmm = GMM(DTR_SF, LTR_SF, n_splits, option="full")

    for i in range(len(priors)):
        min_dcf = DCF.min_DCF(gmm.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1) 
        print("GMM minDCF with prior=%.1f:  %.3f" %(priors[i], min_dcf))
