import utils
from PCA import PCA
from GMM import GMM
import DCF
import plot

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, mu, sigma = utils.Z_normalize(D)
dims_to_reduce = 2
options = ["full", "tied", "diag"]
splits_array = 5
splots = [2, 4, 8, 16, 32]

for o in range(1, len(options)):
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

        for p in range(len(priors)):
            print()
            print("Working on application with prior:", priors[p])

            for n_splits in range(splits_array):
                gmm = GMM(DZ_reduced, L, n_splits, option=options[o])
                tmp = DCF.min_DCF(gmm.predict(DEVAL_SF), LEVAL_SF, priors[p], 1, 1)
                minDCF_array.append(tmp)
                print("For ", 2**(n_splits+1), " components, minDCF = ", tmp)
        
        if (i == 0):
            plot.plotDCF_GMM(splots, minDCF_array, "N. of components", fn=options[o])
        else:
            plot.plotDCF_GMM(splots, minDCF_array, "N. of components", DZ.shape[0]-i)