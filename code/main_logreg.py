from logreg import LogisticRegression
import utils
from PCA import PCA
import numpy
import DCF

priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, mean, sigma = utils.Z_normalize(D)
l = 0.0001
k = 5
dims_to_reduce = 3 # PCA index

# single fold
for i in range(dims_to_reduce): 
    print("Logreg single-fold on Z-normalized features, lambda =", l, "PCA=", DZ.shape[0]-i)
    if i==0:
        DZ_reduced = DZ
    else:
        DZ_reduced = PCA(DZ, DZ.shape[0]-i)

    # single-fold
    (DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)

    for pi_T in range(len(priors)):
        lr = LogisticRegression(DTR_SF, LTR_SF, l, prior=priors[pi_T])
        for p in range(len(priors)):
            min_dcf = DCF.min_DCF(lr.predict(DEVAL_SF), LEVAL_SF, priors[p], 1, 1) 
            print("prior:", priors[p], "; pi_T for model:", priors[pi_T], "; DCF:", min_dcf)


for i in range(dims_to_reduce):
    print("Logreg with " + repr(k) + "-fold on Z-normalized features, lambda =", l, "PCA=", DZ.shape[0]-i)
    if i==0:
        DZ_reduced = DZ
    else:
        DZ_reduced = PCA(DZ, DZ.shape[0]-i)

    # k-fold
    trs, ltrs, es, les = utils.Kfold(DZ_reduced, L, k)
    k_min_dcf = numpy.zeros((3, 3))
    for fold in range(k):
        for pi_T in range(len(priors)):
            
            lr = LogisticRegression(trs[fold], ltrs[fold], l, prior=priors[pi_T])

            for p in range(len(priors)):
                min_dcf = DCF.min_DCF(lr.predict(es[fold]), les[fold], priors[p], 1, 1) 
                print("fold:", fold, "; prior:", priors[p], "; pi_T for model:", priors[pi_T], "; DCF:", min_dcf)
                k_min_dcf[pi_T, p] += min_dcf

    k_min_dcf /= k

    print(k_min_dcf)
