import numpy
import matplotlib.pyplot as plt
import utils
from PCA import PCA
from logreg import LogisticRegression
from GMM import GMM
from MVG import MVG
from SVM import SVM
import DCF
import plot


def calibrate_scores(s, L, lambda_, prior=0.5):
    s = utils.vrow(s)
    logreg_ = LogisticRegression(s, L, lambda_, prior=prior)

    alpha = logreg_.x[0]
    betafirst = logreg_.x[1]
    calibrated_scores = alpha*s+betafirst-numpy.log(prior/(1-prior))
    return calibrated_scores


priors = [0.5, 0.9, 0.1]
D, L = utils.load('data/Train.txt')
DZ, _, _ = utils.Z_normalize(D)

DZ_reduced = PCA(DZ, 7)

eval_points = 10

effPriorLogOdds = numpy.linspace(-3, 3, eval_points)
effPriors = 1/(1+numpy.exp(-1*effPriorLogOdds))

print("")
print("MVG Tied-full cov UNCALIBRATED")
(DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
gaussian_classifier = MVG(DTR_SF, LTR_SF, cov="tied-full")

for i in range(len(priors)):
    act_dcf = DCF.actual_DCF(gaussian_classifier.predict(
        DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
    print("act DCF MVG Tied-full cov with prior=%.1f:  %.3f" %
          (priors[i], act_dcf))

# MVG TIED-FULL BAYES PLOT UNCALIBRATED

actualDCF_array = []
minDCF_array = []
for i in range(eval_points):
    actualDCF_array.append(DCF.actual_DCF(
        gaussian_classifier.predict(DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))
    minDCF_array.append(DCF.min_DCF(
        gaussian_classifier.predict(DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))
    print("minDCF = ", minDCF_array[i], "actDCF = ", actualDCF_array[i])

plot.bayesErrorPlot(actualDCF_array, minDCF_array,
                    effPriorLogOdds, "MVG_Tied-Full_(UNCALIBRATED)")

# ---

# MVG TIED-FULL BAYES PLOT CALIBRATED

print("")
print("MVG Tied-full cov CALIBRATED")
(DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
gaussian_classifier = MVG(DTR_SF, LTR_SF, cov="tied-full")

scores = []
actualDCF_array = []
minDCF_array = []
for i in range(eval_points):
    print(i)
    if (i == 0):
        scores.append(gaussian_classifier.predict(DEVAL_SF))
    else:
        numpy.append(scores, gaussian_classifier.predict(DEVAL_SF))

    scores = calibrate_scores(numpy.hstack(scores), numpy.hstack(LEVAL_SF), 1e-4).flatten()

    actualDCF_array.append(DCF.actual_DCF(
        scores, LEVAL_SF, effPriors[i], 1, 1))

    minDCF_array.append(DCF.min_DCF(gaussian_classifier.predict(
        DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))


plot.bayesErrorPlot(actualDCF_array, minDCF_array,
                    effPriorLogOdds, "MVG_Tied-Full_(CALIBRATED)")


print("")
print("Linear LogReg UNCALIBRATED")
(DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)

l = 0.0001
for i in range(len(priors)):
    lr = LogisticRegression(DTR_SF, LTR_SF, l, prior=priors[i])

    act_dcf = DCF.actual_DCF(lr.predict(DEVAL_SF), LEVAL_SF, 0.5, 1, 1)
    print("prior:", priors[i], "; pi_T for model: 0.5", "; DCF:", act_dcf)


# LINEAR LOGISTIC BAYES PLOT UNCALIBRATED

lr = LogisticRegression(DTR_SF, LTR_SF, l)
actualDCF_array = []
minDCF_array = []
for i in range(eval_points):
    actualDCF_array.append(DCF.actual_DCF(
        lr.predict(DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))
    minDCF_array.append(DCF.min_DCF(lr.predict(
        DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))
    print("minDCF = ", minDCF_array[i], "actDCF = ", actualDCF_array[i])

plot.bayesErrorPlot(actualDCF_array, minDCF_array,
                    effPriorLogOdds, "LINEAR_LOGISTIC_(UNCALIBRATED)")

# ---

# Linear Regression BAYES PLOT CALIBRATED

print("")
print("Linear LogReg CALIBRATED")
(DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
lr = LogisticRegression(DTR_SF, LTR_SF, l)

scores = []
actualDCF_array = []
minDCF_array = []
for i in range(eval_points):
    print(i)
    if (i == 0):
        scores.append(lr.predict(DEVAL_SF))
    else:
        numpy.append(scores, lr.predict(DEVAL_SF))

    scores = calibrate_scores(numpy.hstack(scores), numpy.hstack(LEVAL_SF), 1e-4).flatten()

    actualDCF_array.append(DCF.actual_DCF(
        scores, LEVAL_SF, effPriors[i], 1, 1))

    minDCF_array.append(DCF.min_DCF(lr.predict(
        DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))


plot.bayesErrorPlot(actualDCF_array, minDCF_array,
                    effPriorLogOdds, "LINEAR_LOGISTIC_(CALIBRATED)")


print("")
print("GMM Full cov(16 components)")
(DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
n_splits = 4
gmm = GMM(DTR_SF, LTR_SF, n_splits, option="full")

# for i in range(len(priors)):
#     act_DCF = DCF.actual_DCF(gmm.predict(DEVAL_SF), LEVAL_SF, priors[i], 1, 1)
#     print("act DCF GMM", "with prior=%.1f:  %.3f" %(priors[i], act_DCF))

# GMM BAYES PLOT UNCALIBRATED

actualDCF_array = []
minDCF_array = []
for i in range(eval_points):
    actualDCF_array.append(DCF.actual_DCF(
        gmm.predict(DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))
    minDCF_array.append(DCF.min_DCF(gmm.predict(
        DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))
    print("minDCF = ", minDCF_array[i], "actDCF = ", actualDCF_array[i])

plot.bayesErrorPlot(actualDCF_array, minDCF_array,
                    effPriorLogOdds, "GMM_(UNCALIBRATED)")

# ---


print("")
print("Linear LogReg CALIBRATED")
(DTR_SF, LTR_SF), (DEVAL_SF, LEVAL_SF) = utils.split_single_fold(DZ_reduced, L)
gmm = GMM(DTR_SF, LTR_SF, n_splits, option="full")

scores = []
actualDCF_array = []
minDCF_array = []
for i in range(eval_points):
    print(i)
    if (i == 0):
        scores.append(gmm.predict(DEVAL_SF))
    else:
        numpy.append(scores, gmm.predict(DEVAL_SF))

    scores = calibrate_scores(numpy.hstack(scores), numpy.hstack(LEVAL_SF), 1e-4).flatten()

    actualDCF_array.append(DCF.actual_DCF(
        scores, LEVAL_SF, effPriors[i], 1, 1))

    minDCF_array.append(DCF.min_DCF(gmm.predict(
        DEVAL_SF), LEVAL_SF, effPriors[i], 1, 1))


plot.bayesErrorPlot(actualDCF_array, minDCF_array,
                    effPriorLogOdds, "GMM_(CALIBRATED)")

