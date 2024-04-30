import numpy
import scipy
import MVG_utils
import utils


def cov_constraint(cov, psi=0.01):
    U, s, _ = numpy.linalg.svd(cov)
    s[s < psi] = psi
    cov = numpy.dot(U, utils.vcol(s)*U.T)
    return cov


def split(GMM, alpha=0.1):
    size = len(GMM)
    GMM_components = []
    for i in range(size):
        U, s, _ = numpy.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        GMM_components.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        GMM_components.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    return GMM_components


def Estep(logdens, S):
    # E-step: compute the POSTERIOR PROBABILITY (=responsibilities) for each component of the GMM
    # for each sample, using the previous estimate of the model parameters.
    return numpy.exp(S-logdens.reshape(1, logdens.size))


def Mstep(X, S, posterior, option="full"):
    Z_g = numpy.sum(posterior, axis=1)
    F_g = numpy.zeros((X.shape[0], S.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        F_g[:, g] = tempSum
    S_g = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))

    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * numpy.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        S_g[g] = tempSum
    mu = F_g / Z_g
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))

    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)),
                              mu[:, g].reshape((1, X.shape[0])))
    cov = S_g / Z_g.reshape((Z_g.size, 1, 1)) - prodmu

    if (option == "full"):
        for g in range(S.shape[0]):
            cov[g] = cov_constraint(cov[g])

    elif option == "diag":
        for g in range(S.shape[0]):
            cov[g] = cov_constraint(cov[g] * numpy.eye(cov[g].shape[0]))

    elif option == "tied":
        tsum = numpy.zeros((cov.shape[1], cov.shape[2]))
        for g in range(S.shape[0]):
            tsum += Z_g[g]*cov[g]
        for g in range(S.shape[0]):
            cov[g] = cov_constraint(1/X.shape[1] * tsum)

    w = Z_g/numpy.sum(Z_g)
    return (w, mu, cov)


def EMalgorithm(X, gmm, delta=10**(-6), option="full"):
    flag = True
    while(flag):
        S = joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)
        logmarg = marginal_density_GMM(
            joint_log_density_GMM(logpdf_GMM(X, gmm), gmm))
        loglikelihood1 = log_likelihood_GMM(logmarg, X)
        posterior = Estep(logmarg, S)
        (w, mu, cov) = Mstep(X, S, posterior, option=option)
        for g in range(len(gmm)):
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        logmarg = marginal_density_GMM(
            joint_log_density_GMM(logpdf_GMM(X, gmm), gmm))
        loglikelihood2 = log_likelihood_GMM(logmarg, X)
        if (loglikelihood2-loglikelihood1 < delta):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("log likelihood decreasing -- incorrect implementation")
    return gmm


def LBGalgorithm(GMM, X, iterations, option="full"):
    GMM = EMalgorithm(X, GMM, option=option)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM, option=option)
    return GMM


def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        S[i, :] = MVG_utils.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return S


def joint_log_density_GMM(S, gmm):
    for i in range(len(gmm)):
        S[i, :] += numpy.log(gmm[i][0])
    return S


def marginal_density_GMM(S):
    return scipy.special.logsumexp(S, axis=0)


def log_likelihood_GMM(logmarg, X):
    return numpy.sum(logmarg)/X.shape[1]


def compute_posterior_GMM(X, gmm):
    return marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm))


def computeLogLikelihood(X, gmm):
    tempSum = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        tempSum[i, :] = numpy.log(
            gmm[i][0])+MVG_utils.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return scipy.special.logsumexp(tempSum, axis=0)
