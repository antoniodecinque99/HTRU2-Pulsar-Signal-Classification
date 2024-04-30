import numpy
import GMM_utils

class GMM():
    def __init__(self, D, L, components, option="full"):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        self.option = option

        if (option == "tied"):
            sigma0 =  numpy.cov(D0).reshape((D0.shape[0], D0.shape[0]))
            sigma1 =  numpy.cov(D1).reshape((D1.shape[0], D1.shape[0]))
        
            self.sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*sigma0+D[:, L == 1].shape[1]*sigma1)

            GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), GMM_utils.cov_constraint(self.sigma))]
            GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), GMM_utils.cov_constraint(self.sigma))]
        else:
            GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), GMM_utils.cov_constraint(
                numpy.cov(D0).reshape((D0.shape[0], D0.shape[0]))))]
            GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), GMM_utils.cov_constraint(
                numpy.cov(D1).reshape((D1.shape[0], D1.shape[0]))))]

        self.GMM0 = GMM_utils.LBGalgorithm(GMM0_init, D0, components, option=option)
        self.GMM1 = GMM_utils.LBGalgorithm(GMM1_init, D1, components, option=option)

    def predict(self, X, labels=False):
        if labels:
            PD0 = GMM_utils.compute_posterior_GMM(X, self.GMM0)
            PD1 = GMM_utils.compute_posterior_GMM(X, self.GMM1)

            PD = numpy.vstack((PD0, PD1))
            return numpy.argmax(PD, axis=0)

        LS0 = GMM_utils.computeLogLikelihood(X, self.GMM0)
        LS1 = GMM_utils.computeLogLikelihood(X, self.GMM1)

        llr = LS1-LS0
        return llr
