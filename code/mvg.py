import numpy
import utils
import MVG_utils
import numpy.matlib 


class MVG:

    def __init__(self, *DL, cov="full"):
        if cov != "full" and cov !="diag" and cov != "tied-full" and cov != "tied-diag":
            return

        self.cov = cov
        D = DL[0]
        L = DL[1]

        self.mean0 = utils.vcol(D[:, L == 0].mean(axis=1))
        self.mean1 = utils.vcol(D[:, L == 1].mean(axis=1))

        I=numpy.matlib.identity(D.shape[0])
        
        if self.cov == "diag":
            self.sigma0 =   numpy.multiply(numpy.cov(D[:, L == 0]),I)
            self.sigma1 =   numpy.multiply(numpy.cov(D[:, L == 1]),I)
        elif self.cov == "tied-full":
            self.sigma0 = numpy.cov(D[:, L == 0])
            self.sigma1 = numpy.cov(D[:, L == 1])
            self.sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*self.sigma0+D[:, L == 1].shape[1]*self.sigma1)
            self.sigma0 = self.sigma
            self.sigma1 = self.sigma
        elif self.cov == "tied-diag":
            self.sigma0 = numpy.cov(D[:, L == 0])
            self.sigma1 = numpy.cov(D[:, L == 1])
            self.sigma = numpy.multiply(1/(D.shape[1])*(D[:, L == 0].shape[1]*self.sigma0+D[:, L == 1].shape[1]*self.sigma1),I)
            self.sigma0 = self.sigma
            self.sigma1 = self.sigma
        else: # full
            self.sigma0 = numpy.cov(D[:, L == 0])
            self.sigma1 = numpy.cov(D[:, L == 1])
        
        # class priors
        self.pi0 = D[:, L == 0].shape[1]/D.shape[1]
        self.pi1 = D[:, L == 1].shape[1]/D.shape[1]


    def predict(self, X, labels=False):
        if self.cov == "diag" or self.cov == "tied-diag":
            LS0 = numpy.asarray(MVG_utils.logpdf_GAU_ND(X, self.mean0, self.sigma0 )).flatten()
            LS1 = numpy.asarray(MVG_utils.logpdf_GAU_ND(X, self.mean1, self.sigma1 )).flatten()
        else:
            LS0 = MVG_utils.logpdf_GAU_ND(X, self.mean0, self.sigma0)
            LS1 = MVG_utils.logpdf_GAU_ND(X, self.mean1, self.sigma1)
        
        if labels:
            LS = numpy.vstack((LS0, LS1))
            LS_joint = MVG_utils.joint_log_density(
                LS, utils.vcol(numpy.array([numpy.log(self.pi0), numpy.log(self.pi1)])))
            MLD = MVG_utils.marginal_log_densities(LS_joint)
            LP = MVG_utils.log_posteriors(LS_joint, MLD)
            predictions = numpy.argmax(LP, axis=0)
            
            return predictions
        
        llr = LS1-LS0
        return llr
