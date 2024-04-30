import numpy
import scipy.optimize
import logreg_utils as lru

class LogisticRegression:

    def __init__(self, *DLL, prior=0.5):
        D = DLL[0]
        L = DLL[1]
        l = DLL[2] # lambda val
        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(lru.logreg_obj, numpy.zeros(
            D.shape[0] + 1), args=(D, L, l, prior), approx_grad=True)
        
    def predict(self, X, labels=False):
        scores = numpy.dot(self.x[0:-1], X) + self.x[-1]
        if labels:
            return (scores>0).astype(int)
        return scores
