import numpy
import SVM_utils


class SVM ():
    def __init__(self, DTR, LTR, option1, option2='unbalanced', c=0, d=2, gamma=1.0, C=1.0, K=1.0, piT=0.5):
        self.option1 = option1
        self.DTR = DTR
        self.LTR = LTR
        self.K = K
        self.C = C

        if (option1 == 'linear'):
            if (option2 == 'unbalanced'):
                self.w = SVM_utils.modified_dual_formulation(DTR, LTR, self.C, self.K, option="unbalanced")
            if (option2 == 'balanced'):
                self.w = SVM_utils.modified_dual_formulation(DTR, LTR, self.C, self.K, option="balanced", piT=piT)
        elif (option1 == 'polynomial'):
            self.c = c
            self.d = d
            self.x = SVM_utils.kernel_poly(DTR, LTR, K, C, d, c)
        elif (option1 == 'RBF'):
            self.gamma = gamma
            self.x = SVM_utils.kernel_RBF(DTR, LTR, gamma, K, C)


    def predict(self, DTE, labels=False):
        if (self.option1 == 'linear'):
            DTE = numpy.vstack([DTE, numpy.zeros(DTE.shape[1])+self.K])
            S = numpy.dot(self.w.T, DTE)
        elif (self.option1 == 'polynomial'):
            S = numpy.sum(
                numpy.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (numpy.dot(self.DTR.T, DTE)+self.c)**self.d + self.K), axis=0)

        if labels:
            LP = 1*(S > 0)
            LP[LP == 0] = -1
            return LP
        else:
            return S

