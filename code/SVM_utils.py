import scipy.optimize 
import numpy
from itertools import repeat

def LD_obj_function_dual_formulation(alpha, H):
    grad = numpy.dot(H, alpha) - numpy.ones(H.shape[1])
    return ((1/2)*numpy.dot(numpy.dot(alpha.T, H), alpha)-numpy.dot(alpha.T, numpy.ones(H.shape[1])), grad)

def modified_dual_formulation(DTR, LTR, C, K, option='unbalanced', piT=None):    
    row = numpy.zeros(DTR.shape[1])+K
    D = numpy.vstack([DTR, row])
    Gij = numpy.dot(D.T, D)
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    if (option=="unbalanced"):
        b = list(repeat((0, C), DTR.shape[1]))

    elif (option=="balanced"):
        C1 = C*piT/(DTR[:,LTR == 1].shape[1]/DTR.shape[1])
        C0 = C*(1-piT)/(DTR[:,LTR == 0].shape[1]/DTR.shape[1])
    
        b = []
        for i in range(DTR.shape[1]):
            if LTR[i]== 1:
                b.append ((0,C1))
            elif LTR[i]== 0:
                b.append ((0,C0))

    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(LD_obj_function_dual_formulation,
                                    numpy.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return numpy.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)



def kernel_poly(DTR, LTR, K, C, d, c):
    kernel_f = (numpy.dot(DTR.T, DTR)+c)**d+ K**2
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj * kernel_f
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_obj_function_dual_formulation,
                                    numpy.zeros(DTR.shape[1]), args=(Hij,), bounds=b, factr=1.0)
    return x