import numpy
import utils

def compute_PCA(C, D, m):
    s, U = numpy.linalg.eigh(C)
    # principal components
    P = U[:, ::-1][:, 0:m]
    # projection matrix
    DP = numpy.dot(P.T, D)
    return DP


def PCA(D, m):
    # center data w.r.t mean
    mean = utils.vcol(D.mean(axis=1))
    DC = D - mean
    C = (1/DC.shape[1]) * (numpy.dot(DC, DC.T))
    return compute_PCA(C, D, m)