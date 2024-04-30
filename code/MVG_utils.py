import numpy
import scipy.special
import utils


def logpdf_GAU_ND(x, mi, C):
    return -(x.shape[0]/2)*numpy.log(2*numpy.pi)-(0.5)*(numpy.linalg.slogdet(C)[1]) - (0.5)*numpy.multiply((numpy.dot((x-mi).T, numpy.linalg.inv(C))).T, (x-mi)).sum(axis=0)


def joint_log_density(LS, priors):
    return LS + priors


def marginal_log_densities(joint_probability):
    return utils.vrow(scipy.special.logsumexp(joint_probability, axis=0))


def log_posteriors(joint_probability, marginals):
    return joint_probability-marginals
