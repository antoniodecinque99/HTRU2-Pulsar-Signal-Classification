import numpy
import utils


def detection_cost_function(conf_matrix, pi1, C_fn, C_fp):
    FNR = conf_matrix[0][1]/(conf_matrix[0][1]+conf_matrix[1][1])
    FPR = conf_matrix[1][0]/(conf_matrix[0][0]+conf_matrix[1][0])

    return (pi1*C_fn*FNR + (1-pi1)*C_fp*FPR)


def norm_DCF(DCF, pi1, C_fn, C_fp):
    dummy = numpy.array([pi1*C_fn, (1-pi1)*C_fp])
    index = numpy.argmin(dummy)
    return DCF/dummy[index]


def min_DCF(llr, LTE, pi1, C_fn, C_fp):
    llr_sorted = numpy.sort(llr)
    normalized_DCF = []

    for t in llr_sorted:
        predictions = (llr > t).astype(int)
        conf_matrix = utils.confusion_matrix(predictions, LTE, LTE.max()+1)
        this_dcf = detection_cost_function(conf_matrix, pi1, C_fn, C_fp)
        normalized_DCF.append(norm_DCF(this_dcf, pi1, C_fn, C_fp))

    min = numpy.argmin(normalized_DCF)

    return normalized_DCF[min]


def actual_DCF(llr, LTE, pi1, cfn, cfp):
    
    predictions = (llr > (-numpy.log(pi1/(1-pi1)))).astype(int)
    
    confMatrix =  utils.confusion_matrix(predictions, LTE, LTE.max()+1)
    uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
    NDCF=(norm_DCF(uDCF, pi1, cfn, cfp))
        
    return NDCF
