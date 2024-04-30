import numpy


def rebalance_J_grad(w, b, DTR, LTR, l, prior):
    norm_term = l/2*(numpy.linalg.norm(w)**2)
    sum_class_1 = 0
    sum_class_0 = 0
    for i in range(DTR.shape[1]):
        # pylint: disable=invalid-unary-operand-type
        exp_arg_negative = -numpy.dot(w.T, DTR[:, i]) - b
        exp_arg_negative_flag = False
        exp_arg_positive = numpy.dot(w.T, DTR[:, i]) + b
        exp_arg_positive_flag = False
        if (exp_arg_negative > 709):
            exp_arg_negative_flag = True
        if (exp_arg_positive > 709):
            exp_arg_positive_flag = True
        if LTR[i] == 1:
            if exp_arg_negative_flag:
                sum_class_1 += exp_arg_negative
            else:
                sum_class_1 += numpy.log1p(
                    numpy.exp(-numpy.dot(w.T, DTR[:, i])-b))
        else:
            if exp_arg_positive_flag:
                sum_class_0 += exp_arg_positive
            else:
                sum_class_0 += numpy.log1p(
                    numpy.exp(numpy.dot(w.T, DTR[:, i])+b))
    j = norm_term + (prior/DTR[:, LTR == 1].shape[1])*sum_class_1 + \
        ((1-prior)/DTR[:, LTR == 0].shape[1])*sum_class_0
    return j


def logreg_obj(v, DTR, LTR, l, prior=0.5):
    w, b = v[0:-1], v[-1] # contains w + b (b is integrated in the vector)
    j = rebalance_J_grad(w, b, DTR, LTR, l, prior)
    return j
