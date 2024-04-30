import matplotlib.pyplot as plt
import utils
import numpy

def hist(D, L, feature_idx, feature_desc):
    plt.hist(D[feature_idx, L == 0], color="#FF0900", ec="#AD0900", density=True, alpha=0.5)
    plt.hist(D[feature_idx, L == 1], color="#00C507", ec="#006507", density=True, alpha=0.5)
    plt.legend(["Not pulsars", "Pulsars"])
    plt.savefig("plot/" + feature_desc, bbox_inches='tight')
    plt.show()

def plot_features(data, labels):
    feature_descriptions = [
        "Mean of the integrated profile",
        "Standard deviation of the integrated profile",
        "Excess kurtosis of the integrated profile",
        "Skewness of the integrated profile",
        "Mean of the DM-SNR curve",
        "Standard deviation of the DM-SNR curve",
        "Excess kurtosis of the DM-SNR curve",
        "Skewness of the DM-SNR curve",
    ]
    for i in range(len(feature_descriptions)):
        hist(data, labels, i, feature_descriptions[i])
    plt.show()

def heatmap(data, labels):
    class_0, class_1 = utils.split_by_class(data, labels)

    # Covariance
    plt.imshow(numpy.corrcoef(data), cmap='Greys', interpolation='nearest')
    plt.savefig("plot/heatmap_whole", bbox_inches='tight')
    plt.show()

    plt.imshow(numpy.corrcoef(class_0), cmap='Reds', interpolation='nearest')
    plt.savefig("plot/heatmap_false", bbox_inches='tight')
    plt.show()
    
    plt.imshow(numpy.corrcoef(class_1), cmap='Greens', interpolation='nearest')
    plt.savefig("plot/heatmap_true", bbox_inches='tight')
    plt.show()

def plotDCF(x, y, xlabel, m=None, fn="_"):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='minDCF with prior=0.5', color='darkgreen')
    plt.plot(x, y[len(x): 2*len(x)], label='minDCF with prior=0.9', color='darkred')
    plt.plot(x, y[2*len(x): 3*len(x)], label='minDCF with prior=0.1', color='orange')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["minDCF with prior=0.5", "minDCF with prior=0.9", "minDCF with prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("minDCF")
    if m is None:
        plt.savefig("plot/"+ fn + "_" + xlabel + "_estimation_noPCA.png")
    else:
        plt.savefig("plot/"+ fn + "_" + xlabel + "_estimation_PCA" + repr(m) + ".png")

def plotDCFpoly(x, y, xlabel, m=None, fn="_"):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - c=0', color='midnightblue')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - c=1', color='violet')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - c=5', color='olivedrab')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 - c=10', color='orange')

    
    plt.xlim([1e-5, 1e-1])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - c=0", "min DCF prior=0.5 - c=1", 
                'min DCF prior=0.5 - c=5', 'min DCF prior=0.5 - c=10'])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")

    if m is None:
        plt.savefig("plot/"+ fn + "_" + xlabel + "_estimation_noPCA.png")
    else:
        plt.savefig("plot/"+ fn + "_" + xlabel + "_estimation_PCA" + repr(m) + ".png")

    return

def plotDCF_GMM(x, y, xlabel, fn=""):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='darkgreen')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='darkred')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='orange')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=2)
    plt.legend(["minDCF with prior=0.5", "minDCF with prior=0.9", "minDCF with prior=0.1"])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig("plot/"+ fn + "_" + xlabel + "_estimation.png")

    return


def scatter(i, j, xlabel, ylabel, D, L, classesNames):
    
    plt.scatter(D[i, L == 0], D[j, L == 0], color="darkred", s=10, label=classesNames[0])
    plt.scatter(D[i, L == 1], D[j, L == 1], color="darkgreen", s=10, label=classesNames[1])
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    return


def bayesErrorPlot(dcf, mindcf, effPriorLogOdds, model_name):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model_name + " - act DCF", model_name +" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")

    plt.savefig("plot/"+ "_" + model_name + "_BAYES_ERRORPLOT.png")
    return

def plot_ROC(fpr, tpr, fpr2, tpr2, fpr3, tpr3):
# Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(fpr, tpr, linewidth=2, color='darkred')
    plt.plot(fpr2, tpr2, linewidth=2, color='darkgreen')
    plt.plot(fpr3, tpr3, linewidth=2, color='orange')
    plt.legend(["Tied Full Cov", "LogReg", "GMM 16 Gau"])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig("plot/roc.png")
