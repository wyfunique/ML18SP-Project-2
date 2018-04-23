import numpy as np
from DataAnalysis import DataAnalysis
from TrainMyClassifier import TrainMyClassifier
from TestMyClassifier import TestMyClassifier

def MyCrossValidate(XTrain, ClassLabels, Nf, Parameters):
    DP = DataAnalysis(XTrain, ClassLabels, n_cross_validation=5)
    data, labels = DP.get_dataset()
    Ytrain = np.int(np.zeros((Nf, data.shape[1])))
    EstParametersArray = []
    EstConfMatricesArray = []
    for i in range(data.shape[0]-1):
        XEstimate = np.append(data[:i], data[i+1:])
        ClassLabelsEstimate = np.append(labels[:i], labels[i+1:])
        XValidate = data[i]
        ClassLabelsValidate = labels[i]
        Yvalidate, EstParameters, VecNum = TrainMyClassifier(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters)
        Ytrain[i] = Yvalidate
        EstParametersArray.append(EstParameters)
        
        