import numpy as np
from DataAnalysis import DataAnalysis
from ConfusionPlot import plot_confusion_matrix

from TrainMyClassifier import TrainMyClassifier
from TestMyClassifier import TestMyClassifier
from MyConfusionMatrix import MyConfusionMatrix

from scipy.io import loadmat
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix


#this function takes in training data, class labels, number of folds, and a parameter Set
#this will first divide the data into NFolds
#Next, it will train the DataAnalysis
#Finally it will produce the confusion matrices
def MyCrossValidate(XTrain, ClassLabels, Nf, Parameters):

    DP = DataAnalysis(XTrain, ClassLabels, n_cross_validation=Nf)
    dataRaw, labelsRaw = DP.get_dataset()
    #split data into k folds
    kf_cv = StratifiedKFold(n_splits=Nf, shuffle=True)
    #get scalar values of class labels
    def get_scalar(ClassLabels):
        scalar_labels = np.uint(np.zeros(ClassLabels.shape[0]))
        for i in range(scalar_labels.shape[0]):
            scalar_labels[i] = np.argmax(ClassLabels[i])
        return scalar_labels

    scalar_labels = get_scalar(ClassLabels)
    Ytrain = []
    #arrays for the output for estimated parameter and estimtated confidence matrices
    EstParametersArray = []
    EstConfMatricesArray = []


    #for each fold perform training and plot the confusion matrices
    for est_idx, cv_idx in kf_cv.split(XTrain, scalar_labels):
        est_data = XTrain[est_idx]
        est_labels = ClassLabels[est_idx]
        cv_data = XTrain[cv_idx]
        cv_labels = ClassLabels[cv_idx]

        XEstimate = est_data
        ClassLabelsEstimate = est_labels
        XValidate = cv_data
        ClassLabelsValidate = cv_labels

        #special case: RVM which outputs idx for the number of samples
        if Parameters['type'] == 'RVM':
            Yvalidate, EstParameters, ClassLabelsValidate, VecNum, idx = TrainMyClassifier(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters)
            #YTest = TestMyClassifier(XTrain, EstParameters, Parameters)
        #calls train my classifier wrapper class
        else:
            Yvalidate, EstParameters, VecNum = TrainMyClassifier(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters)
            #YTest = TestMyClassifier(XTrain, EstParameters, Parameters)

         #finds the test accuracy
        cv_acc = sum([1 for i in range(Yvalidate.shape[0]) if np.argmax(Yvalidate[i]) == np.argmax(ClassLabelsValidate[i])]) / float(ClassLabelsValidate.shape[0])
        print "==== Test Accuracy: %f ====" % (cv_acc)
        print "Vector number: %d" % VecNum

        #append to list of predicted lables
        Ytrain.append(Yvalidate)

        #append to list of estimated parameters
        EstParametersArray.append(EstParameters)
        cnf_matrix, accuracy = MyConfusionMatrix(get_scalar(ClassLabelsValidate), get_scalar(Yvalidate))

        #append to list of confidence matrices
        EstConfMatricesArray.append(cnf_matrix)

    #produces the confusion matrix for the entire training data
    if Parameters['type'] == 'RVM':
        OverallValidate, EstParameters, ClassLabelsValidate, VecNum, idx = TrainMyClassifier(XTrain, XTrain, ClassLabels, ClassLabels, Parameters)
        ClassLabels = ClassLabels[idx]
        ConfMatrix, accuracy = MyConfusionMatrix(get_scalar(ClassLabels), get_scalar(OverallValidate))

    else:
        OverallValidate, EstParameters, VecNum = TrainMyClassifier(XTrain, XTrain, ClassLabels, ClassLabels, Parameters)
        ConfMatrix, accuracy = MyConfusionMatrix(get_scalar(ClassLabels), get_scalar(OverallValidate))

    return Ytrain, EstParametersArray, EstConfMatricesArray, ConfMatrix
