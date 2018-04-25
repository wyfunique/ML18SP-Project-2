import numpy as np
from DataAnalysis import DataAnalysis
from ConfusionPlot import plot_confusion_matrix
from TrainMyClassifier import TrainMyClassifier
from TestMyClassifier import TestMyClassifier
from scipy.io import loadmat
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix

def MyCrossValidate(XTrain, ClassLabels, Nf, Parameters):
    
    DP = DataAnalysis(XTrain, ClassLabels, n_cross_validation=Nf)
    dataRaw, labelsRaw = DP.get_dataset()
    
    kf_cv = StratifiedKFold(n_splits=Nf, shuffle=True)
    
    # Convert Classlabel vector to a single scalar indicating which class the sample is in
    def get_scalar(ClassLabels):
        scalar_labels = np.uint(np.zeros(ClassLabels.shape[0]))
        for i in range(scalar_labels.shape[0]):
            scalar_labels[i] = np.argmax(ClassLabels[i])
        return scalar_labels
    
    scalar_labels = get_scalar(ClassLabels)
    Ytrain = []
    EstParametersArray = []
    EstConfMatricesArray = []
    
    # Split estimation and cross validation sets
    for est_idx, cv_idx in kf_cv.split(XTrain, scalar_labels):
        est_data = XTrain[est_idx]
        est_labels = ClassLabels[est_idx]
        cv_data = XTrain[cv_idx]
        cv_labels = ClassLabels[cv_idx]
        
        XEstimate = est_data
        ClassLabelsEstimate = est_labels
        XValidate = cv_data
        ClassLabelsValidate = cv_labels
        Yvalidate, EstParameters, VecNum = TrainMyClassifier(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters)
        '''
        cv_acc = sum([1 for i in range(Yvalidate.shape[0]) if np.argmax(Yvalidate[i]) == np.argmax(ClassLabelsValidate[i])]) / float(ClassLabelsValidate.shape[0])
        print "==== Test Accuracy: %f ====" % (cv_acc)
        print "Vector number: %d" % VecNum
        #print confusion_matrix(get_scalar(ClassLabelsValidate), get_scalar(Yvalidate))
        #plot_confusion_matrix(get_scalar(ClassLabelsValidate), get_scalar(Yvalidate), normalize=True)
        #raw_input()
        '''
        Ytrain.append(Yvalidate)
        EstParametersArray.append(EstParameters)
        EstConfMatricesArray.append(confusion_matrix(get_scalar(ClassLabelsValidate), get_scalar(Yvalidate)))
    
    OverallValidate, EstParameters, VecNum = TrainMyClassifier(XTrain, XTrain, ClassLabels, ClassLabels, Parameters)  
    # Overall confusion matrix
    ConfMatrix = confusion_matrix(get_scalar(ClassLabels), get_scalar(OverallValidate))
    '''
    #print ConfMatrix
    plot_confusion_matrix(get_scalar(ClassLabels), get_scalar(OverallValidate), normalize=True)
    '''
    return Ytrain, EstParametersArray, EstConfMatricesArray, ConfMatrix

