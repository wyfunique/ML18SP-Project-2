import numpy as np
from DataAnalysis import DataAnalysis
from TrainMyClassifier import TrainMyClassifier
from TestMyClassifier import TestMyClassifier
from scipy.io import loadmat
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix

def MyCrossValidate(XTrain, ClassLabels, Nf, Parameters):
    
    DP = DataAnalysis(XTrain, ClassLabels, n_cross_validation=Nf)
    dataRaw, labelsRaw = DP.get_dataset()
    
    kf_cv = StratifiedKFold(n_splits=Nf, shuffle=True)
    
    def get_scalar(ClassLabels):
        scalar_labels = np.uint(np.zeros(ClassLabels.shape[0]))
        for i in range(scalar_labels.shape[0]):
            scalar_labels[i] = np.argmax(ClassLabels[i])
        return scalar_labels
    
    scalar_labels = get_scalar(ClassLabels)
    Ytrain = []
    EstParametersArray = []
    EstConfMatricesArray = []
    
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
        
        #cv_acc = sum([1 for i in range(Yvalidate.shape[0]) if np.argmax(Yvalidate[i]) == np.argmax(ClassLabelsValidate[i])]) / float(ClassLabelsValidate.shape[0])
        #print "==== Test Accuracy: %f ====" % (cv_acc)
        #raw_input()
    
        Ytrain.append(Yvalidate)
        EstParametersArray.append(EstParameters)
        EstConfMatricesArray.append(confusion_matrix(get_scalar(ClassLabelsValidate), get_scalar(Yvalidate)))
    
    OverallValidate, EstParameters, VecNum = TrainMyClassifier(XTrain, XTrain, ClassLabels, ClassLabels, Parameters)    
    ConfMatrix = confusion_matrix(get_scalar(ClassLabels), get_scalar(OverallValidate))
    return Ytrain, EstParametersArray, EstConfMatricesArray, ConfMatrix

'''
if __name__ == '__main__':
    input_dict = loadmat("Proj2FeatVecsSet1.mat")
    input_vec = input_dict['Proj2FeatVecsSet1']
    
    # Read label file in .mat format
    # Warning: Files must match format given in project description
    label_dict = loadmat("Proj2TargetOutputsSet1.mat")
    label_vec = label_dict['Proj2TargetOutputsSet1']

    #DP = DataAnalysis(input_vec, label_vec, n_cross_validation=5)
    # Get processed data
    #data, labels = DP.get_dataset()

    v1, v2, v3, v4 = MyCrossValidate(input_vec, label_vec, 5, {'type':'SVM'})
    #print v1[0].shape
    #raw_input()
    #print v2
    #raw_input()
    #print len(v2)
    #raw_input()
    #print v3[0]
    #raw_input()
    #print v4 
'''