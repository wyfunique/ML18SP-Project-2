import scipy.io as sio
import numpy as np
from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC


def TrainMyRVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters=None):
    training_labels = np.int8(np.zeros(ClassLabelsEstimate.shape[0]))
    validate_labels = np.int8(np.zeros(ClassLabelsValidate.shape[0]))
    for i in range(ClassLabelsEstimate.shape[0]):
        training_labels[i] = np.where(ClassLabelsEstimate[i] == 1)[0]
    for i in range(ClassLabelsValidate.shape[0]):
        validate_labels[i] = np.where(ClassLabelsValidate[i]==1)[0]

    #initialize RVM with classification (RVC class)
    rvm = RVC(gamma = 1, kernel = 'rbf')
    #fit RVM
    rvm.fit(XEstimate, training_labels)
    #predict and return an array of classes for each input
    Yvalidate = rvm.predict(XValidate)
    EstParameters = "";
    return Yvalidate, EstParameters, rvm.relevant_vectors_.shape[0]

def TestMyRVM(XTest, EstParameters,Parameters=None):
    test_clf = RVC()
    test_clf.set_params(EstParameters)
    Ytest = predict_proba(XTest)
    ClassLabels = predict(XTest)
    return Ytest, ClassLabels
