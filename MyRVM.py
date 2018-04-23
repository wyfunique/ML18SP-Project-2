import scipy.io as sio
import numpy as np
from skrvm import RVR
from skrvm import RVC

def TrainMyRVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters=None):
    training_labels = np.int8(np.zeros(ClassLabelsEstimate.shape[0]))
    validate_labels = np.int8(np.zeros(ClassLabelsValidate.shape[0]))
    for i in range(ClassLabelsEstimate.shape[0]):
        training_labels[i] = np.where(ClassLabelsEstimate[i] == 1)[0]
    for i in range(ClassLabelsValidate.shape[0]):
        validate_labels[i] = np.where(ClassLabelsValidate[i]==1)[0]

    #initialize RVM with classification (RVC class)
    clf = RVC()
    #fit RVM
    clf.fit(XEstimate, training_labels)
    #predict and return an array of classes for each input
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params();
    return Yvalidate, EstParameters, clf.relevance_.shape[0]

def TestMyRVM(XTest, EstParameters,Parameters=None):
    test_clf = RVC()
    test_clf.set_params(EstParameters)
    Ytest = predict_proba(XTest)
    ClassLabels = predict(XTest)
    return Ytest, ClassLabels
