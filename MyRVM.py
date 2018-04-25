import scipy.io as sio
import numpy as np
from skrvm import RVC

def TrainMyRVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters=None):


    training_labels = np.int8(np.zeros(ClassLabelsEstimate.shape[0]))
    validate_labels = np.int8(np.zeros(ClassLabelsValidate.shape[0]))
    for i in range(ClassLabelsEstimate.shape[0]):
        training_labels[i] = np.where(ClassLabelsEstimate[i] == 1)[0]
    for i in range(ClassLabelsValidate.shape[0]):
        validate_labels[i] = np.where(ClassLabelsValidate[i]==1)[0]

    #get 4000 samples of training data
    #this will get the indices and will give the data and labels the same indices
    idx_training = np.random.choice(np.arange(len(training_labels)), 4000, replace=False)
    training_labels_sampled = training_labels[idx_training]
    XEstimate_sampled = XEstimate[idx_training]

    #get 1000 samples of training data
    #this will get the indices and will give the data and labels the same indices
    idx_validate = np.random.choice(np.arange(len(validate_labels)), 1000, replace=False)
    XValidate_sampled = XValidate[idx_validate]
    ClassLabelsValidate_sampled = ClassLabelsValidate[idx_validate]

    #initialize RVM with classification (RVC class)
    rvm = RVC(kernel='rbf', n_iter = 1, alpha=1.e-6, beta=1.e-6, verbose=True)
    #fit RVM
    rvm.fit(XEstimate_sampled, training_labels_sampled)
    #predict and return an array of classes for each input
    Yvalidate = rvm.predict(XValidate_sampled)
    EstParameters = rvm
    Rvectors = 1
    return Yvalidate, EstParameters, Rvectors, ClassLabelsValidate_sampled, idx2

def TestMyRVM(XTest, EstParameters,Parameters=None):
    #Est Parameters is the rvm from the training with its parameters
    rvm = EstParameters
    Ytest = rvm.predict_proba(XTest)
    ClassLabels = rvm.predict(XTest)
    return Ytest, ClassLabels
