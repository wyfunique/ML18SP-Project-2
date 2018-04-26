import scipy.io as sio
import numpy as np
from skrvm import RVC
from sklearn.multiclass import OneVsRestClassifier

#input: Estimate Data, Validation Data, Class Label Estimate, Labels for Validate, parameters (default)
#output: Validation Labels, Estimated parameters (returns RVM that is produced), Class labels which were XEstimate_sampled
#        indexes of classes that were sampled
# prints number of relevance vectors to console
def TrainMyRVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters):

    #assign scalar lables to the classes
    training_labels = np.int8(np.zeros(ClassLabelsEstimate.shape[0]))
    validate_labels = np.int8(np.zeros(ClassLabelsValidate.shape[0]))
    for i in range(ClassLabelsEstimate.shape[0]):
        training_labels[i] = np.where(ClassLabelsEstimate[i] == 1)[0]
    for i in range(ClassLabelsValidate.shape[0]):
        validate_labels[i] = np.where(ClassLabelsValidate[i]==1)[0]

    #get 800 samples of training data for the purposes of testing fast
    #this will get the indices and will give the data and labels the same indices

    idx_training = np.random.choice(np.arange(len(training_labels)), 800, replace=False)
    training_labels_sampled = training_labels[idx_training]
    XEstimate_sampled = XEstimate[idx_training]

    #get 200 samples of training data for the purposes of testing fast
    #this will get the indices and will give the data and labels the same indices
    idx_validate = np.random.choice(np.arange(len(validate_labels)), 200, replace=True)
    XValidate_sampled = XValidate[idx_validate]
    ClassLabelsValidate_sampled = ClassLabelsValidate[idx_validate]

    #initialize RVM with classification (RVC class)
    rvc = RVC(n_iter=Parameters['n_iter'])
    rvm = OneVsRestClassifier(rvc)
    #fit RVM
    rvm.fit(XEstimate_sampled, training_labels_sampled)
    #predict and return an array of classes for each input
    Yvalidate = rvm.predict_proba(XValidate_sampled)
    EstParameters = rvm
    NumVectors = rvm.estimators_[-1].relevance_.shape[0]
    return Yvalidate, EstParameters, ClassLabelsValidate_sampled, NumVectors, idx_validate

def TestMyRVM(XTest, EstParameters,Parameters=None):
      #Est Parameters is the rvm from the training with its parameters
      rvm = EstParameters
      #predict RVM
      Ytest = rvm.predict_proba(XTest)
      Ytest_ClassLabels = np.uint(np.zeros(Ytest.shape[0]))
      for i in range(Ytest.shape[0]):
          Ytest_ClassLabels[i] = np.argmax(Ytest[i])
      return Ytest_ClassLabels
