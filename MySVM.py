import scipy.io as sio
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import normalize

def GetClassLabels(svm_predict_prob):
    
    ClassLabels = np.zeros((svm_predict_prob.shape[0], svm_predict_prob.shape[1]+1))
    for i in range(svm_predict_prob.shape[0]):
        ClassLabels[i, :-1] = svm_predict_prob[i]
        sorted_prob = np.sort(svm_predict_prob[i])[::-1]
        interval = np.subtract(sorted_prob[:-1], sorted_prob[1:])
        ClassLabels[i, -1] = 1.0 - interval[0] / np.sum(interval)
    ClassLabels = normalize(ClassLabels, axis=1)
    return ClassLabels
    
def TrainMySVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters=None):
    # Each label in ClassLabelsEstimate and ClassLabelsValidate is 5 dimensional vector like [1,-1,-1,-1,-1];
    # So ClassLabelsEstimate and ClassLabelsValidate are both N-by-5 numpy arrays.
    train_labels = np.int8(np.zeros(ClassLabelsEstimate.shape[0]))
    cv_labels = np.int8(np.zeros(ClassLabelsValidate.shape[0]))
    for i in range(ClassLabelsEstimate.shape[0]):
        train_labels[i] = np.where(ClassLabelsEstimate[i] == 1)[0]
    for i in range(ClassLabelsValidate.shape[0]):
        cv_labels[i] = np.where(ClassLabelsValidate[i] == 1)[0]
    
    C_range = 10. ** np.arange(-2, 3) # 10 is best
    gamma_range = 10. ** np.arange(-2, 3) # 1 is best
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(max_iter=100, decision_function_shape='ovo', probability=True, verbose=True), param_grid=param_grid, cv=5)
    grid.fit(XEstimate, train_labels)
    svm = grid.best_estimator_
    #svm = SVC(C=10, gamma=1, max_iter=100, decision_function_shape='ovo', probability=True, verbose=True)
    svm.fit(XEstimate, train_labels)
    cv_predict_prob = svm.predict_proba(XValidate)
    #cv_predict = svm.predict(XValidate)
    #cv_acc = sum([1 for i in range(cv_labels.shape[0]) if cv_predict[i] == cv_labels[i]]) / float(cv_labels.shape[0])
    #print svm.score(XValidate, cv_labels)
    #print "==== CV Accuracy: %f ===="%cv_acc
    Yvalidate = GetClassLabels(cv_predict_prob)
    EstParameters = {}
    InternalParams = ['support_', 'support_vectors_', 'n_support_', 'dual_coef_', 'coef_', 
    'intercept_', '_sparse', 'shape_fit_', '_dual_coef_', '_intercept_', 'probA_', 'probB_', '_gamma', 'classes_']
    EstParameters['HyperParameters'] = svm.get_params()
    EstParameters['EstimatedParameters'] = {}
    for p in InternalParams:
        try:
            EstParameters['EstimatedParameters'][p] = eval('svm.%s'%p)
        except: 
            continue
    
    return Yvalidate, EstParameters
    
def TestMySVM(XTest, EstParameters, Parameters=None):
    hyperParams = EstParameters['HyperParameters']
    trainedParams = EstParameters["EstimatedParameters"]
    svm = SVC(C=hyperParams['C'], gamma=hyperParams['gamma'], kernel=hyperParams['kernel'], max_iter=hyperParams['max_iter'], decision_function_shape='ovo', probability=True)
    InternalParams = ['support_', 'support_vectors_', 'n_support_', 'dual_coef_', 'coef_', 
    'intercept_', '_sparse', 'shape_fit_', '_dual_coef_', '_intercept_', 'probA_', 'probB_', '_gamma', 'classes_']
    for p in InternalParams:
        try:
            exec('svm.%s = trainedParams[p]'%p) 
        except: 
            continue
    test_predict_prob = svm.predict_proba(XTest)
    Ytest = GetClassLabels(test_predict_prob)
    return Ytest