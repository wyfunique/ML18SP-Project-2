import scipy.io as sio
import numpy as np
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
#from sklearn.preprocessing import normalize
#from MySVM import  TrainMySVM, TestMySVM
from TrainMyClassifier import TrainMyClassifier as MC
from TestMyClassifier import TestMyClassifier as TC
'''
def normalize2D(array):
    if array.dtype != float:
        array = array.astype(float)
    for i in range(array.shape[0]):
        array[i] = np.divide(array[i], np.sum(array[i]))
    return array

def GetClassLabels(svm_predict_prob):
    #print np.sum(svm_predict_prob)
    ClassLabels = np.zeros((svm_predict_prob.shape[0], svm_predict_prob.shape[1]+1))
    for i in range(svm_predict_prob.shape[0]):
        ClassLabels[i, :-1] = svm_predict_prob[i]
        sorted_prob = np.sort(svm_predict_prob[i])[::-1]
        interval = np.subtract(sorted_prob[:-1], sorted_prob[1:])
        ClassLabels[i, -1] = 1.0 - interval[0] / np.sum(interval)
    #ClassLabels = normalize(ClassLabels, axis=1)
    ClassLabels = normalize2D(ClassLabels)
    #print ClassLabels
    #print np.sum(ClassLabels[0])
    #raw_input()
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
'''
"""
    C_range = 10. ** np.arange(-2, 3) # 10 is best
    gamma_range = 10. ** np.arange(-2, 3) # 1 is best
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(max_iter=100, decision_function_shape='ovo', probability=True, verbose=True), param_grid=param_grid, cv=5)
    grid.fit(XEstimate, train_labels)
    svm = grid.best_estimator_
"""   
'''
    svm = SVC(C=10, gamma=1, max_iter=100, decision_function_shape='ovo', probability=True, verbose=True)
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
'''    
if __name__ == '__main__':
    data = sio.loadmat('Proj2FeatVecsSet1.mat')
    data = data['Proj2FeatVecsSet1']
    labels = sio.loadmat('Proj2TargetOutputsSet1.mat')
    labels = labels['Proj2TargetOutputsSet1']
    scalar_labels = np.uint(np.zeros(labels.shape[0]))
    for i in range(scalar_labels.shape[0]):
        scalar_labels[i] = np.where(labels[i] == 1)[0]
    #print labels
    #print sum([1 for i in range(20000, 25000) if np.where(labels[i]==1)[0]==4])
    #raw_input()
    '''
    class_num = 5
    #test_data = data[20000:]
    #test_labels = labels[20000:]
    data_comb = np.zeros((data.shape[0], data.shape[1]+class_num))
    #print data_comb.shape
    #raw_input()
    data_comb[:, :data.shape[1]] = data
    data_comb[:, data.shape[1]:] = labels
    '''
    #print sum([1 for i in range(20000) if np.where(data_comb[i, data[:20000].shape[1]:]==1)[0]==4])
    #raw_input()
    
    #print data_comb[:, data[:20000].shape[1]:].shape
    #print labels[0]
    #print labels[:2, :class_num-5].shape
    #print data_comb
    #print data_comb.shape
    #input() 
    #print data.shape 
    #print labels.shape
    class_num = 5
    nFold = 5
    C = 1
    gamma = 1 # C=1 & gamma=1 has the best performance
    #max_iter = 500
    param = {'type':'GPC'}
    kf_cv = StratifiedKFold(n_splits=nFold, shuffle=True)
    kf_test = StratifiedKFold(n_splits=nFold, shuffle=True)
    i = 0
    
    for train_idx, test_idx in kf_test.split(data, scalar_labels):
        #print cv_idx
        i += 1
        print "------- Fold-%d -------" % i
        '''
        train_data_labels = data_comb[train_idx]
        train_data = train_data_labels[:, :data.shape[1]]
        train_labels = np.int8(np.round(train_data_labels[:, data.shape[1]:])) # Each label is like [1, -1, -1, -1, -1]
        #print train_labels.shape
        #raw_input()
        cv_data_labels = data_comb[cv_idx]
        cv_data = cv_data_labels[:, :data.shape[1]]
        cv_labels = np.int8(np.round(cv_data_labels[:, data.shape[1]:])) # Each label is like [1, -1, -1, -1, -1]
        '''
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        train_scalar_labels = scalar_labels[train_idx]
        test_data = data[test_idx]
        test_labels = labels[test_idx]
        
        for est_idx, cv_idx in kf_cv.split(train_data, train_scalar_labels):
            
            est_data = train_data[est_idx]
            est_labels = train_labels[est_idx]
            cv_data = train_data[cv_idx]
            cv_labels = train_labels[cv_idx]
            
            print param['type']
            Yvalidate, EstParameters, vecNum = MC(est_data, cv_data, est_labels, cv_labels, param)
            print 'Vector Number:%d'%vecNum
            test_predict = TC(test_data, EstParameters, param) # Each label is like [0.1, 0.2, 0.31, 0.1, 0.13, 0.16]
            correct_predict = 0
            out_any_class = 0
            for i in range(test_predict.shape[0]):
            #print test_predict[i], np.argmax(test_predict[i])
            #print cv_labels[i], np.argmax(cv_labels[i])
            #raw_input()
            #print test_predict[i]
            #raw_input()
                if np.argmax(test_predict[i]) == np.argmax(test_labels[i]):
                    correct_predict += 1
                if np.argmax(test_predict[i]) == class_num:
                    out_any_class += 1
            test_acc = correct_predict / float(test_predict.shape[0])
            out_percentage = out_any_class / float(test_predict.shape[0])
        #test_acc = sum([1 for i in range(test_predict.shape[0]) if np.argmax(test_predict[i]) == np.argmax(cv_labels[i])]) / float(cv_labels.shape[0])
            print "==== Test Accuracy: %f Samples out of classes: %d(%f) ====" % (test_acc, out_any_class, out_percentage)
            raw_input()