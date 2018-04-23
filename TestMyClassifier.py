import numpy as np
from MySVM import TrainMySVM, TestMySVM
from MyRVM import TrainMyRVM, TestMyRVM
from MyGPC import TrainMyGPC, TestMyGPC

def TestMyClassifier(XTest, EstParameters, Parameters):
    if Parameters['type'] == 'RVM':
        Ytest = TestMyRVM(XTest, EstParameters, Parameters)
    if Parameters['type'] == 'SVM':
        Ytest = TestMySVM(XTest, EstParameters, Parameters)
    if Parameters['type'] == 'GPC':
        Ytest = TestMyGPC(XEstimate, ClassLabelsEstimate, XValidate, ClassLabelsValidate, Parameters)
    return Ytest