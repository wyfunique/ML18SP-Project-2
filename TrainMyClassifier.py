import numpy as np
from MySVM import TrainMySVM, TestMySVM
from MyRVM import TrainMyRVM, TestMyRVM
from MyGPC import GPC
from MyGPCTest import Test

def TrainMyClassifier(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters):
    VecNum = None
    if Parameters['type'] == 'RVM':
        Yvalidate, EstParameters, VecNum = TrainMyRVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters)
    if Parameters['type'] == 'SVM':
        Yvalidate, EstParameters, VecNum = TrainMySVM(XEstimate, XValidate, ClassLabelsEstimate, ClassLabelsValidate, Parameters)
    if Parameters['type'] == 'GPC':
        Yvalidate, EstParameters = GPC(XEstimate, ClassLabelsEstimate, XValidate, ClassLabelsValidate, Parameters)
    return Yvalidate, EstParameters, VecNum
     
