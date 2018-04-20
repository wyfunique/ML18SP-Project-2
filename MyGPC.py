import scipy.io as sio
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def GPC(XEstimate, ClassLabels, XValidate, YValidate, Parameters):	
	N = XEstimate.shape[0] #number of XEstimate data records
	Nv = XValidate.shape[0] #number of XValidate data records
	Nc = ClassLabels.shape[1] #number of classes

	data = [] #training data list, separated w.r.t. class
	#Params = [] #Parameters
	EstParams = []
	Z = [] #used to calculate classLabels for XValidate
	for i in range(Nc):
		data.append([])
		#Params.append([])
		EstParams.append([])
		Z.append([])
		for j in range(Nc):
			#Params[i].append(None)
			EstParams[i].append(None)
			Z[i].append(None)

	#data[i] are all Training Xs that belong to class i
	for i in range(N):
		for j in range(Nc):
			if ClassLabels[i][j] == 1:
				data[j].append(XEstimate[i])
	for i in range(Nc):
		data[i] = np.matrix(data[i])

	#create a GPC for each pair of classes, store ClassLabels on XValidate in 2d array Z
	for i in range(Nc):
		for j in range(i+1, Nc):
			label0 = np.zeros([data[i].shape[0],])
			label1 = np.ones([data[j].shape[0],])
			y = np.hstack([label0, label1])
			X = np.vstack([data[i], data[j]])
			#here use multi-class = 'one_vs_rest' rather than 'one_vs_one', 
			#because the latter one does not support returning probability.
			#We implement one_vs_one method manually here 
			myGPC = GaussianProcessClassifier(kernel=Parameters['kernel'], optimizer=Parameters['optimizer'], 
				n_restarts_optimizer=Parameters['n_restarts_optimizer'], max_iter_predict=Parameters['max_iter_predict'], 
				warm_start=Parameters['warm_start'], copy_X_train=Parameters['copy_X_train'], random_state=Parameters['random_state'], 
				multi_class='one_vs_rest', n_jobs=Parameters['n_jobs'])
			myGPC.fit(X, y)
			#Params[i][j] = myGPC.get_params()
			estparam = {'classes_': myGPC.classes_, 'n_classes_': myGPC.n_classes_, 'base_estimator_': myGPC.base_estimator_}
			EstParams[i][j] = estparam
			Z[i][j] = myGPC.predict_proba(XValidate)
			Z[j][i] = 1 - Z[i][j]
			print('GPC: class ' + str(i) + ' to class ' + str(j) + ' fitted')
		Z[i][i] = np.zeros([XValidate.shape[0], Nc])
		print('GPC for class ' + str(i) + ' all fitted')
	#class labels
	CL = np.zeros([Nv, Nc])
	for i in range(Nv):
		for j in range(Nc):
			for k in range(Nc):
				CL[i][j] += Z[j][k][i][0]
	CL/=(2*Nc)
	#calculate probability that the data belongs to none of those classes
	#the less the deviation is, the more likely it is from another class
	#so use 1/(exp(std)) basically, and since std tend to be quite small, use 10*Nc*std
	addColumn = []
	for i in range(Nv):
		std = np.std(CL[i])
		mean = np.mean(CL[i])
		bounded_std = 1/(np.exp(10*Nc*std))
		#bounded_std = 1/np.exp(10*Nc*(sigmoid(std)-0.5))
		addColumn.append(bounded_std)
	addColumn = np.matrix(addColumn).transpose()
	CL = np.hstack([CL, addColumn])
	#normalize after adding new probability
	for i in range(Nv):
		norm_denominator = 1 + addColumn[i]
		CL[i] /= norm_denominator
	print CL
	rightnum = 0
	anothernum = 0
	for i in range(Nv):
		if np.argmax(CL[i]) == Nc:
			anothernum+=1
			continue
		if YValidate[i][np.argmax(CL[i])] == 1:
			rightnum+=1
	print "accuracy:", float(rightnum)/Nv
	print "anotherrate:", float(anothernum)/Nv

	YValidate = CL
	EstParameters = {'Parameters': Parameters, 'EstParameters': EstParams, 'Nclasses': Nc}
	return YValidate, EstParameters