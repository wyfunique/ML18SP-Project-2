import scipy.io as sio
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
def Test(XTest, YValidate, EstParameters, Parameters = None):
	Parameters = EstParameters['Parameters']
	EstParams = EstParameters['EstParameters']
	Nc = EstParameters['Nclasses']
	Nt = XTest.shape[0]

	#used to calculate classLabels for XTest
	Z = []
	for i in range(Nc):
		Z.append([])
		for j in range(Nc):
			Z[i].append(None)

	for i in range(Nc):
		for j in range(i+1, Nc):
			myGPC = GaussianProcessClassifier(
				kernel=Parameters['kernel'], 
				optimizer=Parameters['optimizer'], 
				n_restarts_optimizer=Parameters['n_restarts_optimizer'], 
				max_iter_predict=Parameters['max_iter_predict'], 
				warm_start=Parameters['warm_start'], 
				copy_X_train=Parameters['copy_X_train'], 
				random_state=Parameters['random_state'], 
				multi_class='one_vs_rest', 
				n_jobs=Parameters['n_jobs'])
			#myGPC.set_params(Parameters[i][j])
			myGPC.classes_ = EstParams[i][j]['classes_']
			myGPC.n_classes_ = EstParams[i][j]['n_classes_']
			myGPC.base_estimator_ = EstParams[i][j]['base_estimator_']
			Z[i][j] = myGPC.predict_proba(XTest)
			Z[j][i] = 1 - Z[i][j]
			print('GPC: class ' + str(i) + ' to class ' + str(j) + ' tested')
		Z[i][i] = np.zeros([Nt, Nc])
		print('GPC for class ' + str(i) + ' all tested')

	#class labels
	CL = np.zeros([Nt, Nc])
	for i in range(Nt):
		for j in range(Nc):
			for k in range(Nc):
				CL[i][j] += Z[j][k][i][0]
	CL/=(2*Nc)

	#calculate probability that the data belongs to none of those classes
	#the less the deviation is, the more likely it is from another class
	#so use 1/(exp(std)) basically, and since std tend to be quite small, use 10*Nc*std
	addColumn = []
	for i in range(Nt):
		std = np.std(CL[i])
		mean = np.mean(CL[i])
		bounded_std = 1/(np.exp(10*Nc*std))
		addColumn.append(bounded_std)
	addColumn = np.matrix(addColumn).transpose()
	CL = np.hstack([CL, addColumn])
	#normalize after adding new probability
	for i in range(Nt):
		norm_denominator = 1 + addColumn[i]
		CL[i] /= norm_denominator

	#accuracy
	rightnum = 0
	anothernum = 0
	for i in range(Nt):
		if np.argmax(CL[i]) == 5:
			anothernum+=1
			continue
		if YValidate[i][np.argmax(CL[i])] == 1:
			rightnum+=1
	print "accuracy:", float(rightnum)/Nt
	print "anotherrate:", float(anothernum)/Nt

	YTest = CL
	return YTest