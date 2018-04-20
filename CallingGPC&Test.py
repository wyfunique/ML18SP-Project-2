import scipy.io as sio
import numpy as np
from MyGPC import GPC
from MyGPCTest import Test

if __name__ == '__main__':
	loaded_input = sio.loadmat('Proj2FeatVecsSet1.mat')
	X = loaded_input['Proj2FeatVecsSet1']

	loaded_target_output = sio.loadmat('Proj2TargetOutputsSet1.mat')
	Y = loaded_target_output['Proj2TargetOutputsSet1']

	Parameters = {'kernel': None, 'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0, 
	'copy_X_train': True, 'random_state': None, 'max_iter_predict': 100, 'warm_start': False, 
	'multi_class': 'one_vs_rest', 'n_jobs': 1}

	#Y[Y == -1] = 0
	#N = Y.shape[0]
	#Y = np.hstack( (Y, np.zeros((N,1))) )
	#print Y
	
	XE0 = X[:200]
	XE1 = X[5000:5200]
	XE2 = X[10000:10200]
	XE3 = X[15000:15200]
	XE4 = X[20000:20200]
	
	XE = np.vstack([XE0, XE1, XE2, XE3, XE4])

	YE0 = Y[:200]
	YE1 = Y[5000:5200]
	YE2 = Y[10000:10200]
	YE3 = Y[15000:15200]
	YE4 = Y[20000:20200]
	YE = np.vstack([YE0, YE1, YE2, YE3, YE4])

	
	XE0 = X[:4000]
	XE1 = X[5000:9000]
	XE2 = X[10000:14000]
	XE3 = X[15000:19000]
	XE4 = X[20000:24000]
	
	XEE = np.vstack([XE0, XE1, XE2, XE3, XE4])

	YE0 = Y[:4000]
	YE1 = Y[5000:9000]
	YE2 = Y[10000:14000]
	YE3 = Y[15000:19000]
	YE4 = Y[20000:24000]
	YEE = np.vstack([YE0, YE1, YE2, YE3, YE4])
	

	XV0 = X[4000:5000]
	XV1 = X[9000:10000]
	XV2 = X[14000:15000]
	XV3 = X[19000:20000]
	XV4 = X[24000:25000]
	XV_a = np.vstack([XV0, XV1, XV2, XV3, XV4])
	XV_b = np.vstack([XV3, XV4])

	YV0 = Y[4000:5000]
	YV1 = Y[9000:10000]
	YV2 = Y[14000:15000]
	YV3 = Y[19000:20000]
	YV4 = Y[24000:25000]
	YV_a = np.vstack([YV0, YV1, YV2, YV3, YV4])
	YV_b = np.vstack([YV3, YV4])
	YValidate, EstParameters = GPC(XE, YE, XV_a, YV_a, Parameters)
	YTest = Test(XV_b, YV_b, EstParameters)
#	np.savetxt("output.txt", YV, newline = "")	