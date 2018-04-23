import scipy.io as sio
import numpy as np
from ConfusionPlot import plot_confusion_matrix

def MyConfusionMatrix(Y, ClassNames):
	YList = []
	ClassNamesList = []
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			if Y[i][j] == 1:
				YList.append(j)
				continue

	for i in range(ClassNames.shape[0]):
		for j in range(ClassNames.shape[1]):
			if ClassNames[i][j] == 1:
				ClassNamesList.append(j)
				continue
	print YList,ClassNamesList

	cnf_matrix, accuracy = plot_confusion_matrix(ClassNamesList, YList, True)
	return cnf_matrix, accuracy
'''
if __name__ == '__main__':

	loaded_input = sio.loadmat('../Proj2FeatVecsSet1.mat')
	X = loaded_input['Proj2FeatVecsSet1']

	loaded_target_output = sio.loadmat('../Proj2TargetOutputsSet1.mat')
	Y = loaded_target_output['Proj2TargetOutputsSet1']

	Y1 = Y[5000:5100]
	Y2 = Y[5100:5200]

	Y1 = [1,2,3]
	Y2 = [2,2,2]
	#print Y1
	c, a = plot_confusion_matrix(Y1, Y2, True)
	print c,a
'''