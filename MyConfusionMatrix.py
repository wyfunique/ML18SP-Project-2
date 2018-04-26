import scipy.io as sio
import numpy as np
from sklearn.metrics import confusion_matrix
from ConfusionPlot import plot_confusion_matrix
'''
Program that produces the confusion matrix from the Y predicted values and the ground truth
#input: Y from Test and Validation, Class Labels
#output: confusion matrix and average accuracy
'''
def MyConfusionMatrix(Y, ClassNames, normalize=True):
	  """
      This function generates the confusion matrix and plots it. Code partially derived
      from library example.
      Source URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-...
      ...auto-examples-model-selection-plot-confusion-matrix-py
      Author: Ronald Wilson
      :param Y: True Class labels
      :param ClassNames: Predicted Class labels
      :param normalize: Set flag to 'True' to normalize the confusion matrix. Default: False
      :return: None
      """
	   #print accuracy
	  # Generate Confusion Matrix
	  cnf_matrix = confusion_matrix(Y, ClassNames)
      #calculate accuracy
	  total_num = 0
	  correct_num = 0
	  if normalize:
	  	cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
	  for i in range(cnf_matrix.shape[0]):
		  for j in range(cnf_matrix.shape[1]):
			  total_num += cnf_matrix[i][j]
			  if i == j:
				  correct_num += cnf_matrix[i][j]
				  accuracy = float(correct_num) / total_num
				  # Normalize the Confusion Matrix if flag is set

	  plot_confusion_matrix((Y), (ClassNames), normalize=True)
	  return cnf_matrix, accuracy
