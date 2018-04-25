import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(actual_labels, predicted_labels, normalize=False):
    """
    This function generates the confusion matrix and plots it. Code partially derived
    from library example.
    Source URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-...
    ...auto-examples-model-selection-plot-confusion-matrix-py
    Author: Ronald Wilson
    :param actual_labels: True Class labels
    :param predicted_labels: Predicted Class labels
    :param normalize: Set flag to 'True' to normalize the confusion matrix. Default: False
    :return: None
    """

    # Color Map for the plot
    cmap = plt.cm.Blues

    # Generate Confusion Matrix
    cnf_matrix = confusion_matrix(actual_labels, predicted_labels)
    #calculate accuracy
    total_num = 0
    correct_num = 0
    for i in range(cnf_matrix.shape[0]):
        for j in range(cnf_matrix.shape[1]):
            total_num += cnf_matrix[i][j]
            if i == j:
                correct_num += cnf_matrix[i][j]
    accuracy = float(correct_num) / total_num
    # Normalize the Confusion Matrix if flag is set
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot Confusion Matrix
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(actual_labels)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #print accuracy
    return cnf_matrix, accuracy


if __name__ == "__main__":
    # Test Case
    plot_confusion_matrix([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [1, 1, 2, 1, 2, 1, 2, 2, 3, 3, 4, 2, 4], True)
