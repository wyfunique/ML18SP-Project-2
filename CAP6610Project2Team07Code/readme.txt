1. Dependencies:
    python 2.7.8
    scipy 1.0.0: loadmat() to load image mats
    numpy 1.13.1: main computing package 
    scikit-learn 0.19.1: SVM, GPR
    matplotlib 2.2.2
    # add your library here
    

2. Notice:
    
    (1) In code folder, there are some other modules that are needed by the 4 main py files, like MySVM, MyGOC, MyRVM and so on.
    
    (2) In function TrainMyClassifier(), since it is needed to pass class labels of estimation and validation into this function, we seperate them as two parameters, 'ClassLabelsEstimate' and 'ClassLabelsValidate'.
        In addition, since we need to get the number of support and relevance vectors after training, we add a return parameter called 'VecNum' to do this.
    
    (3) In function MyCrossValidate(), the parameter 'Parameters' contains which model you want to use.
        The return items of this function are all lists instead of numpy arrays.
        
    (4) In all functions, the parameter 'Parameters' is a dictionary containing some special parameters you need. 
        For case of using SVM, 'Parameters' only need to have one key-value pair, {'type':'SVM'}.
        For case of using RVM, ### EDIT HERE
        For case of using GPC, other than 'type': 'GPC', 'Parameters' need to include some hyper-parameters. e.g:
        Parameters = {'kernel': None, 'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0, 'copy_X_train': True, 'random_state': None, 'max_iter_predict': 100, 'warm_start': False, 'multi_class': 'one_vs_rest', 'n_jobs': 1, 'type': 'GPC'}

    (5) We added two variables in TrainMyClassifier() and TestMyClassifier(): ClassLabelsEstimate and ClassLabelsValidate, representing class labels for estimation and validation.

    (6) As GPR in sklearn doesn't support returning probabilities with all-pairs(i.e. one_vs_one) method, we have to use one_vs_rest to train a classifier for each pair of classes. So returned EstParameters is a dict of parameters and hyper-parameters, each of them is a 2d array containing parameters for each pair of classes.

    (7) For MyConfusionMatrix(), the input Y should be estimated class names, and ClassNames is the true answer. Their format should be the same as input 'Proj2TargetOutputsSet1.mat', instead of class labels with probabilities.
    
    # please add what you want professor to notice. 