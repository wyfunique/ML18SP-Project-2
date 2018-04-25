1. Dependencies:
    python 2.7.8
    scipy 1.0.0: loadmat() to load image mats
    numpy 1.13.1: main computing package 
    scikit-learn 0.19.1: SVM, GPR
    matplotlib 2.2.2
    # add your library here
    

2. Notice:
    
    (1) In code folder, there are some other modules that are needed by the 4 main py files, like MySVM, MyGOC, MyRVM and so on.
    
    (2) In function TrainMyClassifier(), since it is needed to pass class labels of estimation and validation into this function, 
        we seperate them as two parameters, 'ClassLabelsEstimate' and 'ClassLabelsValidate'.
        In addition, since we need to get the number of support and relevance vectors after training, we add a return parameter called 'VecNum' to do this.
    
    (3) In function MyCrossValidate(), the parameter 'Parameters' contains which model you want to use.
        The return items of this function are all lists instead of numpy arrays.
        
    (4) In all functions, the parameter 'Parameters' is a dictionary containing some special parameters you need. 
        For case of using SVM, 'Parameters' only need to have one key-value pair, {'type':'SVM'}.
        For case of using RVM, # Edit here
        For case of using GPC, # Edit here
    
    # please add what you want professor to notice. 