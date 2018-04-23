ML18SP-Project-2

This is the second project of Machine Learning in 2018 Spring.

#About Gaussian Process Regression

Used sklearn.gaussian_process.GaussianProcessClassifier to classify

As this method doesn't support returning probabilities with all-pairs(i.e. one_vs_one) method, we have to use one_vs_rest to train a classifier for each pair of classes.

Also, returned EstParameters is a 2d array containing parameters for each pair of classes.