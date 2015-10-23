#Introduction
This script is for predicting the MINST digit recognition dataset. The test and training set are public avaliable and can be downloaded at [here](https://www.kaggle.com/c/digit-recognizer/data). Since the MINST dataset is very well formatted, there is not much preprocessing needed. The algorithms used in this script are a combination of neural network and random forest classifier from the sklearn toolbox.



    import pandas
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import cross_validation
    from sklearn.neural_network import BernoulliRBM
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline


    train = pandas.read_csv("train.csv")
    test = pandas.read_csv("test.csv")

#Preprocessing
We first examine the shape of training set and test set. There are 42000 training samples 
and 28000 test samples, each with 784 features/pixels. Each featuere represents the intensity of the pixels with an integer between 0 to 255, inclusive. The first column of the training set gives the label of the number.


    print(train.shape)
    print(test.shape)
    print(train.describe())

    (42000, 785)
    (28000, 784)
                  label  pixel0  pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  \
    count  42000.000000   42000   42000   42000   42000   42000   42000   42000   
    mean       4.456643       0       0       0       0       0       0       0   
    std        2.887730       0       0       0       0       0       0       0   
    min        0.000000       0       0       0       0       0       0       0   
    25%        2.000000       0       0       0       0       0       0       0   
    50%        4.000000       0       0       0       0       0       0       0   
    75%        7.000000       0       0       0       0       0       0       0   
    max        9.000000       0       0       0       0       0       0       0   
    
           pixel7  pixel8    ...         pixel774      pixel775      pixel776  \
    count   42000   42000    ...     42000.000000  42000.000000  42000.000000   
    mean        0       0    ...         0.219286      0.117095      0.059024   
    std         0       0    ...         6.312890      4.633819      3.274488   
    min         0       0    ...         0.000000      0.000000      0.000000   
    25%         0       0    ...         0.000000      0.000000      0.000000   
    50%         0       0    ...         0.000000      0.000000      0.000000   
    75%         0       0    ...         0.000000      0.000000      0.000000   
    max         0       0    ...       254.000000    254.000000    253.000000   
    
              pixel777      pixel778      pixel779  pixel780  pixel781  pixel782  \
    count  42000.00000  42000.000000  42000.000000     42000     42000     42000   
    mean       0.02019      0.017238      0.002857         0         0         0   
    std        1.75987      1.894498      0.414264         0         0         0   
    min        0.00000      0.000000      0.000000         0         0         0   
    25%        0.00000      0.000000      0.000000         0         0         0   
    50%        0.00000      0.000000      0.000000         0         0         0   
    75%        0.00000      0.000000      0.000000         0         0         0   
    max      253.00000    254.000000     62.000000         0         0         0   
    
           pixel783  
    count     42000  
    mean          0  
    std           0  
    min           0  
    25%           0  
    50%           0  
    75%           0  
    max           0  
    
    [8 rows x 785 columns]


Not much preprocessing is needed for this dataset, since there are no missing values and everything is very well formatted.
# Prediction
Let's first attempt to evalute this dataset using random forest.


    feature_names = list(train.columns.values[1:])
    alg_rf = RandomForestClassifier(random_state = 1, 
                                    n_estimators = 100, 
                                    min_samples_split = 4, 
                                    min_samples_leaf = 2)
    score = cross_validation.cross_val_score(alg_rf, train[feature_names], train["label"], cv = 3)
    print(score, score.mean())

    [ 0.96008283  0.95970853  0.96313759] 0.960976318117


After tweaking the parameters of the RF classifier, the best score we can come up is 0.96097.
Let's try Neural Network model. The following choice of hyperparameter seems to be good for a small sample size of 5000. A more rigirous way to find the correct combinations would be to use GridSearchCV. For now we will use these choice for our model.


    %%capture
    logistic = LogisticRegression()
    logistic.C = 1.0
    rbm = BernoulliRBM(random_state = 1, verbose = True)
    rbm.learning_rate = 0.02
    rbm.n_components = 256
    rbm.batch_size = 20
    rbm.n_iter = 5
    alg_nn = Pipeline(steps = [('rbm', rbm), ('logistic', logistic)])
    train_s = train[feature_names]/255.0
    
    train_label = train["label"]
    test_N = 1000
    
    score = cross_validation.cross_val_score(alg_nn, train_s.iloc[1:test_N,:], train_label.iloc[1:test_N] , cv = 3);
    print(score, score.mean())

Read more about the hyperparameters that NN uses [here](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM)
and logistic regression [here](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM.). Hyper-parameters can be searched using GridSearchCV.


    alg_rf.fit(train_s, train_label)
    prediction_rf = alg_rf.predict(test[feature_names]/255.0)



    prediction_rf_prob = alg_rf.predict_proba(test[feature_names]/255.0)
    print( prediction_rf_prob[1,:] )

    [ 0.955  0.     0.005  0.     0.01   0.03   0.     0.     0.     0.   ]



    rbm.learning_rate = 0.02
    rbm.n_components = 256
    rbm.batch_size = 20
    rbm.n_iter = 100
    
    alg_nn.fit(train_s, train_label)

    [BernoulliRBM] Iteration 1, pseudo-likelihood = -103.13, time = 29.67s
    [BernoulliRBM] Iteration 2, pseudo-likelihood = -89.86, time = 36.32s
    [BernoulliRBM] Iteration 3, pseudo-likelihood = -82.41, time = 35.78s
    [BernoulliRBM] Iteration 4, pseudo-likelihood = -78.01, time = 35.99s
    [BernoulliRBM] Iteration 5, pseudo-likelihood = -75.19, time = 35.57s
    [BernoulliRBM] Iteration 6, pseudo-likelihood = -74.67, time = 35.52s
    [BernoulliRBM] Iteration 7, pseudo-likelihood = -72.05, time = 35.52s
    [BernoulliRBM] Iteration 8, pseudo-likelihood = -71.70, time = 35.60s
    [BernoulliRBM] Iteration 9, pseudo-likelihood = -69.97, time = 35.58s
    [BernoulliRBM] Iteration 10, pseudo-likelihood = -70.45, time = 35.51s
    [BernoulliRBM] Iteration 11, pseudo-likelihood = -69.10, time = 36.77s
    [BernoulliRBM] Iteration 12, pseudo-likelihood = -68.26, time = 35.70s
    [BernoulliRBM] Iteration 13, pseudo-likelihood = -68.83, time = 35.43s
    [BernoulliRBM] Iteration 14, pseudo-likelihood = -68.01, time = 35.51s
    [BernoulliRBM] Iteration 15, pseudo-likelihood = -67.47, time = 35.48s
    [BernoulliRBM] Iteration 16, pseudo-likelihood = -67.03, time = 35.49s
    [BernoulliRBM] Iteration 17, pseudo-likelihood = -66.73, time = 35.59s
    [BernoulliRBM] Iteration 18, pseudo-likelihood = -66.78, time = 35.45s
    [BernoulliRBM] Iteration 19, pseudo-likelihood = -66.20, time = 35.48s
    [BernoulliRBM] Iteration 20, pseudo-likelihood = -66.17, time = 35.77s
    [BernoulliRBM] Iteration 21, pseudo-likelihood = -65.57, time = 35.53s
    [BernoulliRBM] Iteration 22, pseudo-likelihood = -66.03, time = 35.59s
    [BernoulliRBM] Iteration 23, pseudo-likelihood = -65.13, time = 35.86s
    [BernoulliRBM] Iteration 24, pseudo-likelihood = -65.59, time = 35.58s
    [BernoulliRBM] Iteration 25, pseudo-likelihood = -64.72, time = 35.62s
    [BernoulliRBM] Iteration 26, pseudo-likelihood = -64.48, time = 35.56s
    [BernoulliRBM] Iteration 27, pseudo-likelihood = -65.25, time = 35.58s
    [BernoulliRBM] Iteration 28, pseudo-likelihood = -64.04, time = 35.43s
    [BernoulliRBM] Iteration 29, pseudo-likelihood = -63.64, time = 35.80s
    [BernoulliRBM] Iteration 30, pseudo-likelihood = -64.75, time = 35.55s
    [BernoulliRBM] Iteration 31, pseudo-likelihood = -63.88, time = 35.58s
    [BernoulliRBM] Iteration 32, pseudo-likelihood = -64.56, time = 35.81s
    [BernoulliRBM] Iteration 33, pseudo-likelihood = -64.79, time = 36.04s
    [BernoulliRBM] Iteration 34, pseudo-likelihood = -63.94, time = 35.54s
    [BernoulliRBM] Iteration 35, pseudo-likelihood = -63.52, time = 35.44s
    [BernoulliRBM] Iteration 36, pseudo-likelihood = -63.83, time = 35.57s
    [BernoulliRBM] Iteration 37, pseudo-likelihood = -63.67, time = 35.89s
    [BernoulliRBM] Iteration 38, pseudo-likelihood = -62.79, time = 35.80s
    [BernoulliRBM] Iteration 39, pseudo-likelihood = -63.13, time = 35.46s
    [BernoulliRBM] Iteration 40, pseudo-likelihood = -62.84, time = 35.57s
    [BernoulliRBM] Iteration 41, pseudo-likelihood = -63.18, time = 35.57s
    [BernoulliRBM] Iteration 42, pseudo-likelihood = -62.80, time = 35.52s
    [BernoulliRBM] Iteration 43, pseudo-likelihood = -63.30, time = 35.52s
    [BernoulliRBM] Iteration 44, pseudo-likelihood = -63.05, time = 35.51s
    [BernoulliRBM] Iteration 45, pseudo-likelihood = -62.97, time = 35.60s
    [BernoulliRBM] Iteration 46, pseudo-likelihood = -63.13, time = 35.81s
    [BernoulliRBM] Iteration 47, pseudo-likelihood = -63.96, time = 35.67s
    [BernoulliRBM] Iteration 48, pseudo-likelihood = -63.32, time = 35.58s
    [BernoulliRBM] Iteration 49, pseudo-likelihood = -63.29, time = 35.58s
    [BernoulliRBM] Iteration 50, pseudo-likelihood = -62.06, time = 35.50s
    [BernoulliRBM] Iteration 51, pseudo-likelihood = -62.82, time = 35.52s
    [BernoulliRBM] Iteration 52, pseudo-likelihood = -63.47, time = 35.94s
    [BernoulliRBM] Iteration 53, pseudo-likelihood = -63.30, time = 35.49s
    [BernoulliRBM] Iteration 54, pseudo-likelihood = -63.23, time = 35.53s
    [BernoulliRBM] Iteration 55, pseudo-likelihood = -62.42, time = 35.66s
    [BernoulliRBM] Iteration 56, pseudo-likelihood = -62.85, time = 35.52s
    [BernoulliRBM] Iteration 57, pseudo-likelihood = -62.50, time = 35.52s
    [BernoulliRBM] Iteration 58, pseudo-likelihood = -62.52, time = 35.47s
    [BernoulliRBM] Iteration 59, pseudo-likelihood = -62.49, time = 35.45s
    [BernoulliRBM] Iteration 60, pseudo-likelihood = -63.67, time = 35.39s
    [BernoulliRBM] Iteration 61, pseudo-likelihood = -62.46, time = 35.47s
    [BernoulliRBM] Iteration 62, pseudo-likelihood = -61.77, time = 35.48s
    [BernoulliRBM] Iteration 63, pseudo-likelihood = -62.67, time = 36.14s
    [BernoulliRBM] Iteration 64, pseudo-likelihood = -62.90, time = 35.62s
    [BernoulliRBM] Iteration 65, pseudo-likelihood = -61.93, time = 35.45s
    [BernoulliRBM] Iteration 66, pseudo-likelihood = -62.05, time = 35.43s
    [BernoulliRBM] Iteration 67, pseudo-likelihood = -61.41, time = 35.53s
    [BernoulliRBM] Iteration 68, pseudo-likelihood = -62.81, time = 35.45s
    [BernoulliRBM] Iteration 69, pseudo-likelihood = -62.24, time = 35.39s
    [BernoulliRBM] Iteration 70, pseudo-likelihood = -61.85, time = 35.73s
    [BernoulliRBM] Iteration 71, pseudo-likelihood = -62.01, time = 35.59s
    [BernoulliRBM] Iteration 72, pseudo-likelihood = -62.20, time = 35.52s
    [BernoulliRBM] Iteration 73, pseudo-likelihood = -62.38, time = 35.64s
    [BernoulliRBM] Iteration 74, pseudo-likelihood = -62.54, time = 35.53s
    [BernoulliRBM] Iteration 75, pseudo-likelihood = -62.43, time = 40.13s
    [BernoulliRBM] Iteration 76, pseudo-likelihood = -62.68, time = 36.50s
    [BernoulliRBM] Iteration 77, pseudo-likelihood = -61.70, time = 35.48s
    [BernoulliRBM] Iteration 78, pseudo-likelihood = -62.77, time = 35.48s
    [BernoulliRBM] Iteration 79, pseudo-likelihood = -61.90, time = 35.42s
    [BernoulliRBM] Iteration 80, pseudo-likelihood = -61.65, time = 35.43s
    [BernoulliRBM] Iteration 81, pseudo-likelihood = -62.12, time = 35.50s
    [BernoulliRBM] Iteration 82, pseudo-likelihood = -62.26, time = 35.71s
    [BernoulliRBM] Iteration 83, pseudo-likelihood = -62.40, time = 35.55s
    [BernoulliRBM] Iteration 84, pseudo-likelihood = -61.71, time = 35.56s
    [BernoulliRBM] Iteration 85, pseudo-likelihood = -62.10, time = 36.53s
    [BernoulliRBM] Iteration 86, pseudo-likelihood = -61.59, time = 35.53s
    [BernoulliRBM] Iteration 87, pseudo-likelihood = -61.55, time = 36.52s
    [BernoulliRBM] Iteration 88, pseudo-likelihood = -62.03, time = 35.55s
    [BernoulliRBM] Iteration 89, pseudo-likelihood = -61.51, time = 35.64s
    [BernoulliRBM] Iteration 90, pseudo-likelihood = -61.87, time = 35.60s
    [BernoulliRBM] Iteration 91, pseudo-likelihood = -61.92, time = 35.49s
    [BernoulliRBM] Iteration 92, pseudo-likelihood = -62.07, time = 35.68s
    [BernoulliRBM] Iteration 93, pseudo-likelihood = -60.92, time = 35.51s
    [BernoulliRBM] Iteration 94, pseudo-likelihood = -61.88, time = 35.56s
    [BernoulliRBM] Iteration 95, pseudo-likelihood = -61.48, time = 35.55s
    [BernoulliRBM] Iteration 96, pseudo-likelihood = -61.76, time = 35.50s
    [BernoulliRBM] Iteration 97, pseudo-likelihood = -62.40, time = 35.52s
    [BernoulliRBM] Iteration 98, pseudo-likelihood = -60.95, time = 35.69s
    [BernoulliRBM] Iteration 99, pseudo-likelihood = -60.82, time = 35.97s
    [BernoulliRBM] Iteration 100, pseudo-likelihood = -61.24, time = 35.51s





    Pipeline(steps=[('rbm', BernoulliRBM(batch_size=20, learning_rate=0.02, n_components=256, n_iter=100,
           random_state=1, verbose=True)), ('logistic', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr',
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0))])




    print(  prediction_rf_prob.argmax(axis = 1)  ) 

    [2 0 9 ..., 3 9 2]



    prediction_nn = alg_nn.predict_proba(test[feature_names]/255.0)
    print(  prediction_nn.argmax(axis = 1)  )  

    [2 0 9 ..., 3 9 2]



    prediction_ensemble = prediction_nn + prediction_rf_prob
    print(prediction_ensemble.argmax(axis = 1))


    [2 0 9 ..., 3 9 2]


#Summary
The best score obtained so far is by adding the probability prediction of random forest and neural network, and choosing the label with the maximum probability. The final submission is ensemble of these two methods, which gave a score of 96.8% accuracy. The best method to date uses convolutional neural network and can reach up to 99.8% accuracy.


    submission = pandas.DataFrame({
            "ImageId" : range(1,len(test)+1),
            "Label": prediction_ensemble.argmax(axis = 1),
        })
    print(submission.head(10))

       ImageId  Label
    0        1      2
    1        2      0
    2        3      9
    3        4      9
    4        5      3
    5        6      7
    6        7      0
    7        8      3
    8        9      0
    9       10      3



    submission.to_csv("kaggle3.csv", index = False)
