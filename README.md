# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset contains data about socio-economical statuses, financial records and similars. Here we are seeking to predict if for a specific person a bankmarketing campaing will take place or not.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was in this case was the "VotingEnsemble" with an accuracy of 0.91551

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
The pipeline architecture could be described under the following steps:
- Workspace Creation:
This workspace is used to manage data, compute resources, code, models, and other artifacts related to machine learning workloads.
- Compute Instance Creation:
A compute instance should be created. This instance will be needed for training and managing our model, in this case to run the training process.
- Hyperdrive Configuration:
The Hyperdrive Configuration allows us to automate the hyperparameter tuning to control the model training process on much more efficient way thant the classic "manual" approach
Within the Hyperdrive Configuration two inputs allowed some nice benefits. These are:

- Parameter Sample and early stopping policy.

Here in short some of the benefits of the above mentioned features:

- Parameter Sample:
  This supports discrete and continuous hyperparameters.In random sampling, hyperparameter values are randomly selected from the defined search space, which allows supports of   early termination of low-performance runs which again reduces computational expensiveness.

- Early Stopping Policy:
  This policy allows to automatically terminate poorly performing runs with an early termination policy. The result of this early termination is to improve computational   efficiency. In this specif case the setup of the policy was the following: Bandit Policy which based on slack factor/slack amount and evaluation interval terminates runs where   the primary metric is not within the specified slack factor/slack amount compared to the best performing run

## AutoML Model 
The results of the AutoML gave as a winner a "VotingEnsemble" model with an accuracy of 0.91551
The voting ensemble method combines conceptually different machine learning classifiers and uses a majority vote or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses.
In this case the model recommended from the AutoMl (VotingEnsemble) presents the following (main)parameters:

-Estimators: This refers to the different ensembled algorithms that were tested. In this case the following were performed:
'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'SGD', 'RandomForest'
- Weights: This refers to the sequence of weights (float or int) to weight the occurrences of predicted class labels (hard voting) or class probabilities before averaging (soft voting). The ensemble weights for each of the above mentioned algorithms are the following:
0.2857142857142857, 0.14285714285714285, 0.07142857142857142, 0.14285714285714285, 0.21428571428571427, 0.07142857142857142, 0.07142857142857142]

In addition to the above mentioned information you can see below a small dictionary containing all the parameters of the AutoMl:

{'ensembled_iterations': '[0, 27, 1, 26, 32, 10, 14]',
 'ensembled_algorithms': "['LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'SGD', 'RandomForest']",
 'ensemble_weights': '[0.2857142857142857, 0.14285714285714285, 0.07142857142857142, 0.14285714285714285, 0.21428571428571427, 0.07142857142857142, 0.07142857142857142]',
 'best_individual_pipeline_score': '0.9166803789014268',
 'best_individual_iteration': '0',
 'model_explanation': 'True'}



## Pipeline comparison
Comparing the results of the Hyperdrive agains the AutoMl there are actually not many differences. Let's evaluate for instance the main metric of interest:

- Hyperdrive Accuracy: 0.9104151493080845
- AutoMl Accuravy: 0.91551

## Future work
A good start would be to perform some sampling up/down since the classes are not balanced. This would definetely help to reduce model bias


