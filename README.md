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
Workspace Creation:
This workspace is used to manage data, compute resources, code, models, and other artifacts related to machine learning workloads.
Compute Instance Creation:
A compute instance should be created. This instance will be needed for training and managing our model, in this case to run the training process.
Hyperdrive Configuration:

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
