# Optimizing an ML Pipeline in Azure

## Overview

In this project, I created and optimized an ML pipeline.

I updated custom-coded model using standard Scikit-learn Logistic Regression—the hyperparameters and optimized using HyperDrive.

I also  used AutoML to build and optimize a model on the same dataset, so that you can compare the results of the two methods.

I worked on three different use cases for which I'm attaching images of results 

These models are then compared to an Azure AutoML run.






## Useful Resources

- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

For a given set of banking data predict whether a customer will be interested in borowing loan.
Parameters considered to run load prediction are  numeric, strings and boolean like age, job, marital, education, loan, default etc. 


As described I ran three use cases first using HyperDrive and two using AutoML SDK




**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**


My conda jupyter notebook environment details.
        python version: 3.8.5, azureml version: 1.41.0, sklearn version: 1.0.2

I used Optum Azure account to run all my experiments and did not use Udacity Azure labs.


## Scikit-learn Pipeline


**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**



**a) Using Scikit-Learn HyperDrive Hyperparameters:**

        ***Pipeline Architecture :***

        **1. Connect to your Workspace using config.**
        **1. Connect to your Workspace using config.**
 

Sampling policy: RANDOM and Parameter space {"--C":["uniform",[0.1,0.4]],"--max_iter":["choice",[[50,100,200,250]]]}

Early termination policy: BANDIT with Properties {"evaluation_interval":1,"delay_evaluation":5,"slack_factor":0.2}

**Accuracy: 0.916 at Max iterations: 200 and Regularization Strength: 0.155**

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_BestChild1.png)




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


### Images

## Hyperdrive Use Case 

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_ResultsOverview.png)


![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_Results1.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_Results2.png)


## AutoML User Case I no validation set

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview2.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview3.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview4.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_Results1.png)


## Usw Case II

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoML_Run2_Overview1.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoML_Run2_ConfigPrams.png)


![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoML_Run2_Algoview.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun2_BestChildPerf.png)

