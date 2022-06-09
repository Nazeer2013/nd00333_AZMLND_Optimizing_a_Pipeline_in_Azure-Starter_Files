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


My conda jupyter notebook environment details.
        python version: 3.8.5, azureml version: 1.41.0, sklearn version: 1.0.2

I used Optum Azure account to run all my experiments and did not use Udacity Azure labs.


## Scikit-learn HyperDrive Pipeline Use case I


**a) Using Scikit-Learn HyperDrive Hyperparameters:**

***Pipeline Architecture :***

        1. Connect to your Workspace using config.
        2. Create Experiment within given workspace.
        3. Add Compute Target
        4. Specify Parameter sampling policy
        5. Specify early terminating policy
        6. Setup sklearn environment
        7. Specify Script run config
        8. Specify Hyperdrive config
        9. Submit job to run
       10. Gather performance and accuracy results.

Details of input parameters and results of this Experiment: This uses Logistic Regression 

****x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)****

Sampling policy: RANDOM and Parameter space {"--C":["uniform",[0.1,0.4]],"--max_iter":["choice",[[50,100,200,250]]]}

Early termination policy: BANDIT with Properties {"evaluation_interval":1,"delay_evaluation":5,"slack_factor":0.2}


![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_BestChild1.png)


**Results: Accuracy: 0.916 at Max iterations: 200 and Regularization Strength: 0.155**


![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_Results1.png)


**Benefits of the selected parameter**

Lower regularization strength (C) and median number of iterations choice resulted in better accuracy. 

**Benefits of the early stopping policy**

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.


## AutoML User Case II without validation data size

**Results: Accuracy: 0.918 with VotingEnsemble as best model selected**

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview.png)

In this case I initially forgot to add test data validation_size to AutoMLConfig but this resulted in best performance overall.


## AutoML User Case III witho validation data size

In this case I added validation_size as 25% to AutoMLConfig. Results for the best fit model 'VotingEnsemble' were same.

**Results: Accuracy: 0.918 with VotingEnsemble as best model selected** 

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoML_Run2_Overview1.png)


## Pipeline comparison

Using AutoML SDK pipeline resulted in slightly better performance of 0.918 accuracy compare to Scikit-learn Hyperdrive 0.916.

AutoML performed better due to number of iteration it could run and tune. Voting ensamble resulted in better accuracy,
this wasn't an option for HyperDrive.


Lower regularization strength (C) and median number of iterations choice resulted in better accuracy for Hyperdrive.


Given the amount of processing cost and time overhead with AutoML, I would prefer HyperDrive option as accuracy for both
cases is almost same.


## Future work

I would prefer to run more use cases and would like to deep dive into NLP and recommendation systems.

## Proof of cluster clean up

I used Optum Azure account for all my lab work. Compute resource is stopped or at 0 node for clusters.


### Please find below additional screen shots for above use cases 

## Hyperdrive Use Case I

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_ResultsOverview.png)




![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/HyperdriveRun_Results2.png)


## AutoML User Case II without validation data size


![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview2.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview3.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_ResultsOverview4.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun1_Results1.png)


## Use Case III

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoML_Run2_ConfigPrams.png)


![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoML_Run2_Algoview.png)

![image](https://github.com/Nazeer2013/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/images/AutoMLRun2_BestChildPerf.png)


## References:
https://azure.github.io/azureml-sdk-for-r/reference/bandit_policy.html

