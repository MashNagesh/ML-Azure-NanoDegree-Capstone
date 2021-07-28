#  MNIST CLASIFICATION -CAPSTONE PROJECT - AZURE ML ENGINEER 

This Capstone project is part of the Azure ML Engineer NanoDegree.The Key components that were covered as part of the project areas follows:

1.Traning and identifying the Best model Run using AUTOML

2.Traning and identifying the Best model Run using Hyperdrive

3.Deploying the best model from the above 2 steps  and testing the functioning of the API


## Dataset

### Overview
The data used for this project is the famous MNIST dataset.It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9

Source of the data: https://www.kaggle.com/c/digit-recognizer/data


### Task
To predict the digit(0-9) in the label column 

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.The objective is to predict the label column using the other columns which provide the Pixel values

### Access
The data is accessed in the AzureML notebook using Kaggle API.

#### Steps:

1.Install Kaggle

2.Setup the Directory structure

3.Generate API token from Kaggle Account Page

4.Upload the Kaggle.Json containing UserName and Key

5.Once uploaded use chmod to change access permissions

6.Download the CompetitionZip file into the data directory

Reference:https://inclusive-ai.medium.com/how-to-use-kaggle-api-with-azure-machine-learning-service-da056708fc5a

## Automated ML

#### Choice of AutoML Settings:

1. n_cross_validation
Indicates how many cross validations to perform and in our case splitting it into 5 portions will ensure that we have ~8000 records for training and ~2000 for validation.

2. Primary Metric
Primary metric chosen here is accuracy to understand how much of the sample has been correctly classified.We could also use AUC as metric where we can see multiple one versus all Precision recall curves for each of the MNIST digits

3. enable early stopping
Early stopping is enabled to prevent overfitting

4. Experiment Stop time
To handle costs and time

5. Local compute
Going for the Local compute since we can pass dataframes and also to not create a separate compute (better cost)

### Results
Below are the models which have been chosen by Azure ML for this experiment

1.Voting Ensemble

2.Stack Ensemble

3.Max ABS scaler/Light GBM

4.Max ABS scaler/XGBoost Classifier

5.Random Forest

The Ensemble models perform better as opposed to the individual models since they combine bagging,bosting and stacking to provide the results. They also combine the results and minimise the variance component of the error

#### Results from AUTOML run using RunDetails
The Voting Ensemble models performs the best interms of the Primary Metric - Accuracy

![RunWidget](https://user-images.githubusercontent.com/26400438/127309495-ac6ac341-9556-48d6-aeed-8c8f17344850.PNG)

#### Best Model trained Parameters
Parameters of the Best Model - Voting Ensemble.The best model is a combination of 

1.Max ABS Scaler, XGBoost Classifier
Parameter used - tree_method ='auto'

2.MAX ABS Sclaer Light GBM
Parameter used - min_data_in_leaf =20

#### Scope of Improvement:

1.We can use a larger training data. For this experiment only 10,000 records have been considered and given that we have 0-9 digits only a sample of roughly 1000 is available per digit.

2.Use Deep Learning models in AUTOML.("enable_dnn": True in automl_settings)

#### Best Model trained Metrics:
The Primary metric used for model evaluation is Accuracy in this case.However we are able to see good values across multiple evlauation metrics for the best model

![best_run_metrics](https://user-images.githubusercontent.com/26400438/127309974-2170828f-555b-4dd6-b77a-467040868e2a.PNG)

## Hyperparameter Tuning

#### Choice of Model
The model being used is a simple Logistic regression. The focus of this excercise has been to understand the features of hyperdrive and to try out the same.

#### Early termination Policy
MedianStopping is a Conservative policy that provides savings without terminating promising jobs.It computes running averages across all runs and cancels runs whose best performance is worse than the median of the running averages.

#### Sampling Policy
The sampling Policy used is a Random Sampling Policy since the grid search suffers from limitations pertaining to higher dimensionality issues and Random Sampling though it functions very similar to grid search has been able to provide equal or better results in many scenarios. The chances of finding the optimal parameter are comparatively higher in random search because of the random search pattern where the model might end up being trained on the optimised parameters.

Range and values of Parameters as stated below 
'C': uniform(0.1, 10),'max_iter': choice(50,100,200,300)

#### Hyperparamters
Below hyperparameters are tuned in this model

C - Inverse of Regularisation strength

Max_iter - Maximum number of iterations to converge


### Results

Below are the results from the HyperDrive model

#### Run Details Widget - Model Running
![HyperDrive_Run_Details_Widget_2](https://user-images.githubusercontent.com/26400438/127315201-2d719767-2186-46ef-91ef-a4e4b4249fc7.PNG)

#### Run Details Widget - Model completion
![HyperDrive_Run_Details_Widget_3](https://user-images.githubusercontent.com/26400438/127315262-7d8518e1-990c-4b29-9e15-e87a05ff18c6.PNG)

#### Results from the Best Model 
The best model has an accuracy of 0.85492

![HD_Best_run_4](https://user-images.githubusercontent.com/26400438/127315588-16203112-9411-4a3f-b30e-2e528e7c46b4.PNG)

#### Parameters of the Best Model
C = 6.65
max_iter = 50

![HD_model_Save](https://user-images.githubusercontent.com/26400438/127315436-226b6d28-be79-4a38-a88c-ec81fb2b5fb1.PNG)

*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
