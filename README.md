# Anomaly Detection of Sensor Data

## Introduction
In the industrial manufacturing field, machines and equipment are essential to modern business, which are expected to operate at peak performance for extended periods. However, when the equipment experiences failure, it needs to be replaced, which often results in higher costs and waste of resources due to downtime and labor. Preventive maintenance can offer a sustainable solution by preventing unexpected problems through continuous monitoring and prediction of equipment data or sensor data, allowing for timely interventions.

## Problem Statement
In the semiconductor industry, a wafer goes through fabrication process that involves various tools, and a recipe defines how the wafer is processed by a tool. A wafer lot is a batch of multiple wafers that belong to the same technology and product.The sensor values of a tool are recorded for each wafer run, and this data is important for engineers to determine if the tool is working properly or not. Therefore, an FDC (fault detection and classification) system is necessary to detect any tool sensor anomalies to prevent further wafer scraps. 

## Aims and Objectives
This project focuses on building a anomaly detection model to detect wafer runs that are anomalous. The dataset does not contain labelled data (anomalous/non-anomalous), therefore an unsupervised learning method is utilised. Python and the Sci-kitLearn machine learning libraries are the primary tools used in this project. The objective of this project is to understand the characteristics of the data using Exploratory Data Analysis, and then carry out the following to build a data science/machine learning pipeline to perform the anomaly detection:

1.	Data preprocessing
2.	Feature engineering
3.	Feature selection
4.	Hyperparameter tuning
5.	Model training
6.	Model prediction
7.	Anomaly detection

## Solution Approach
The Isolation Forest algorithm is used to train the anomaly detection model. To validate the model's performance, the K-fold cross-validation is used to evaluate the performance of a model by splitting the data into multiple subsets, training the model on some subsets, and testing it on the remaining subsets. This process helps to ensure that the model's performance is robust and not dependent on any specific subset of the data. StratifiedKFold is used to ensure each fold has a similar distribution of anomalies and normal runs.

For further analysis, Principal Component Analysis (PCA) was used to reduce the dimensionality of the dataset, so that the anomalous and non-anomalous data can be visualised more clearly. As the dataset consist of categorical and numerical features, One Hot Encoding was executed for the categorical features and combined with the numerical features so that the Isolation Forest can be applied. The Isolation Forest algorithm is also applied to solely numerical features to compare the results between combined features vs only numerical features. The contamination parameter within the Isolation Forest algorithm was also tuned to obtain better model performance.

## Model Performance and Evaluation Metrics
The graphs depicting the model's performance can be visualised within the Jupyter Notebook files, which also provides the interpretation of the results. The below metrics were used to evaluate the model's performance, along with the visual inspection on the clusters formed by the anomalous and non-anomalous data. A greater separation between both clusters indicates the model's ability to differentiate between anomalous and non-anomalous data. 

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The harmonic mean of precision and recall.

## Key Findings
The anomaly detection models were able to achieve an F1-score of at least 75%. The F1-score is improved upon further hyperparameter tuning and feature selection to achieve a score of above 95%. The overlapping of anomalous and non-anomalous data was also greatly reduced, indicating that the model can potentially predict anomalous wafer runs to an acceptable degree.

## Further Findings and Improvements
As there is no labelled data on which runs are anomalous and non-anomalous, the results from this model prediction need to be further validated by performing a more in-depth analysis into the specific conditions (such as the MachineRecipe and Technology) to investigate the influence of these features on the results. The outliers should also be investigated to understand why these points stand out. As the fluctuation in sensor readings is quite high, it is difficult to tell if a run is anomalous using visual inspection. Furthermore, different sensors exhibit a different run pattern. Thus, rolling statistics can be used to calculate the rolling mean, variance and other statistics to understand the trends and anomalies in the run. Collaboration with domain experts may also help to deliver new insights into the run pattern to identify the anomalous sensor data. 

