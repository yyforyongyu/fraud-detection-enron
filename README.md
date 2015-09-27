# Identifying Fraud from Enron Email

This is a report on the process of builing estimators for Fraud Detection using machine learning.
A more compact and summurized report can be found as Documentation (html), or Documentation (ipynb).

==============

## Overview

In this report, there are series of investigations performed to make a robust, strong final estimator to predict a person-of-interest(poi). These include,
- an overview of the dataset.
- outlier cleaning.
- a performance comparison among different feature scaling methods, including MinMaxScaler, StandardScaler, and Normalizer.
- creating three features, "stock_salary_ratio", "poi_from_ratio", "poi_to_ratio", and evaluating them.
- a performance comparison between two different feature selection methods, SelectKBest and ExtraTreesClassifier.
- a performance comparison between including PCA and excluding PCA.
- a performance comparison between different classifiers, LinearSVC and KNeighborsClassifier.
- tuning algorithms using F1 score as evaluation metric.
- cross-validation on the final estimator.

Several helper functions are built for this project in poi_helper.py. Since this report only focuses on methodology in machine learning, we will not cover them here. For more details, report poi_id.ipynb has all the thoughts and steps in building these functions.


A nanodegree project. All the files are under final_project/ folder, a list of files are explained as below.

## project_5_machine_learning.html
This is the detailed report for this project which contains all the information needed. It is highly recommeneded to use this report over others.

## project_5_machine_learning.ipynb
A notebook file to generate the report.

## poi_id.py
A simplified version of the report.

## poi_helper.py
Several helper functions in support for poi_id.py.

## documentation.html
This is the documentation for all the work.

## documentation.ipynb
A notebook file to generate the documentation.

## poi_id.ipynb
A notebook file has the draft to the final report. This report documented the thoughts behind this machine learning.