# Identifying Fraud from Enron Email

## About This Project

The goal of this project is to identify whether a person is guilty for the notorious Enron Fraud, using publicly available information like financial incomes and emails.

From oil-trading, bandwidth trading, and weather trading with market-to-market accounting, Enron has spread its hands to a variety of commodities, got in touch with politicians like George W. Bush, and caused a great loss to the public, including the California electricity shortage. All these information can be useful if text learning was applied, and certain patterns could be found out like, a pattern indicating a decision-making person could very likely be a person of interest. However, this is not applied in this analysis since it's a more advanced topic.

This analysis used a finacial dataset containing people's salary, stock information, and so on. During Enron Fraud, people like Jefferey Skilling, Key Lay, and Fastow all have dumped large amounts of stock options, and they are all guilty. This information can be very helpful to check on other person of interest, and can be easily refelected in the dataset. This is also where machine learning comes into play. By creating models which calculate relationships between a person of interest and its available quantitative data, machine learning tries to find and memorize a pattern that helps us identify a guilty person in the future.

## Overview of Main Report

The report is under directory documentations, named as ["training_main.html"](https://github.com/yyforyongyu/fraud-detection-enron/blob/master/documentations/training_main.html). For more compact and summarized reporting, please check ["documentation.html"](https://github.com/yyforyongyu/fraud-detection-enron/blob/master/documentations/documentation.html) in the same directory.

To view the reports in repo, please use the following links,
[summary](https://github.com/yyforyongyu/fraud-detection-enron/blob/master/main/documentation.ipynb)
[detailed report](https://github.com/yyforyongyu/fraud-detection-enron/blob/master/main/training_main.ipynb)

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

Several helper functions are built for this project in poi_helper.py, which can be found in tools/. For more details, report poi_id.ipynb has all the thoughts and steps in building these functions.
