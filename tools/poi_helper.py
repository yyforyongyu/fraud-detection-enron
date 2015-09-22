#!/usr/bin/python

"""
    A helper library for poi_id.py
"""

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import sys
import os
import csv
from feature_format import featureFormat, targetFeatureSplit
import numpy as np
import pandas as pd
from time import time


def buildRegression(features, labels):
    """
        Build a linear regression model for outliers cleaning.

        Return the predictions and the score.
    """

    ### fit the model and get predictions
    reg = LinearRegression().fit(features, labels)
    predictions = reg.predict(features)

    return predictions, reg.score(features, labels)

def outlierCleaner(features, labels, percent=.1):
    """
        clean away the given percent of points that have
        the largest residual errors (different between
        the prediction and the actual label value)

        return two lists - normals and outliers. Outliers
        are data points with the top given percent largest
        residual errors, the rest are in normals. Both of
        the lists are formatted as numpy array, and exactly
        like the formats after calling featureFormat.
    """

    normals, outliers, data = [], [], []

    ### get predictions
    predictions, score = buildRegression(features, labels)

    ### define the number of data points to be kept in normals
    length = int(len(predictions) * (1 - percent)) + 1

    ### create a dataset with a format: tuple(feature, label, residual errors)
    for i in range(len(predictions)):
        result = features[i], labels[i], (labels[i] - predictions[i]) ** 2
        data.append(tuple(result))

    ### sort dataset by deviations
    data.sort(key=lambda value: value[2])

    ### access dataset and create normals and outliers
    count = 0
    for values in data:
        count += 1
        if count <= length:
            normals.append(np.append([values[1]],values[0]))
        else:
            outliers.append(np.append([values[1]],values[0]))

    return normals, outliers

def featureReformat(numpy_array, features_list):
    """
        Format a numpy array object into a python
        dictionary object.

        Take a numpy array and features as inputs and
        return a python dictionary using features as
        keys and numpy array as values.
    """

    result = []

    for array in numpy_array:
        data_point = {}
        for i in range(len(features_list)):
            value = array[i]
            key = features_list[i]
            data_point[key] = value
        result.append(data_point)

    return result

def personMapping(dict_list, dataset, features_list):
    """
        Mapping a person's name based on the values of
        features.

        Take a list of dictionaries that has all the values
        of person's features, and map it with a dataset
        which has a person's name as a key, and its features
        and values as the key's item.

        Return a dictionary with a person's name as its key,
        and another dictionary as its value, which has features
        as its key, and values of features as its values,
        {name_of_person_1:
            {feature_1: value,
             feature_2: value,
             feature_3: value,
             ...},
         name_of_person_2:
             {...}}
    """

    my_dataset = {}
    ### iter through the dataset
    for key, item in dataset.iteritems():
        ### open the dictionary list
        for data in dict_list:
            ### open the features list
            for feature in features_list:
                ### filter out 'NaN' in the dataset
                ### check all the '0' values
                if item[feature] == "NaN":
                    if int(data[feature]) == 0:
                        find = True
                    else:
                        find = False
                        break
                else:
                    ### check every other feature between dictionary list and dataset
                    ### using a logical value 'find' to determine if a match is found
                    if int(data[feature]) == item[feature]:
                        find = True
                    else:
                        find = False
                        break
            ### iter through all features once
            ### if found, map the data to my_dataset
            if find:
                my_dataset[key] = item
    return my_dataset

def featureLabelSplit(my_dataset, features_list):
    """
        A simple function creates features and labels.

        Return features and labels
    """
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    return features, labels

def trainTestSplit(my_dataset, features_list, percent=.05):
    """
        A training and testing set split function.

        Take my_dataset and features_list as input, call on
        featueLabelSplit to create features and labels. Then
        use train_test_split to split datasets.

        Return training and testing datasets.
    """
    features, labels = featureLabelSplit(my_dataset, features_list)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    ### clean outliers
    cleaned_data, outliers = outlierCleaner(features_train, labels_train, percent)
    labels_train, features_train = targetFeatureSplit(cleaned_data)

    return features_train, features_test, labels_train, labels_test

def evaluateModel(y_true, y_pred):
    """
        A model evaluator.
        Calculate the model's accuracy score, f1 score,
        precision score, and recall score.

        Return scores and print out the scores as side effects.
    """

    accuracy = round(accuracy_score(y_true, y_pred), 3)
    f1 = round(f1_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred), 3)
    recall = round(recall_score(y_true, y_pred), 3)

    print """    Accuracy score: {}
    F1 score: {}
    Precision score: {}
    Recall score: {}""".format(accuracy, f1, precision, recall)

    return accuracy, f1, precision, recall

def tuneEstimator(pipeline, param, features_train, features_test, labels_train, cv=5):
    """
        Tune the classifiers to find the best estimator.

        Return the best estimator, predictions and scores.
    """

    clf = GridSearchCV(pipeline, param, scoring='f1', cv=cv)
    ### train the model
    clf.fit(features_train, labels_train)
    ### store the tuning results
    tuned_scores = clf.grid_scores_
    ### use the best estimator
    best_clf = clf.best_estimator_
    labels_pred = best_clf.predict(features_test)

    return best_clf, labels_pred, tuned_scores

def makePipelines(scalers, pca_methods, feature_selections, classifiers):
    """

    """
    pipeline_info = []
    for scaler in scalers:
        for pca in pca_methods:
            for feature_selection in feature_selections:
                for classifier in classifiers:
                    params = classifier[2]
                    if pca[0] == "none":
                        if scaler[0] == "none":
                            pipeline = Pipeline([feature_selection, classifier[:2]])
                        else:
                            pipeline = Pipeline([scaler, feature_selection, classifier[:2]])

                        name = (scaler[0], feature_selection[0], pca[0], classifier[0])
                        pipeline_info.append((pipeline, name, params))

                    else:
                        if scaler[0] == "none":
                            pipeline = Pipeline([feature_selection, pca, classifier[:2]])
                        else:
                            pipeline = Pipeline([scaler, feature_selection, pca, classifier[:2]])

                        name = (scaler[0], feature_selection[0], pca[0], classifier[0])
                        pipeline_info.append((pipeline, name, params))

    return pipeline_info

def trainModel(my_dataset, features_list, pipelines, filename='result.csv'):
    """
        A model training function.

        Take a dataset in python dictionary format, a list of
        features, a list of feature selection methods, and a
        list of classification methods. Iter through each list
        and make combinations of different feature selection
        method with different classification method. Then use
        tuneEstimator to tune the model. Finally, it evaluates
        the model based on accuracy score, precision score,
        recall score, and f1 score.

        Return a list of models and tuned scores, and writes a csv
        file to store the results.
    """

    ### split the training and testing sets and clean outliers on training set
    features_train, features_test, labels_train, labels_test = trainTestSplit(my_dataset, features_list, percent=.05)

    trained_model, tuned_score, model_results = [], [], []
    count = 0

    ### iter through each pipeline
    for pipeline_info in pipelines:
        pipeline = pipeline_info[0]
        scaler_name, selector_name, pca_name, clf_name = pipeline_info[1]
        params = pipeline_info[2]
        count += 1
        print "Model {} \n-working on classifier {}, using slection method {}, feature scaling {}, PCA {}".format(count, clf_name, selector_name, scaler_name, pca_name)

        ### add a time function to calculate time used by each model
        t0 = time()

        try:
            sss = StratifiedShuffleSplit(labels_train, n_iter=100, random_state=42)
            print "--start tuning..."
            clf, labels_pred, grid_scores = tuneEstimator(pipeline, params, features_train, features_test, labels_train, cv=sss)

            ### store the tuning results
            tuned_score.append(grid_scores)

            ### store model's information, including name, function, and parameters
            model_name = "{} with {} with {} with {}".format(clf_name, scaler_name, pca_name, selector_name)
            model_info = (model_name, clf)
            trained_model.append(model_info)

            t1 = time() - t0
            print "--training on {} complete, time used: {} \n--start cross validating...".format(model_name, t1)

            ### print out evaluation scores
            accuracy, f1, precision, recall = crossValidate(my_dataset, features_list, clf)

            ### claculate time used
            t2 = time() - t0 - t1
            print "cross validation complete, time used: {}".format(t2)

            ### store the information of models
            ### model number, scaler, pca, feature selection, classifier, accuracy, f1, precision, recall, time used
            model_results.append((count, scaler_name, pca_name, selector_name, clf_name, accuracy, f1, precision, recall, round((time() - t0), 3)))
            print ""

        except Exception, e:
            print "--error on tuning: \n", e, "\n"

    ### dump the model information into a csv file
    if filename != None:
        dumpResult(model_results, filename)

    return trained_model, tuned_score

def dumpResult(data, filename='result.csv'):
    """
        Take the results from running models and dump
        into a csv file named "result.csv"
    """

    ordered_data = findBest(data)

    with open(filename, "w") as f:
        writer = csv.writer(f)

        ### write row for a new file
        writer.writerow(["model", "scaler", "feature_selection_method", "pca",
                         "classification_method", "accuracy_score", "f1_score",
                         "precision_score", "recall_score", "time_used"])
        for model in ordered_data:
            writer.writerow(model)

def findBest(data):
    """
        Take the results from running models and filter out
        precision and recall scores less than 0.2.

        Return a list of reordered data.
    """

    ordered_data = []

    ### exclude model that has lower precision/recall scores
    for model in data:
        if model[6] < 0.2 or model[7] < 0.2:
            pass
        else:
            ordered_data.append(model)

    ### order by runtime
    ordered_data.sort(key=lambda value:value[-1])

    return ordered_data

def crossValidate(my_dataset, features_list, clf, percent=.05):
    """
        Take dataset, features list, estimator as inputs, run StratifiedShuffleSplit
        with n_iter = 1000, and calculate the scores.

        return accuracy, f1, recall, precision scores.
    """

    features, labels = featureLabelSplit(my_dataset, features_list)
    sss = StratifiedShuffleSplit(labels, n_iter=1000, random_state=42)

    PERF_FORMAT_STRING = "Accuracy: {}, Precision: {}, Recall: {}, F1: {}, F2: {}"
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_index, test_index in sss:
        features_train = [features[ii] for ii in train_index]
        features_test = [features[ii] for ii in test_index]
        labels_train = [labels[ii] for ii in train_index]
        labels_test = [labels[ii] for ii in test_index]

        cleaned_data, outliers = outlierCleaner(features_train, labels_train, percent)
        labels_train, features_train = targetFeatureSplit(cleaned_data)

        try:
            clf.fit(features_train, labels_train)
            predictions = clf.predict(features_test)
            for prediction, truth in zip(predictions, labels_test):
                if prediction == 0 and truth == 0:
                    true_negatives += 1
                elif prediction == 0 and truth == 1:
                    false_negatives += 1
                elif prediction == 1 and truth == 0:
                    false_positives += 1
                elif prediction == 1 and truth == 1:
                    true_positives += 1
                else:
                    print "Warning: Found a predicted label not == 0 or 1."
                    print "All predictions should take value 0 or 1."
                    print "Evaluating performance for processed predictions:"
                    break
        except Exception, e:
            print e
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = round(1.0*(true_positives + true_negatives)/total_predictions, 4)
        precision = round(1.0*true_positives/(true_positives+false_positives), 4)
        recall = round(1.0*true_positives/(true_positives+false_negatives), 4)
        f1 = round(2.0 * true_positives/(2*true_positives + false_positives+false_negatives), 4)
        f2 = round((1+2.0*2.0) * precision*recall/(4*precision + recall), 4)

        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5),
        print ""
    except Exception, e:
        print "Got a divide by zero when trying out", e

    return accuracy, f1, precision, recall

def gridScoreReader(tuning_score):
    """
        Take grid_scores and format it into a pandas dataframe.
    """

    result = []
    header = []
    for item in tuning_score:
        tuned_result = []
        params = item[0].keys()
        for key, value in item[0].iteritems():
            tuned_result.append(value)
        tuned_result.append(round(item[1], 4))
        tuned_result.append(round(np.std(item[2]), 4))
        result.append(tuned_result)
    for param in params:
        header.append(param.split("__")[-1])
    header += ['mean', 'std']
    return pd.DataFrame(result, columns = header)

def stdMeanReader(features, features_list, scaler=None):
    """
        Take features and features_list as inputs and turn them into
        a pandas dataframe. Use the scaler to transform features.
        Calculate the mean and standard deviations on each feature.

        Return a pandas dataframe of names, mean, and standard deviation
        of each feature.
    """
    if scaler == None:
        pass
    else:
        features = scaler.fit_transform(features)
        
    df_data = pd.DataFrame(features, columns = features_list[1:])

    result = [(features_list[1:][i], round(df_data.mean()[i], 4), round(df_data.std()[i], 4)) 
              for i in range(len(df_data.mean()))]
    return pd.DataFrame(result, columns = ['feature', 'mean', 'std']).sort(['std'], ascending = [0])