#!/usr/bin/python

"""
    A helper library for poi_id.py
"""

from feature_format import featureFormat, targetFeatureSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np

def outlierCleaner(predictions, features, labels):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual label value)

        return two lists - normals and outliers. Outliers
        are data points with the top 10% largest residual
        errors, the rest are in normals. Both of the lists
        are formatted as numpy array, and exactly like the
        formats after calling featureFormat.
    """

    normals = []
    outliers = []
    data = []
    length = int(len(predictions) * 0.9) + 1 # define the number of data points to be kept in normals

    ### create a dataset with a format:
    ### tuple(feature, label, residual errors)
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

def featureReformat(numpy_array, features):
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
        for i in range(len(features)):
            value = array[i]
            key = features[i]
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
        A simple function creates features and labels

        Return features and labels
    """
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return features, labels

def trainTestSplit(my_dataset, features_list):
    """
        A training and testing set split function.

        Take my_dataset and features_list as input, call on
        featueLabelSplit to create features and labels. Then
        use train_test_split to split datasets.

        Return training and testing datasets.
    """

    features, labels = featureLabelSplit(my_dataset, features_list)

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.25, random_state=42)

    return features_train, features_test, labels_train, labels_test

def evaluateModel(y_true, y_pred):
    """
        A model evaluator.
        Calculate the model's accuracy score, f1 score,
        precision score, and recall score.

        Return nothing. Print out the scores as side effects.
    """

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print """    Accuracy score: {}
    F1 score: {}
    Precision score: {}
    Recall score: {}""".format(accuracy, f1, precision, recall)

def tuneEstimator(pipeline, param, features_train, features_test, labels_train):
    """
        Tune the classifiers to find the best estimator.

        Return the best estimator, predictions and scores.
    """

    clf = GridSearchCV(pipeline, param)
    ### train the model
    clf.fit(features_train, labels_train)
    ### store the tuning results
    tuned_scores = clf.grid_scores_
    ### use the best estimator
    best_clf = clf.best_estimator_
    labels_pred = best_clf.predict(features_test)
    return best_clf, labels_pred, tuned_scores

def trainModel(my_dataset, features_list, feature_selection, classifiers):
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

        Return a list of models and tuned scores.
    """

    ### split the training and testing sets
    features_train, features_test, labels_train, labels_test = trainTestSplit(my_dataset, features_list)

    trained_model = []
    count = 0
    tuned_score = []
    ### iter through feature selection and classification methods
    for selection_method in feature_selection:
        for item in classifiers:
            count += 1
            print "Model {} \n-working on classifier {}, using slection method {}".format(count, item[0], selection_method[0])
            ### add a time function to calculate time used by each model
            from time import time
            t0 = time()
            ### unpack name, function and parameters
            classifier = item[:2]
            param = item[2]
            try:
                ### build pipeline
                pipeline = Pipeline([selection_method, classifier[:2]])
                ### tune the model
                try:
                    print "--start tuning..."
                    clf, labels_pred, grid_scores = tuneEstimator(pipeline, param, features_train, features_test, labels_train)
                    ### store the tuning results
                    tuned_score.append(grid_scores)
                    ### store model's information, including name, function, and parameters
                    model_name = item[0] + " with " + selection_method[0]
                    model_info = (model_name, clf)
                    trained_model.append(model_info)
                    print "--training on {} complete, time used {}".format(model_name, time() - t0)
                    ### print out evaluation scores
                    evaluateModel(labels_test, labels_pred)
                    print ""
                except Exception, e:
                    print "--error on tuning: \n", e, "\n"
            except Exception, e:
                print "-error on classifying: \n", e, "\n"
    return trained_model, tuned_score
