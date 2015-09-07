#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from poi_helper import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'expenses',
                 'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'from_poi_to_this_person']# You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


### Task 2: Remove outliers
data_dict.pop("TOTAL")

### Store to my_dataset for easy export below.
my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### use a regression model to find outliers
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(features, labels)
predictions = reg.predict(features)

### extract normal data points and outliers
cleaned_data, outliers = outlierCleaner(predictions, features, labels)


### Task 3: Create new feature(s)
### add new features to dataset
for key, item in data_dict.iteritems():
    ### add stock_salary_ratio
    if item['salary'] != "NaN" and item['total_stock_value'] != "NaN":
        item['stock_salary_ratio'] = float(item['total_stock_value']) / item['salary']
    else:
        item['stock_salary_ratio'] = "NaN"

    ### add poi_from_ratio
    if item['from_messages'] != "NaN" and item['from_poi_to_this_person'] != "NaN":
        item['poi_from_ratio'] = float(item['from_poi_to_this_person']) / item['from_messages']
    else:
        item['poi_from_ratio'] = "NaN"

    ### add poi_to_ratio
    if item["to_messages"] != "NaN" and item["from_this_person_to_poi"] != "NaN":
        item["poi_to_ratio"] = float(item["from_this_person_to_poi"]) / item["to_messages"]
    else:
        item["poi_to_ratio"] = "NaN"

### update dataset
my_cleaned_dataset = personMapping(featureReformat(cleaned_data, features_list), data_dict, features_list)

### update features_list
features_list += ["stock_salary_ratio", "poi_from_ratio", "poi_to_ratio"]

### choose a feature selection method
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

feature_selection = [('k_best', SelectKBest(k = 5)),
                     ('linear_svc_l1', LinearSVC(C=0.01, penalty="l1", dual=False, random_state=31)),
                     ('logistic_reg', RandomizedLogisticRegression(C=1, selection_threshold=0.01, random_state=31)),
                     ('extra_tree', ExtraTreesClassifier(max_features=5, random_state=31))]

from sklearn.pipeline import FeatureUnion, Pipeline
### combine pca to feature selection
combined_feature = []
for method in feature_selection:
    new_method = FeatureUnion([('pca', PCA(n_components=5)), method])
    name = method[0] + "_with_pca"
    combined_feature.append((name, new_method))

### update feature selection list
feature_selection += combined_feature

### Task 4: Try a varity of classifiers
linear_svc = LinearSVC(random_state=31, dual=False)

params_svc = {'linear_svc__C':[1e-2, 1e-1, 1, 1e2, 1e3],
              'linear_svc__penalty': ["l1", "l2"],
              'linear_svc__tol': [1e4, 1e3, 1e2, 1],
              'linear_svc__max_iter': [1e2, 1e3, 1e4, 1e5]}

from sklearn.neighbors import KNeighborsClassifier
k_neighbors = KNeighborsClassifier()
params_kneighbors = {'k_neighbors__n_neighbors': [1, 5, 10, 20, 50],
                     'k_neighbors__weights': ["uniform", "distance"],
                     'k_neighbors__algorithm':["ball_tree", "kd_tree", "brute"],
                     'k_neighbors__leaf_size': [2, 5, 10, 30, 50, 100],
                     'k_neighbors__p': [1,2]}

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_boost = AdaBoostClassifier(random_state=31)
params_adaboost = {'ada_boost__base_estimator': [DecisionTreeClassifier(), None],
                   'ada_boost__n_estimators': [1, 5, 10, 20, 50, 100],
                   'ada_boost__algorithm': ['SAMME', 'SAMME.R'],
                   'ada_boost__learning_rate': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10],
                   }

classifiers = [('linear_svc', linear_svc, params_svc),
               ('k_neighbors', k_neighbors, params_kneighbors),
               ('ada_boost', ada_boost, params_adaboost)]

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

cleaned_model_sets_scaled, tuned_score_2 = trainModel(my_cleaned_dataset, features_list,
    feature_selection=feature_selection, classifiers=classifiers, scaling=True)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = pipeline_pca_linearsvc = cleaned_model_sets_scaled[0][1]
my_dataset = my_cleaned_dataset

dump_classifier_and_data(clf, my_dataset, features_list)