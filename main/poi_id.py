#!/usr/bin/python

import sys
sys.path.append("../tools/")
from poi_helper import *
import pickle
sys.path.append("../test/")
from tester import dump_classifier_and_data, test_classifier
import warnings
warnings.filterwarnings("ignore")

### Load the dictionary containing the dataset
data_dict = pickle.load(open("../datasets/final_project_dataset.pkl", "r") )

# remove the outlier 'TOTAL'
data_dict.pop("TOTAL")

# create features for plots
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
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
                 'from_poi_to_this_person']

# format the dataset
data = featureFormat(data_dict, features_list)

# create a pandas dataframe
df = pd.DataFrame(data, columns = features_list)

### prepare feature scaling
from sklearn.preprocessing import MinMaxScaler
scalers = [('minmaxscaler', MinMaxScaler())]

### prepare for feature selection
from sklearn.feature_selection import SelectKBest

feature_selections = [('k_best', SelectKBest(k = 'all'))]

### prepare for pca
from sklearn.decomposition import PCA
pca = [('pca', PCA())]

### prepare for classifiers
from sklearn.svm import LinearSVC
linear_svc = LinearSVC(class_weight='auto', penalty='l1', dual=False, random_state=42)

params_svc = {'linear_svc__C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10],
              'linear_svc__tol': [1e-4, 1e-3, 1e-2, 1],
              'linear_svc__max_iter': [1e3, 1e4]}

### put all classifiers together
classifiers = [('linear_svc', linear_svc, params_svc)]

### create pipelines
pipelines = makePipelines(scalers, pca, feature_selections, classifiers)

### train models on features list
model_sets, scores = trainModel(data_dict, features_list, pipelines, filename = None)

### extract the pipeline
pipeline = model_sets[0][1]
tuning_score = scores[0]

### set the training and testing set with clean ratio of 0.02
features_train, features_test, labels_train, labels_test = trainTestSplit(data_dict, features_list, percent=0.02)

### prepare the cross validation
sss = StratifiedShuffleSplit(labels_train, n_iter=100, random_state=42)

### record time
t0 = time()
print "\nStarting tuning on seleck k_best..."

### set the parameters
params_k_best = {"k_best__k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}

### fit and search
estimator = GridSearchCV(pipeline, params_k_best, scoring='precision', cv=sss)
estimator.fit(features_train, labels_train)

### extract scores
score_k_best = estimator.grid_scores_

### get the best estimator
clf = estimator.best_estimator_

### check the model performance
crossValidate(data_dict, features_list, clf, percent=0.02)

print "tuning complete, time used: ", time() - t0

### record time
print "\nStart tuning on pca..."
t0 = time()

### set the parameters
params_pca = {"pca__n_components": [1, 2, 3, 4, 5],
              "pca__whiten": [True, False]}

### fit and search
estimator = GridSearchCV(clf, params_pca, scoring='precision', cv=sss)
estimator.fit(features_train, labels_train)

### extract scores
score_pca = estimator.grid_scores_

### get the best estimator
clf = estimator.best_estimator_

### check the model performance
crossValidate(data_dict, features_list, clf, percent=0.02)

print "tuning compelte, time used: ", time() - t0

### prepare for the test
clf = clf
my_dataset = data_dict

### get chosen features from select k best
k_best_result = zip(features_list[1:], clf.steps[1][1].scores_)
k_best_result.sort(key=lambda value:value[1], reverse=True)

features_list = ['poi'] + [i[0] for i in k_best_result[:5]]

### dump for testing
dump_classifier_and_data(clf, my_dataset, features_list)

### check the score from tester
print "\nChecking scores..."
test_classifier(clf, my_dataset, features_list, folds = 1000)