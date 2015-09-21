#!/usr/bin/python

sys.path.append("../tools/")
from poi_helper import *
import pickle
from tester import dump_classifier_and_data, test_classifier

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

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

### update features list
new_features_list = features_list + ["stock_salary_ratio", "poi_from_ratio", "poi_to_ratio"]

### prepare feature scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

### create scalers
scalers = [('none', None),
           ('standardscaler', StandardScaler()),
           ('minmaxscaler', MinMaxScaler()),
           ('normalier', Normalizer())]

### prepare for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier

feature_selection = [('k_best', SelectKBest(k = 'all')),
                     ('extra_tree', ExtraTreesClassifier(class_weight='auto', random_state=42))]

### prepare for pca
from sklearn.decomposition import PCA
pca = PCA()

### chain pca to feature selection
from sklearn.pipeline import FeatureUnion
combined_feature = []
for method in feature_selection:
    new_method = FeatureUnion([('pca', pca), method])
    name = method[0] + "_with_pca"
    combined_feature.append((name, new_method))

### update feature selection list
feature_selection += combined_feature

### prepare for classifiers
from sklearn.svm import LinearSVC
linear_svc = LinearSVC(class_weight='auto', penalty='l1', dual=False, random_state=42)

params_svc = {'linear_svc__C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10],
              'linear_svc__tol': [1e-4, 1e-3, 1e-2, 1],
              'linear_svc__max_iter': [1e3, 1e4]}

from sklearn.neighbors import KNeighborsClassifier
k_neighbors = KNeighborsClassifier(weights='distance', algorithm='auto')

params_kneighbors = {'k_neighbors__n_neighbors': [1, 3, 10],
                     'k_neighbors__leaf_size': [2, 5, 10, 30, 50, 100]}

### put all classifiers together
classifiers = [('linear_svc', linear_svc, params_svc),
               ('k_neighbors', k_neighbors, params_kneighbors)]

### train models on features list
model_sets, scores = trainModel(data_dict, features_list, 
                                             feature_selection=feature_selection,
                                             classifiers=classifiers,
                                             scalers = scalers,
                                             filename = 'model_metrix.csv')

### train models on new features list
### you don't need to run this to get the final result
### uncomment below to run the training

# model_sets_new, scores_new = trainModel(data_dict, new_features_list, 
#                                              feature_selection=feature_selection,
#                                              classifiers=classifiers,
#                                              scalers = scalers,
#                                              filename = 'model_metrix_new_features.csv')

### extract the pipeline
pipeline = model_sets[22][1]
tuning_score = scores[22]

### get the training and testing set
features_train, features_test, labels_train, labels_test = trainTestSplit(data_dict, features_list)

### prepare the cross validation
sss = StratifiedShuffleSplit(labels_train, n_iter=100, random_state=42)

### record time
t0 = time()
print "start tuning on extra tree..."

### set the parameters
params_extra_tree = {"extra_tree_with_pca__extra_tree__n_estimators": [1, 3, 10, 30, 100],
                     "extra_tree_with_pca__extra_tree__max_features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

### fit and search
estimator = GridSearchCV(pipeline, params_extra_tree, scoring='f1', cv=sss)
estimator.fit(features_train, labels_train)

### extract scores
score_extra_tree = estimator.grid_scores_

### get the best estimator
clf = estimator.best_estimator_

### check the model performance
crossValidate(data_dict, features_list, clf)

print "extra tree time used: ", time() - t0

### record time
t0 = time()
print "start tuning on pca..."

### set the parameters
params_pca = {"extra_tree_with_pca__pca__n_components": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              "extra_tree_with_pca__pca__whiten": [True, False]}

### fit and search
estimator = GridSearchCV(clf, params_pca, scoring='f1', cv=sss)
estimator.fit(features_train, labels_train)

### extract scores
score_pca = estimator.grid_scores_

### get the best estimator
clf = estimator.best_estimator_

### check the model performance
crossValidate(data_dict, features_list, clf)

print "pca time used: ", time() - t0

### prepare for the test
clf = clf
my_dataset = data_dict
features_list = features_list

### dump for testing
dump_classifier_and_data(clf, my_dataset, features_list)