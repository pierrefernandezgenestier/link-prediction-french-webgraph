import pandas as pd
import numpy as np
import csv
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


print('>>> Loading data...')

# loading the training DataFrames containing the features
training = pd.read_csv('training.csv', sep=',')

# drop 1st column which is automatically added when the file has been saved in feature_engineering.py
training = training.drop(training.columns[[0]], axis=1)

# compute training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(training.drop(['source', 'target', 'Y'], 
axis=1), training.Y, test_size=0.35)

# creating a gradient boosting classifier with no special parameters
GD_Classifier = GradientBoostingClassifier()

#####################################################
# Starting of Cross Validation to find optimal      #
# parameters for the Gradient Boosting classifier   #
#####################################################      

print('>>> Starting Randomnized Search Cross Validation...')

# create random grid parameters to be explored
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
max_features = ['auto', 'log2']
max_features.append(None)
max_depth = [int(x) for x in np.linspace(2, 10, num = 5)]
min_samples_split = [2, 3, 5]
min_samples_leaf = [1, 2, 4, 6, 8]
loss = ['deviance', 'exponential']

grid_random = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'loss': loss}

# random search of parameters, parameters of the randomnized search can be modified
GB_random = RandomizedSearchCV(estimator = GD_Classifier, param_distributions = grid_random, 
n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# fit the randomnized search CV
GB_random.fit(X_train, Y_train)

print('>>> Starting Grid Search Cross Validation to find optimal Gradient Boosting Classfifier...')

# create the parameter grid based on the results of random search 
# varying slightly the optimised parameters found
grid_parameters = {
    'loss': [GB_random.best_params_['loss']],
    'max_depth': [GB_random.best_params_['max_depth']-1,
                  GB_random.best_params_['max_depth'],
                  GB_random.best_params_['max_depth']+1],
    'max_features': [GB_random.best_params_['max_features']],
    'min_samples_leaf': [GB_random.best_params_['min_samples_leaf'],
                  GB_random.best_params_['min_samples_leaf']+2],
    'min_samples_split': [GB_random.best_params_['min_samples_split'],
                  GB_random.best_params_['min_samples_split']+1],
    'n_estimators': [GB_random.best_params_['n_estimators']-100,
                  GB_random.best_params_['n_estimators'],
                  GB_random.best_params_['n_estimators']+100]
}

# create a new gradient boosting classifier classifier
GDC = GradientBoostingClassifier()
# create grid search model
grid_search = GridSearchCV(estimator = GDC, param_grid = grid_parameters, 
                          cv = 3, n_jobs = -1, verbose = 2)
# train grid search model
grid_search.fit(X_train, Y_train)

####################################################################
#   Create and train Optimized Gradient Boosting Classifier        #
####################################################################

print('>>> Creating optimized Gradient Boosting Classifier...')

# best Gradient Boosting Classifer
best_GDC = grid_search.best_estimator_

print('>>> Training the Gradient Boosting Classifier...')

# train selected GDC with data
best_GDC.fit(X_train, Y_train)

print('F1 score number ', f1_score(Y_test, best_GDC.predict(X_test)))

with open('best_GDC.pkl', 'wb') as file:
    pickle.dump(best_GDC, file)

# we produce the csv prediction file to be uploaded on Kaggle

# loading the test DataFrames containing the features
test = pd.read_csv('test.csv', sep=',')

# drop 1st column which is automatically added when the file has been saved in feature_engineering.py
test = test.drop(training.columns[[0]], axis=1)
test = test.drop(['source', 'target'], axis=1)
predictions = best_GDC.predict(test)
predictions = zip(range(len(predictions)), predictions)

#Write the prediction in a csv file
with open("predictions_gradient_boosting.csv", "w") as pred:
    csv_out = csv.writer(pred, lineterminator='\n')
    csv_out.writerow(['id', 'predicted'])
    for row in predictions:
        csv_out.writerow(row)