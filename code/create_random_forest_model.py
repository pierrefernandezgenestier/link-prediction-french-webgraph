import pandas as pd
import numpy as np
import csv
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# creating a random forest with no special parameters
RF_classifer = RandomForestClassifier()

#####################################################
# Starting of Cross Validation to find optimal      #
# parameters for the Random Forest                  #
#####################################################      

print('>>> Starting Randomnized Search Cross Validation...')

# create random grid parameters to be explored
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1800, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 9]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

grid_random = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# random search of parameters, parameters of the randomnized search can be modified
RF_random = RandomizedSearchCV(estimator = RF_classifer, param_distributions = grid_random, 
n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# fit the randomnized search CV
RF_random.fit(X_train, Y_train)

print('>>> Starting Grid Search Cross Validation to find optimal Random Forest Classfifier...')

# create the parameter grid based on the results of random search 
# varying slightly the optimised parameters found
grid_parameters = {
    'bootstrap': [RF_random.best_params_['bootstrap']],
    'max_depth': [RF_random.best_params_['max_depth']-2,
                  RF_random.best_params_['max_depth'],
                  RF_random.best_params_['max_depth']+2],
    'max_features': [RF_random.best_params_['max_features']],
    'min_samples_leaf': [RF_random.best_params_['min_samples_leaf'],
                  RF_random.best_params_['min_samples_leaf']+2],
    'min_samples_split': [RF_random.best_params_['min_samples_split'],
                  RF_random.best_params_['min_samples_split']+1],
    'n_estimators': [RF_random.best_params_['n_estimators']-100,
                  RF_random.best_params_['n_estimators'],
                  RF_random.best_params_['n_estimators']+100]
}

# create a new random forest classifier
RF = RandomForestClassifier()
# create grid search model
grid_search = GridSearchCV(estimator = RF, param_grid = grid_parameters, 
                          cv = 3, n_jobs = -1, verbose = 2)
# train grid search model
grid_search.fit(X_train, Y_train)

#####################################################
#   Create and train Optimized Random Forest        #
#####################################################  

print('>>> Creating optimized Random Forest Classifier...')

# best Random Forest Classifer
best_RF = grid_search.best_estimator_

print('>>> Training the Random Forest Classifier...')

# train selected RF with data
best_RF.fit(X_train, Y_train)

print('F1 score number ', f1_score(Y_test, best_RF.predict(X_test)))

with open('best_RF.pkl', 'wb') as file:
    pickle.dump(best_RF, file)

# we produce the csv prediction file to be uploaded on Kaggle

# loading the test DataFrames containing the features
test = pd.read_csv('test.csv', sep=',')

# drop 1st column which is automatically added when the file has been saved in feature_engineering.py
test = test.drop(training.columns[[0]], axis=1)
test = test.drop(['source', 'target'], axis=1)
predictions = best_RF.predict(test)
predictions = zip(range(len(predictions)), predictions)

#Write the prediction in a csv file
with open("predictions_random_forest.csv", "w") as pred:
    csv_out = csv.writer(pred, lineterminator='\n')
    csv_out.writerow(['id', 'predicted'])
    for row in predictions:
        csv_out.writerow(row)