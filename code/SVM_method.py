import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import svm


#reading the training file with all its features
train_file_name = 'training.csv'
df= pd.read_csv(train_file_name)

#defining the different rando state we will use for train test split
random_state = [23, 67, 92, 52, 79]

#Using StandardScaler to scale the data as SVM algorithms are not scale invariant
scale = StandardScaler()
feature = ['jaccard_coefficient','preferential_attachment','resource_allocation_index','adamic_adar_index','pagerank_source','pagerank_target','boost_reg','nn_proba']
df[feature]=pd.DataFrame(scale.fit_transform(df[feature].values), columns= feature)

#Using cross validation to decide on the best Kernel
for i in range (5):
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(['source', 'target', 'Y'], axis=1), df.Y,
                                                        test_size=0.3, random_state=random_state[i])
    for kernel in ('poly', 'rbf', 'sigmoid', 'linear'):
        clf = svm.SVC(kernel = kernel)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print(kernel, ' ',i,'  Test f1 score number ', f1_score(Y_test, Y_pred))
        print(kernel, ' ', i,'  Confusion matrix')
        print(confusion_matrix(Y_test, Y_pred))
        print(kernel, ' ', i,'  Classifcation Report')
        print(classification_report(Y_test, Y_pred))

### The best kernel is linear
### now we try to optimize C, the regularization parameter

C = [0.1, 0.5, 1, 1.5, 2, 3]
for j in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(['source', 'target', 'Y'], axis=1), df.Y,
                                                        test_size=0.3, random_state=random_state[j])
    for i in range (len(C)):
        clf = svm.SVC(kernel='linear', C=C[i])
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print("C=", C[i], '  Test f1 score number ', f1_score(Y_test, Y_pred))
        print("C=", C[i], '  Confusion matrix')
        print(confusion_matrix(Y_test, Y_pred))
        print("C=", C[i], '  Classifcation Report')
        print(classification_report(Y_test, Y_pred))

### Yet, other methods like Random Forest or logistic Regression gaves us better result (better F1 score)
### Thus we did not use SVM in our final model