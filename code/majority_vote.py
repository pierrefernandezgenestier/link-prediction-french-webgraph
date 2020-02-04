import pandas as pd
import numpy as np
import csv

#takes a list of 1 and 0 and returns the majority vote 
def majority_vote(votes):
    count0 = 0
    count1 =0
    for vote in votes:
        if vote==0:
            count0 +=1
        else:
            count1 +=1 
            
    return 1 if count1>count0 else 0

#takes a list of prediction files and returns the majority vote for each row
def majority_vote_from_preds(preds):
    Y_pred=[]
    for i in range(len(preds[0])):
        votes=[]
        for pred in preds:
            votes.append(int(pred[i][1]))
        Y_pred.append(majority_vote(votes))
    return np.array(Y_pred)

def count1(Y):
    count=0
    for i in range (len(Y)):
        if Y[i]==1 :
            count += 1
    return count/len(Y)

#majority_vote between best predictions
pred_0 = pd.read_csv('all_pred\predictions_0.31_best_grid_4coef.csv', sep=',', header=None).values[1:]
pred_1 = pd.read_csv('all_pred\predictions_11_RF_5coefboostreg.csv', sep=',', header=None).values[1:]
pred_2 = pd.read_csv('all_pred\predictions_12_nn_with_node2vec.csv', sep=',', header=None).values[1:]
pred_3 = pd.read_csv('all_pred\predictions_13_nn_with_n2v_d2v.csv', sep=',', header=None).values[1:]
pred_4 = pd.read_csv('all_pred\predictions_21.csv', sep=',', header=None).values[1:]
pred_5 = pd.read_csv('all_pred\predictions19_rf_with_nn_coef.csv', sep=',', header=None).values[1:]
pred_6 = pd.read_csv('all_pred\predictions20_logreg_with_nn_proba.csv', sep=',', header=None).values[1:]
pred_7 = pd.read_csv('all_pred\predictions21_rf_dirGraph.csv', sep=',', header=None).values[1:]
pred_8 = pd.read_csv('all_pred\predictions22_rf_dirGraph.csv', sep=',', header=None).values[1:]

Y_pred = majority_vote_from_preds([pred_0,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_8])

with open("predictions_majority_vote.csv", "w") as pred:
    csv_out = csv.writer(pred, lineterminator='\n')
    csv_out.writerow(['id', 'predicted'])
    for i in range(len(Y_pred)):
        csv_out.writerow([i,Y_pred[i]])
