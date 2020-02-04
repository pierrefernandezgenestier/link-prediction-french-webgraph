import networkx as nx
import pandas as pd
import numpy as np
import gensim
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow.keras as keras
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

## Before running this code check that the path for the 2 original file is correct as well
# as the path for the 3 model below

## path for the 2 original data file
train_file = "training.txt"
test_file = "testing.txt"

#Path for the model engineered in
# create_d2v_model.py, create_n2v_model.py, create_nn_model.py
doc2vec_model_name = 'final_code/models/models/doc2vec_model_20_3_40'
node2vec_model_name = 'final_code/models/models/node2vec_model_64_15_100'
neural_network_name = 'final_code/models/models/nn_model_32_20_1_dir30.h5'

# loading model
doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(doc2vec_model_name)
model = gensim.models.word2vec.Word2Vec.load(node2vec_model_name)
nn_model = keras.models.load_model(neural_network_name)

##From now you can run the entire file without changing any other parameter

# to be able to visualize progression of data added to data frame
tqdm.pandas()

print('>>> Loading data set...')
# read the training.txt file
training = pd.read_csv(train_file, sep=' ', header=None)
training_list = training.values.tolist()

print('>>> Creating graph...')
# creating the graph
G = nx.Graph()
for line in training_list:
    if line[2] == 1:
        G.add_edge(line[0], line[1])
    if not G.has_node(line[0]):
        G.add_node(line[0])
    if not G.has_node(line[1]):
        G.add_node(line[1])


print("Number of nodes : " + str(G.number_of_nodes()))
print("Number of edges : " + str(G.number_of_edges()))

# we keep a small subset of the graph for training
# it is not necessary to keep the entIre graph as it will not impact our model performance later
training_reduced = training.sample(frac=0.1, random_state=42)  # We keep 10%
training_reduced.columns = ['source', 'target', 'Y']
print("Size of training reduced :", training_reduced.shape)

# read the test data
test = pd.read_csv(test_file, sep=' ', header=None)
test.columns = ['source', 'target']

# we calculate some coefficient related to the graph's topology
# more detail on each coef can be found in our report
print('>>> Adding Jaccard Coefficient to training...')
training_reduced['jaccard_coefficient'] = training_reduced.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.jaccard_coefficient(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding Preferential Attachment to training...')
training_reduced['preferential_attachment'] = training_reduced.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.preferential_attachment(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding Resource Allocation Index to training...')
training_reduced['resource_allocation_index'] = training_reduced.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.resource_allocation_index(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding Adamic Adar Index to training...')
training_reduced['adamic_adar_index'] = training_reduced.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.adamic_adar_index(G, [(row.source, row.target)])][0][2], axis=1)

page_rank = nx.pagerank_scipy(G)
print('>>> Adding PageRank Source Index to training...')
training_reduced['pagerank_source'] = training_reduced.progress_apply(lambda row: page_rank[row.source],axis=1)

print('>>> Adding PageRank Target Index to training...')
training_reduced['pagerank_target'] = training_reduced.progress_apply(lambda row: page_rank[row.target],axis=1)

# creating gradient boosting regressor to add its prediction to the data frame
# compute training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(training_reduced.drop(['source', 'target', 'Y'], axis=1),
                                     training_reduced.Y, test_size=0.31)
GBR = GradientBoostingRegressor()
GBR.fit(X_train, Y_train)

print('>>> Adding Gradient Boosting Regression predicitons to training...')
training_reduced['boost_reg']=training_reduced.progress_apply(
    lambda row: GBR.predict([[row.jaccard_coefficient, row.preferential_attachment, row.resource_allocation_index,
    row.adamic_adar_index, row.pagerank_source, row.pagerank_target]])[0], axis=1)

# we calculate the same coefficient on the test dataset
print('>>> Adding Jaccard Coefficient to test...')
test['jaccard_coefficient'] = test.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.jaccard_coefficient(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding Preferential Attachment to test...')
test['preferential_attachment'] = test.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.preferential_attachment(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding Resource Allocation Index to test...')
test['resource_allocation_index'] = test.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.resource_allocation_index(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding Adamic Adar Index to test...')
test['adamic_adar_index'] = test.progress_apply(
    lambda row: [[u, v, p] for u, v, p in nx.adamic_adar_index(G, [(row.source, row.target)])][0][2], axis=1)

print('>>> Adding PageRank Source Index to test...')
test['pagerank_source'] = test.progress_apply(lambda row: page_rank[row.source],axis=1)

print('>>> Adding PageRank Target Index to test...')
test['pagerank_target'] = test.progress_apply(lambda row: page_rank[row.target],axis=1)

print('>>> Adding Gradient Boosting Regression predicitons to test...')
test['boost_reg']=test.progress_apply(
    lambda row: GBR.predict([[row.jaccard_coefficient, row.preferential_attachment, row.resource_allocation_index,
    row.adamic_adar_index, row.pagerank_source, row.pagerank_target]])[0], axis=1)

print('>>> Creating Neural Network based on doc2vec and node2vec features...')
# we now integrate link probability calculated by the neural network engineered in :
# create_nn_model.py, create_d2v_model.py, create_n2v_model.py


model.wv.get_vector(str(0))
model.wv.most_similar(str(0))
np.concatenate((model.wv.get_vector(str(0)), doc2vec_model.docvecs[0]))

def create_with_textdata_XY(dataset):
    X=[]
    Y=[]
    for ii in range(len(dataset)):
        X.append(np.array(
            [np.concatenate((model.wv.get_vector(str(dataset[ii][0])),doc2vec_model.docvecs[dataset[ii][0]])),
             np.concatenate((model.wv.get_vector(str(dataset[ii][1])),doc2vec_model.docvecs[dataset[ii][1]]))
            ]))
        if len(dataset[ii])>2 :
            Y.append(dataset[ii][2])
    return np.array(X),np.array(Y)

ds_train = training_reduced[['source','target','Y']].values
X_train, Y_train = create_with_textdata_XY(ds_train)


# class prediction unsing the neural network and F1 score on the training set
Y_pred_classes = nn_model.predict_classes(X_train)
print('Test f1 score number of neural network ', f1_score(Y_train, Y_pred_classes))

# add link probability feature to training
Y_float = nn_model.predict(X_train)
training_reduced['nn_proba'] = Y_float

# add link probability feature to test
ds_test = test[['source', 'target']].values
X_test, Y_test= create_with_textdata_XY(ds_test)
Y_test_pred = nn_model.predict(X_test)
test['nn_proba'] = Y_test_pred

# below we create the doc2vec_similarity coefficient, which takes a lot of time to run (more than 10 hours)
"""
# we create the same list of tokens for each webpage as we did in create_d2v_model.py
data = glob.glob("node_information/text/*.txt")
list_of_tokens = []
for i in range(len(data)):
    if(i % 1000 == 0):
        print(i/len(data), '%')
    data_path = 'node_information/text/'+str(i)+'.txt'
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        lines = [line.strip("\n") for line in lines]
        unique_string = ' '.join(lines)
        tokens = preprocess_string(unique_string, CUSTOM_FILTERS)
        list_of_tokens.append(tokens)

# adding doc2vec similarity takes a lot of time and one might want to comment the two lines of 
# code below, predictions will not be changed significantly
training_reduced['doc2vec_similarity']=training_reduced.progress_apply(
    lambda row: sklearn.metrics.pairwise.cosine_similarity(doc2vec_model.infer_vector(list_of_tokens[int(row.source)]).reshape(1, -1), 
    doc2vec_model.infer_vector(list_of_tokens[int(row.target)]).reshape(1, -1))[0][0], axis=1)
test['doc2vec_similarity']=test.progress_apply(
    lambda row: sklearn.metrics.pairwise.cosine_similarity(doc2vec_model.infer_vector(list_of_tokens[int(row.source)]).reshape(1, -1), 
    doc2vec_model.infer_vector(list_of_tokens[int(row.target)]).reshape(1, -1))[0][0], axis=1)
"""

# saving traning.csv and test.csv
test.to_csv("test.csv", header=True)
training_reduced.to_csv("training.csv", header=True)

# visualizing the correlation between the features

plt.figure(figsize=(14,12))
sns.heatmap(training_reduced.drop(['source', 'target', 'Y'], axis=1).corr(),
            vmax=0.5,
            square=True,
            annot=True)
plt.show()


