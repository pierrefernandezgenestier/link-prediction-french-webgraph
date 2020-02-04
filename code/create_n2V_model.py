from node2vec import Node2Vec
import networkx as nx
import pandas as pd
import numpy as np


##############################################################
# First read and create the graph using the training dataset #
##############################################################

print(">>> Creating nodes and edges...")

training = pd.read_csv('training.txt', sep=' ', header=None)
training_list = training.values.tolist()
G = nx.Graph()
# G = nx.DiGraph() #to create a directed graph, revealed to have worse results
for line in training_list:
    if line[2] == 1:
        G.add_edge(line[0], line[1])
    if not G.has_node(line[0]):
        G.add_node(line[0])
    if not G.has_node(line[1]):
        G.add_node(line[1])

print("Number of nodes: " + str(G.number_of_nodes()))
print("Number of edges: " + str(G.number_of_edges()))
print("")

#############################
# Create the node2vec model #
#############################

# Using node2vec : Python3 implementation of the node2vec algorithm Aditya Grover, Jure Leskovec and Vid Kocijan.
# node2vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec.
# ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016.

def create_node2vec_model(dimensions, walk_length, num_walks):
    print(">>> Creating the model...")
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1)

    print(">>> Fitting the model...")
    node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Save embeddings for later use
    node2vec_model.wv.save_word2vec_format("node2vec_wv_model_"+str(dimensions)+"_"+str(walk_length)+"_"+str(num_walks))
    # Save model for later use
    node2vec_model.save("models/node2vec_model_"+str(dimensions) +"_"+str(walk_length)+"_"+str(num_walks))

create_node2vec_model(64, 15, 100)
