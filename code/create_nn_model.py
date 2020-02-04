import tensorflow.keras as keras
import seaborn as sns
import numpy as np
import pandas as pd
import gensim

###################################
# Load models and preprocess data #
###################################

print(">>> Loading Doc2Vec and Node2Vec models...")

# loading node2vec model
n2v_model = gensim.models.word2vec.Word2Vec.load('models/node2vec_model_64_15_100')
# loading doc2vec model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec_model_20_3_40')

print(">>> Creating embedded dataset...")
# compute training set
ds_train = pd.read_csv("training.txt", header=None, delimiter=' ').values

# creates an embedding of each node as a 64-coordinate-vector given by the node2vec model
def create_XY(dataset):
    X = []
    Y = []
    for ii in range(len(dataset)):
        X.append(np.array(
            [np.array(n2v_model.wv.get_vector(str(dataset[ii][0]))),
             np.array(n2v_model.wv.get_vector(str(dataset[ii][1])))
             ]))
        Y.append(dataset[ii][2])
    return np.array(X), np.array(Y)

# creates an embedding of each node as a (64+20)-coordinate-vector given by the node2vec and the doc2vec models
def create_with_textdata_XY(dataset):
    X = []
    Y = []
    for ii in range(len(dataset)):
        X.append(np.array(
            [np.concatenate((n2v_model.wv.get_vector(str(dataset[ii][0])), d2v_model.docvecs[dataset[ii][0]])),
             np.concatenate((n2v_model.wv.get_vector(
                 str(dataset[ii][1])), d2v_model.docvecs[dataset[ii][1]]))
             ]))
        Y.append(dataset[ii][2])
    return np.array(X), np.array(Y)


X, Y = create_with_textdata_XY(ds_train)

###################################
# Create and train neural network #
###################################

print(">>> Creating and training neural network...")

layers = [
    keras.layers.Flatten(input_shape=(2, 84)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
    keras.layers.Dense(1, activation="sigmoid") # to get a 0-1 probability as a prediction
]

nn_model = keras.Sequential(layers)

nn_model.compile(optimizer='sgd',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# callback to stop when val_loss increases, prevent overfitting
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = nn_model.fit(X, Y, batch_size=64, epochs=40, validation_split=0.2, callbacks=[es_callback])

print("")
print(">>> Saving model...")
nn_model.save("models/nn_model_32_20_1.h5")