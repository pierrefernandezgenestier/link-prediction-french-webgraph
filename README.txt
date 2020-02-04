The following libraries are required to run the python files for this project:
- networkx
- node2vec
- gensim
- nltk
- tensorflow
 -panda
-numpy
-matplotlib
-seaborn
-tqdm
-sklearn

You should run first feature_engineering.py file. This requires to have in the same folder the ‘training.txt’ and ‘test.txt’ files provided on Kaggle or provide their paths at the beginning of the file.

Then, the files producing the final result are:
- create_random_forest_model.py
- create_gradient_boosting_model.py
 
You can run either one to produce the predictions file that is uploaded to Kaggle. (The one we used is the random forest, but the gradient boosting exhibits similar performances)
 
This first file (feature_engineering.py) use Doc2Vec, Node2Vec, and Neural Network models that can be found in the following drive:
https://drive.google.com/drive/folders/1wB5hWt3i3ypmzEJ0WCDhUJ1y3QxbirqZ?usp=sharing
References to the path/name of the required models can be modified easily at the beginning of the feature_engineering.py. It takes a lot of time to train this models.
 
If you wish to generate these models yourself, you can do so using the corresponding python files:
- create_d2v_model.py
- create_n2v_model.py
- create_nn_modle.py
You can change the parameters used to create the models in the last function of the files (they are set to the values that we used for our models). Models created that way will be saved in the “models” folder. You can then use them for the feature_engineering.py by writing their name and path in the latter file.

The file SVM_method.py implement a SVM classifier, although we did not use it for prediction, you can still run it to build a SVM classifier.

