import pickle
import glob

import gensim
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_non_alphanum, strip_punctuation, strip_short

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


#########################################################
# Preprocess text data form the node_information folder #
#########################################################

print(">>> Processing text data...")

# filters that are applied to the text
# putting in lowercase, removing punctuation and non alphanumeric characters, removing short words (length<3)
CUSTOM_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_non_alphanum, strip_short]

# Splitting the text into tokens
print("Tokenizing text...")
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

# Removing stop words from the tokens
print("Removing stop words...")
stop_words = stopwords.words('english') + stopwords.words('french')
list_of_tokens = [[w for w in ws if not w in stop_words] for ws in list_of_tokens]

def save_tokens(list_of_tokens):
    # to save the list of tokens
    with open("list_of_tokens.txt", "wb") as lot:
        pickle.dump(list_of_tokens, lot)

def load_tokens():
    # to load the list of tokens
    with open("list_of_tokens.txt", "rb") as fp:
        list_of_tokens = pickle.load(fp)
    return list_of_tokens

print("")

############################
# Create the doc2vec model #
############################

# Creating tagged document for the doc2vec generator
def create_tagged_document(list_of_tokens):
    for i, tokens in enumerate(list_of_tokens):
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_data = list(create_tagged_document(list_of_tokens))

# Using doc2vec : Python3 implementation of the doc2vec algorithm Quoc Le, Tomas Mikolov.
# Distributed Representations of Sentences and Documents.
# Google Inc, 1600 Amphitheatre Parkway, Mountain View, CA 94043

def create_doc2vec_model(vector_size, min_count, epochs):
    print(">>> Creating Doc2Vec model...")
    print("")
    # create the model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    # create the vocabulary
    model.build_vocab(train_data)
    # training the model
    model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
    # saving model
    model.save("models/doc2vec_model_"+str(vector_size) +"_"+str(min_count)+"_"+str(epochs))


create_doc2vec_model(20, 3, 40)
