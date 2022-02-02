
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import linear_kernel
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords

train_df = pd.read_csv("./kaggle/input/train.csv")
test_df = pd.read_csv("./kaggle/input/test.csv")

# View of train data 

# print(train_df[train_df["target"] == 0]["text"])

# print(train_df[train_df["target"] == 0]["text"].values[1])

count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])
# Vectorizer test 
# example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

# print(example_train_vectors[0].todense().shape)
# print(example_train_vectors[0].todense())
# print(example_train_vectors[0])

# Convert text column to feature array
## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])

#print(test_vectors)

clf = linear_model.RidgeClassifier()
# Calculate model accuracy with various subsets of the labled data  - F1 score 
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
#print(scores)
  
    
stop_words = set(stopwords.words('english')) 
#https://gist.github.com/4OH4/f727af7dfc0e6bb0f26d2ea41d89ee55       
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in stop_words]


tokenizer=LemmaTokenizer()
token_stop = tokenizer(' '.join(stop_words))

# for x in train_df["text"]:
#     i=0
#     #print(preprocess_text(x))
#     bow_df[i] = preprocess_text(x)
#     i=i+1

 #TF idf to improve scores 
 #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
 
def preprocess_text(text):
   # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)w
    
    lemmas= word_tokenize(text)
    lemmas= WordNetLemmatizer()
   
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]
    
    # Remove stop words
    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords



#bow_df =  pd.DataFrame(train_df["text"].apply(preprocess_text))

# Create an instance of TfidfVectorizer
vectoriser = TfidfVectorizer(analyzer=preprocess_text)
# Fit to the data and transform to feature matrix
X_train = vectoriser.fit_transform(train_df["text"])
# Convert sparse matrix to dataframe
X_train = pd.DataFrame.sparse.from_spmatrix(X_train)
# Save mapping on which index refers to which words
col_map = {v:k for k, v in vectoriser.vocabulary_.items()}
# Rename each column using the mapping
for col in X_train.columns:
    X_train.rename(columns={col: col_map[col]}, inplace=True)
print(X_train)






#https://towardsdatascience.com/introduction-to-nlp-part-1-preprocessing-text-in-python-8f007d44ca96
# tfIdfVectorizer=TfidfVectorizer(use_idf=True,stop_words=token_stop, tokenizer=tokenizer )
# tfIdf = tfIdfVectorizer.fit_transform(train_df["text"])
# df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print (df.head(25))


# countVectorizer = CountVectorizer()
# tfIdfTransformer = TfidfTransformer(use_idf=True)
# wordCount = countVectorizer.fit_transform(train_df["text"])
# newTfIdf = tfIdfTransformer.fit_transform(wordCount)

# df = pd.DataFrame(newTfIdf[0].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF"])

# df = df.sort_values('TF-IDF', ascending=False)
# print (df.head(25))




# #https://towardsdatascience.com/tf-idf-explained-and-python-sklearn-implementation-b020c5e83275

# tfIdfVectorizer=TfidfVectorizer(use_idf=True)
# tfIdf = tfIdfVectorizer.fit_transform(train_df["text"])
# df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print (df.head(25))

## The same as the above but a different way
##adding in an analyser as opposed to stop words and a tokenizer
