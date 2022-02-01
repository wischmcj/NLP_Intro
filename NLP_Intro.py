
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



 #TF idf to improve scores 
 #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
 
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


train_df_bow =train_df["text"]
 

# for x in train_df["bow"]:
#     i=0
#     #train_df[i,'BOW'] = 
#     print(type(x.split(' ')))
#     #i=i+1
 
def preprocess_text(document):
    for text in document:
   # Tokenise words while ignoring punctuation
        tokeniser = RegexpTokenizer(r'\w+')
        tokens = tokeniser.tokenize(text)
     
    return tokens
 
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


# bow_df = pd.DataFrame().values.tolist()

# for x in train_df["text"]:
#     i=0
# #print(preprocess_text(x))
#     bow_df.insert(i, x.split())
#     i=i+1
  

tfIdfVectorizer=TfidfVectorizer(use_idf=True,stop_words=token_stop, 
                              tokenizer=tokenizer )
tfIdf = tfIdfVectorizer.fit_transform(train_df["text"])
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print (df.head(25))


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

# def computeTF(train_vectors, bagOfWords):
#  ##PRedict answers based on model
#  clf.fit(train_vectors, train_df["target"])
#  ##Submit
#  #sample_submission = pd.read_csv("./kaggle/submission.csv")
# # sample_submission["target"] = clf.predict(test_vectors)
# # print(sample_submission.head())

