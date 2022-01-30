
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("./kaggle/input/train.csv")
test_df = pd.read_csv("./kaggle/input/test.csv")

# View of train data 

# print(train_df[train_df["target"] == 0]["text"])

# print(train_df[train_df["target"] == 0]["text"].values[1])

count_vectorizer = feature_extraction.text.CountVectorizer()

# Vectorizer test 
# example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

# print(example_train_vectors[0].todense().shape)
# print(example_train_vectors[0].todense())
# print(example_train_vectors[0])

# Converte text column to feature array
train_vectors = count_vectorizer.fit_transform(train_df["text"])

print(train_vectors)
## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 
# i.e. that the train and test vectors use the same set of tokens.
test_vectors = count_vectorizer.transform(test_df["text"])
clf = linear_model.RidgeClassifier()

# Calculate model accuracy with various subsets of the labled data  - F1 score 

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
print(scores)


 #TF idf to improve scores 
 
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


train_df_text = train_df["text"]
train_df_bow = train_df_text
for x in train_df_text:
    i=0
    train_df_bow[i] = x.split(' ')
    i=i+1
    
print(train_vectors[1].todense())


#def computeTF(train_vectors, bagOfWords):




# ##PRedict answers based on model
# clf.fit(train_vectors, train_df["target"])

# ##Submit
# sample_submission = pd.read_csv("./kaggle/submission.csv")
# sample_submission["target"] = clf.predict(test_vectors)
# print(sample_submission.head())

