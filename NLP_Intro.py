
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("./kaggle/input/train.csv")
test_df = pd.read_csv("./kaggle/input/test.csv")

print(train_df[train_df["target"] == 0]["text"])

print(train_df[train_df["target"] == 0]["text"].values[1])

