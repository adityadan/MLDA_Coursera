from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import pydotplus
import matplotlib.pylab as plt
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
os.chdir("C:/Users/Aditya/mlda")

AH_data = pd.read_csv("Cat_stats.csv")
data_clean = AH_data.dropna()
data_clean.dtypes
data_clean.describe()

predictors = data_clean[['Body_length', 'Tail_length', 'Height', 'Weight', 'Tail_texture', 'Coat_colour']]

targets = data_clean.Wildcat
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size = .4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train, tar_train)
predictions=classifier.predict(pred_test)
sklearn.metrics.confusion_matrix(tar_test, predictions)

from sklearn import tree
from io import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
graph.write_png("cat.png")