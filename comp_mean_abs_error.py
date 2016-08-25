import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(X, y)

reg = DecisionTreeRegressor()
reg.fit(features_train, labels_train)
print "Decision Tree mean absolute error: {:.2f}".format(mae(reg.predict(features_test),labels_test))
dtmae = mae(reg.predict(features_test), labels_test)

reg = LinearRegression()
reg.fit(features_train, labels_train)
print "Linear regression mean absolute error: {:.2f}".format(mae(reg.predict(features_test),labels_test))
lrmae = mae(reg.predict(features_test), labels_test)

results = {
 "Linear Regression": lrmae,
 "Decision Tree": dtmae
}
