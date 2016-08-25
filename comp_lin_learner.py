# In this exercise we'll examine a learner which has high bias, and is incapable of
# learning the patterns in the data.
# Use the learning curve function from sklearn.learning_curve to plot learning curves
# of both training and testing error. Use plt.plot() within the plot_curve function
# to create line graphs of the values.

from sklearn.linear_model import LinearRegression
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np

size = 1000
cv = KFold(size,shuffle=True)
score = make_scorer(explained_variance_score)

X = np.reshape(np.random.normal(scale=2,size=size),(-1,1))
y = np.array([[1 - 2*x[0] +x[0]**2] for x in X])

def plot_curve():
    reg = LinearRegression()
    reg.fit(X,y)
    print reg.score(X,y)
    
    # TODO: Create the learning curve with the cv and score parameters defined above.
    #print learning_curve(estimator=LinearRegression(), X=X, y=y, cv=KFold)
    train_sizes, train_scores, test_scores = learning_curve(LinearRegression(),X,y,cv=cv,scoring=score, train_sizes=np.linspace(.1, 1.0, 5))
    

    # TODO: Plot the training and testing curves.
    # print train_sizes, train_scores, valid_scores 
    plt.plot(train_sizes, np.mean(train_scores, axis=1))
    plt.plot(train_sizes, np.mean(test_scores, axis=1))
    
    # Sizes the window for readability and displays the plot.
    plt.ylim(-.1,1.1)
    plt.show()
