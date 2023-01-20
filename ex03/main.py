import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

inputs = np.load('dataset/inputs.npy')
labels = np.load('dataset/labels.npy')

print('prediction of the winner of a nba game')

def logistic_regression():
    print('With LogisticRegression:')
    start = time.time()
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.30)

    print('starting model fitting')
    model.fit(x_train, np.ravel(y_train))

    print('calculating accuracy...')

    accuracy = model.score(x_test, y_test)

    end = time.time()
    print('Took %.2f sec' % (end - start))
    print('Accuracy is %.3f' % np.round(accuracy, 3))


logistic_regression()
