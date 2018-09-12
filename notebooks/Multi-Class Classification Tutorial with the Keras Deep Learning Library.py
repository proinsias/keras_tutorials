#!/usr/bin/env python
# coding: utf-8
"""Multi-Class Classification Tutorial with the Keras Deep Learning Library."""
# From https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/.
from IPython import get_ipython
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
get_ipython().system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data --output-document\
    ../data/iris.csv')
# load dataset
dataframe = pd.read_csv("../data/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
X.shape, Y.shape
# encode class values as integers
encoder = LabelEncoder()
_ = encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# define baseline model
def baseline_model():
    """Create baseline model."""
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
