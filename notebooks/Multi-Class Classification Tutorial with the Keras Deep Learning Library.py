#!/usr/bin/env python
# -*- coding: utf-8 -*-
# In[1]:
"""Multi-Class Classification Tutorial with the Keras Deep Learning Library."""
# In[13]:
# From https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/.
# In[12]:
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
# In[3]:
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# In[4]:
get_ipython().system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data --output-document ../data/iris.csv')
# In[5]:
# load dataset
dataframe = pd.read_csv('../data/iris.csv', header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
# In[6]:
X.shape, Y.shape
# In[7]:
# encode class values as integers
encoder = LabelEncoder()
_ = encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# In[16]:
encoded_Y
# In[15]:
dummy_y[0]
# In[8]:
# define baseline model
def baseline_model():
    """Create baseline model."""

    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# In[9]:
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# In[10]:
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# In[11]:
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print('Baseline: {:.2f}% ({:.2f}%)'.format(results.mean() * 100, results.std() * 100))
