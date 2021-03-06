#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
From https://github.com/plaidml/plaidml/blob/master/README.md.
"""

import numpy as np
import os
import time

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'  # noqa: E402 module level import not at top

import keras.applications as kapp
from keras.datasets import cifar10


def test_plaidml_keras_vgg19():
    (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
    batch_size = 8
    x_train = x_train[:batch_size]
    x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
    model = kapp.VGG19()
    model.compile(
        optimizer='sgd', loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print('Running initial batch (compiling tile program)')
    model.predict(x=x_train, batch_size=batch_size)

    # Now start the clock and run 10 batches
    print('Timing inference...')
    start = time.time()
    for i in range(10):
        model.predict(x=x_train, batch_size=batch_size)
    print('Ran in {} seconds'.format(time.time() - start))


if __name__ == '__main__':
    test_plaidml_keras_vgg19()
