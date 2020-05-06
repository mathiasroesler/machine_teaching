#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the training functions.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from init_fct import *
from strategies import *


def classic_training(train_data, train_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Trains a student model fitted with the train set and
    tested with the test set. 
    Input:  train_data -> tf.tensor[float32], list
                of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
            test_data -> tf.tensor[float32], list
                of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[float32], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracies -> np.array[float32], accuracies for the training and the test. 
    """

    model = model_init(train_data[0].shape) # Declare student model
    
    hist = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.array(np.append(hist.history.get('accuracy'), [test_score[1]]), dtype=np.float32)


def curriculum_training(train_data, train_labels, test_data, test_labels, class_nb, batch_size=32, epochs=10):
    """ Trains the model using a two step curriculum. 
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, label.
            test_data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
            class_nb -> int, class selected for the one vs all.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracies -> np.array[float], accuracy at each epoch of training and
                for the test set.
    """

    max_class_nb = np.max(train_labels) # Highest class number

    try:
        assert(isinstance(class_nb, int))
        assert(class_nb >= 0)
        assert(max_class_nb >= class_nb)

    except AssertionError:
        print("Error in function class_training: class_nb must be an integer smaller than the number of classes.")
        exit(1)

    model = model_init(train_data[0].shape) # Student model to train
    acc_hist = np.array([], dtype=np.float32)  # List for the accuracies
    easy_indices, hard_indices = two_step_curriculum(train_data, train_labels)

    # Convert train labels to one hot labels
    train_labels = prep_data(train_labels, class_nb)

    # Train model with easy then hard examples
    hist = model.fit(tf.gather(train_data, easy_indices), tf.gather(train_labels, easy_indices), batch_size=batch_size, epochs=epochs//2)
    acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 
    hist = model.fit(tf.gather(train_data, hard_indices), tf.gather(train_labels, hard_indices), batch_size=batch_size, epochs=epochs//2)
    acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 

    # Test the model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.concatenate((acc_hist, [accuracy[1]]), axis=0)
