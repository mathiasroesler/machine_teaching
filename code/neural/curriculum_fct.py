#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function for curriculum learning. 
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from data_fct import prep_data
from custom_fct import *
from init_fct import *


def two_step_curriculum(data, labels):
    """ Creates a curriculum dividing the data into easy and
    hard examples taking into account the classes.
    Input:  data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, label.
    Output: easy_indices -> np.array[int], list indices associated
                with the easy examples of the data.
            hard_indices -> np.array[int], list of hard indices associatied
                with the hard examples of the data.
    """

    max_class_nb = np.max(labels) # Highest class number
    easy_indices = np.array([], dtype=np.intc)
    hard_indices = np.array([], dtype=np.intc)
    
    classes = np.random.choice(range(max_class_nb+1), max_class_nb+1, replace=False)
    averages = average_examples(data, labels)  # List of average example for each class
    examples = find_examples(data, labels)     # List of examples for each class
    
    for i in classes:
        dist = tf.norm(averages[i]-examples[i], axis=(1, 2)) # Estimate distance to average
        dist = dist/np.max(dist)    # Normalize distances

        easy_indices = np.concatenate([easy_indices, np.where(dist <= np.mean(dist))[0]], axis=0)
        hard_indices = np.concatenate([hard_indices, np.where(dist > np.mean(dist))[0]], axis=0)
        
    return easy_indices, hard_indices


def class_training(train_data, train_labels, test_data, test_labels, class_nb, batch_size=32, epochs=10):
    """ Trains the model by showing classes one by one.
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

    model = student_model(train_data[0].shape) # Student model to train
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
