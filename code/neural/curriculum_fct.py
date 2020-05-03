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
from custom_fct import *
from init_fct import *


def continous_curriculum(data, labels):
    """ Creates the curriculum by sorting the examples from easiest to 
    hardest ones. 
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: positive_distances -> np.array[float], list of distances for the positive
                examples.
            positive_sorted_indices -> np.array[int], list of sorted indices from the
                easiest to hardest positive example.
            negative_distances -> np.array[float], list of distances for the negative
                examples.
            negative_sorted_indices -> np.array[int], list of sorted indices from the
                easiest to hardest negative example.
    """
    
    positive_examples, negative_examples = find_examples(data, labels)  # Find positive and negative examples
    positive_average, negative_average = average_examples(data, labels) # Estimate the positive and negative average

    # Distance of positive examples to positive and negative examples
    positive_pdist = tf.squeeze(tf.norm(positive_examples-positive_average, axis=(1, 2)))
    positive_ndist = tf.squeeze(tf.norm(positive_examples-negative_average, axis=(1, 2)))

    # Distance of negative examples to negative and positive examples
    negative_ndist = tf.squeeze(tf.norm(negative_examples-negative_average, axis=(1, 2)))
    negative_pdist = tf.squeeze(tf.norm(negative_examples-positive_average, axis=(1, 2)))

    positive_distances = positive_ndist-positive_pdist
    negative_distances = negative_pdist-negative_ndist

    positive_sorted_indices = np.flip(np.argsort(positive_distances, kind='heapsort')) # Indices from easiest to hardest
    negative_sorted_indices = np.flip(np.argsort(negative_distances, kind='heapsort')) # Indices from easiest to hardest

    return tf.gather(positive_distances, positive_sorted_indices), positive_sorted_indices, tf.gather(negative_distances, negative_sorted_indices), negative_sorted_indices


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
            test_labels -> np.array[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, label.
            class_nb -> int, class selected for the one vs all.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracies -> np.array[float], accuracy at each epoch of training and
                for the test set.
    """

    model = student_model(train_data[0].shape) # Student model to train
    acc_hist = np.array([], dtype=np.float32)  # List for the accuracies
    max_class_nb = np.max(train_labels)        # Highest class number
    
    try:
        assert(isinstance(class_nb, int))
        assert(class_nb >= 0)
        assert(max_class_nb >= class_nb)

    except AssertionError:
        print("Error in function class_training: class_nb must be an integer smaller than the number of classes.")
        exit(1)

    # Get positive data
    positive_indices = np.where(train_labels == class_nb)[0] 
    positive_examples = tf.gather(train_data, positive_indices, axis=0) 
    positive_percent = np.linspace(0, len(positive_indices)-1, max_class_nb+2, endpoint=True, dtype=np.intc)

    for i in range(max_class_nb+1):
        positive_data = tf.gather(train_data, positive_indices[positive_percent[i]:positive_percent[i+1]-1]) 

        if i != class_nb:
            # Get data
            negative_data = tf.gather(train_data, np.where(train_labels == i)[0], axis=0)
            data  = tf.concat([positive_data, negative_data], axis=0)
            labels = tf.one_hot(np.concatenate((np.ones(positive_data.shape[0], dtype=np.intc), np.zeros(negative_data.shape[0], dtype=np.intc))), 2) 

            # Train model
            hist = model.fit(data, labels, batch_size=batch_size, epochs=epochs)
            acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0)

        else:

            # Train model with only positive data
            hist = model.fit(positive_data, tf.one_hot(np.ones(positive_data.shape[0], dtype=np.intc), 2), batch_size=batch_size, epochs=epochs)
            acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0)


    # Test the model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.append(acc_hist, [accuracy[1]], axis=0)
    


def continuous_training(train_data, train_labels, test_data, test_labels, rounds, batch_size=32, epochs=10):
    """ Trains the model by adding harder examples at each rounds.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
            test_data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
            rounds -> int, number of rounds of training.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracies -> np.array[float], accuracy at each epoch of training and
                for the test set.
    """

    model = student_model(train_data[0].shape) # Student model to train
    acc_hist = np.array([], dtype=np.float32)  # List for the accuracies

    positive_distances, positive_sorted_indices, negative_distances, negative_sorted_indices = continous_curriculum(train_data, train_labels)
    positive_examples, negative_examples = find_examples(train_data, train_labels)

    positive_percent = (len(positive_sorted_indices)-1)/rounds # Number of positive examples per round
    negative_percent = (len(negative_sorted_indices)-1)/rounds # Number of negative examples per round

    for i in range(1, rounds+1):
        # For each round get data
        positive_data = tf.gather(positive_examples, positive_sorted_indices[int((i-1)*positive_percent):int(i*positive_percent)])
        negative_data = tf.gather(negative_examples, negative_sorted_indices[int((i-1)*negative_percent):int(i*negative_percent)])
        data  = tf.concat([positive_data, negative_data], axis=0)
        labels = tf.one_hot(np.concatenate((np.ones(positive_data.shape[0], dtype=np.intc), np.zeros(negative_data.shape[0], dtype=np.intc))), 2)

        # Train model
        hist = model.fit(data, labels, batch_size=batch_size, epochs=epochs)
        acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0)

    # Test the model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.append(acc_hist, [accuracy[1]], axis=0)
