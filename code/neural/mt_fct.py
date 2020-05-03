#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the teacher algorithm.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from numpy.random import default_rng 
from selection_fct import *
from init_fct import *
from custom_fct import *


def create_teacher_set(train_data, train_labels, lam_coef, set_limit, batch_size=32, epochs=10):
    """ Produces the optimal teaching set given the train_data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
            lam_coef -> int, coefficiant for the exponential distribution
                for the thresholds.
            set_limit -> int, maximum number of examples to be put in the 
                teaching set.
    Output: teaching_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], list of labels associated with the
                teaching data.
            teaching_set_len -> np.array[int], number of examples in teaching set at
                each iteration.
    """

    rng = default_rng() # Set seed 

    # Variables
    ite = 1
    old_acc = 0
    eps = 10e-3
    nb_examples = train_data.shape[0]
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    missed_set_len = np.array([len(train_data)], dtype=np.intc) # List of number of misclassified examples at each iteration
    model = student_model(train_data[0].shape) # Declare student model

    positive_index, negative_index = rndm_init(train_labels) # Find index of intial examples

    model, teaching_data, teaching_labels = teacher_initialization(model, train_data, train_labels, positive_index, negative_index, batch_size=batch_size, epochs=epochs)

    # Remove inital examples from the data and labels
    # train_data = np.delete(train_data, [positive_index, negative_index], axis=0)
    # train_labels = np.delete(train_labels, [positive_index, negative_index], axis=0)

    while len(teaching_data) != nb_examples and len(teaching_data) < set_limit:
        # Exit if all of the examples are in the teaching set or enough examples in the teaching set
        missed_indices = np.array([], dtype=np.intc) # List of the indices of the missclassified examples
        added_indices = np.array([], dtype=np.intc)  # List of the indices of added examples to the teaching set
        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.unique(np.nonzero(np.round(model.predict(train_data)) != train_labels)[0])
        missed_set_len = np.concatenate((missed_set_len, [len(missed_indices)]), axis=0)
    
        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        #added_indices = select_rndm_examples(missed_indices, 200)
        added_indices = select_examples(missed_indices, thresholds, weights)
        #added_indices = select_min_avg_dist(missed_indices, 200, train_data, train_labels, positive_average, negative_average)
        #added_indices = select_curriculum_examples(200, train_data, train_labels, ite-1)

        # Update the teacher set and add length to list
        teaching_data, teaching_labels = update_teaching_set(teaching_data, teaching_labels, train_data, train_labels, added_indices)
        teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)
    
        # Update model
        hist = model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 
        new_acc = np.sum(hist.history.get('accuracy'))/epochs

        if (abs(new_acc-old_acc) < eps):
            # Avoid overlearning
            break

        old_acc = new_acc

        # Reset the weights for the cnn model
        #model = student_model(train_data[0].shape)

        """
        # Remove train data and labels, weights and thresholds of examples in the teaching set
        train_data = np.delete(train_data, added_indices, axis=0)
        train_labels = np.delete(train_labels, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)
        """
        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, teaching_set_len, missed_set_len


def train_student_model(train_data, train_labels, test_data, test_labels, batch_size=32, epochs=10):
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

    model = student_model(train_data[0].shape) # Declare student model
    
    print("\nSet length", len(train_data))

    hist = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.array(np.append(hist.history.get('accuracy'), [test_score[1]]), dtype=np.float32)


def update_teaching_set(teaching_data, teaching_labels, data, labels, added_indices):
    """ Updates the teaching data and labels using the train data and labels and the indices of the examples
    to be added depending on the model type.
    Input:  teaching_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            teaching_labels -> tf.tensor[int], list of labels associated
                with the teaching data.
                First dimension, number of examples.
                Second dimension, one hot label.
            data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
            added_indices -> np.array[int], list of indices of examples to be added to the
                teaching set.
    Output: teaching_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            teaching_labels -> tf.tensor[int], list of labels associated
                with the teaching data.
                First dimension, number of examples.
                Second dimension, one hot label.
    """

    teaching_data = tf.concat([teaching_data, tf.gather(data, added_indices)], axis=0)
    teaching_labels = tf.concat([teaching_labels, tf.gather(labels, added_indices)], axis=0)

    return teaching_data, teaching_labels

