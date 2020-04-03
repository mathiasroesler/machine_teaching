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


def create_teacher_set(model_type, train_data, train_labels, test_data, test_labels, lam_coef, set_limit, batch_size=32, epochs=10):
    """ Produces the optimal teaching set given the train_data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  model_type -> str, {'svm', 'cnn'} type of model used for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
            test_data -> np.array[np.array[int]] or tf.tensor, list of examples. 
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with the test data.
            lam_coef -> int, coefficiant for the exponential distribution
                for the thresholds.
            set_limit -> int, maximum number of examples to be put in the 
                teaching set.
    Output: teaching_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], list of labels associated with the
                teaching data.
            accuracy -> np.array[int], accuracy of the model at each iteration.
            teaching_set_len -> np.array[int], number of examples in teaching set at
                each iteration.
    """

    rng = default_rng() # Set seed 

    # Variables
    ite = 1
    nb_examples = train_data.shape[0]
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    missed_set_len = np.array([len(train_data)], dtype=np.intc) # List of number of misclassified examples at each iteration
    model = student_model(model_type) # Declare student model

    positive_average, negative_average = average_examples(model_type, train_data, train_labels)
    positive_indices, negative_indices = find_indices(model_type, train_labels)

    #positive_index, negative_index = rndm_init(positive_indices, negative_indices)
    positive_index, negative_index = min_avg_init(model_type, positive_indices, negative_indices, positive_average, negative_average, train_data)

    model, teaching_data, teaching_labels = teacher_initialization(model, model_type, train_data, train_labels, positive_index, negative_index, batch_size=batch_size, epochs=epochs)

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

        added_indices = select_rndm_examples(missed_indices, 200)
        #added_indices = select_examples(missed_indices, thresholds, weights)
        #added_indices = select_min_avg_dist(model_type, missed_indices, 200, train_data, train_labels, positive_average, negative_average)

        teaching_data, teaching_labels = update_teaching_set(model_type, teaching_data, teaching_labels, train_data, train_labels, added_indices)
        
        curr_accuracy = update_model(model, model_type, teaching_data, teaching_labels, test_data, test_labels, batch_size=batch_size, epochs=epochs)

        """
        if model_type == 'cnn':
            # Reset the weights for the cnn model
            model = student_model(model_type)
        """

        # Test model accuracy
        accuracy = np.concatenate((accuracy, [curr_accuracy]), axis=0)
        teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)

        # Remove train data and labels, weights and thresholds of examples in the teaching set
        train_data = np.delete(train_data, added_indices, axis=0)
        train_labels = np.delete(train_labels, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)
        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, accuracy, teaching_set_len, missed_set_len


def update_teaching_set(model_type, teaching_data, teaching_labels, train_data, train_labels, added_indices):
    """ Updates the teaching data and labels using the train data and labels and the indices of the examples
    to be added depending on the model type.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int] or tf.one_hot, list of labels associated with
                the teaching data.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int] or tf.one_hot, list of labels associated with
                the train data.
            added_indices -> np.array[int], list of indices of examples to be added to the
                teaching set.
    Output: teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int] or tf.one_hot, list of labels associated with
                the teaching data.
    """

    if model_type == 'cnn':
    # For the neural network
        teaching_data = tf.concat([teaching_data, tf.gather(train_data, added_indices)], axis=0)
        teaching_labels = tf.concat([teaching_labels, tf.gather(train_labels, added_indices)], axis=0)

    else:
    # For the svm
        teaching_data = np.concatenate((teaching_data, train_data[added_indices]), axis=0)
        teaching_labels = np.concatenate((teaching_labels, train_labels[added_indices]), axis=0)

    return teaching_data, teaching_labels


def update_model(model, model_type, teaching_data, teaching_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Updates the student model using the teaching data and labels and evaluates the new 
    performances with the test data and labels.
    Input:  model -> svm or cnn model, student model.
            model_type -> str, {'svm', 'cnn'} model used for the student.
            teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], list of labels associated with
                the teaching data.
            test_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with
                the test data.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracy -> float, score obtained with the updated model on the test data.
    """

    if model_type == 'cnn':
    # Update cnn model
        model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 

        # Test the updated model
        accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

        return accuracy[1]

    else:
    # Update svm model
        model.fit(teaching_data, teaching_labels.ravel())

        # Test the updated model
        accuracy = model.score(test_data, test_labels)

        return accuracy


def average_examples(model_type, train_data, train_labels):
    """ Calculates the average positive and negative examples given
    the train data and labels.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
    Output: positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
    """

    if model_type == 'cnn':
    # For cnn student model
        positive_examples = tf.gather(train_data, np.nonzero(train_labels[:, 0] == 0)[0], axis=0)
        negative_examples = tf.gather(train_data, np.nonzero(train_labels[:, 0] == 1)[0], axis=0)
        return tf.constant(np.mean(positive_examples, axis=0)), tf.constant(np.mean(negative_examples, axis=0))

    else:
    # For svm student model
        positive_examples = train_data[np.nonzero(train_labels == 1)[0]]
        negative_examples = train_data[np.nonzero(train_labels == 0)[0]]
        return np.mean(positive_examples, axis=0), np.mean(negative_examples, axis=0)
