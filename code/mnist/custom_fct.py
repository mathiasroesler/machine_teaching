#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
from init_fct import *
from sklearn import svm


def train_student_model(model_type, train_data, train_labels, test_data, test_labels, max_iter=5000, batch_size=32, epochs=10):
    """ Trains a student model fitted with the train set and
    tested with the test set. Returns None if an error occured.
    Input:  model_type -> str, {'svm', 'cnn'} model type for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the
                train examples.
            test_data -> np.array[np.array[int]] or tf.tensor, list of examples
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with the 
                test examples.
            max_iter -> int, maximum iterations for the model fitting.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: test_score -> float, score obtained with the test set. 
    """

    model = student_model(model_type, max_iter=max_iter) # Declare student model
    
    if model == None:
        print("Error in function train_svm_model: the model was not created.")
        return None

    print("\nSet length", len(train_data))

    if model_type == 'cnn':
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
        test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)
        print("\nTest score", test_score[1])
        return test_score[1]

    else: 
        model.fit(train_data, train_labels) # Train model with data
        test_score = model.score(test_data, test_labels)    # Test score for fully trained model
        print("\nTest score", test_score)
        return test_score


def find_indices(model_type, labels):
    """ Returns the indices for the positive and negative examples given
    the labels. The positive labels must be 1 and the negative ones 0.
    Input:  model_type -> str, {svm, cnn} student model type.
            labels -> np.array[int], labels for a given data set.
    Output: positive_indices -> np.array[int], list of positive labels indices.
            negative_indices -> np.array[int], list of negative labels indices.
    """
    
    if model_type == 'cnn':
    # The data is one_hot
        positive_indices = np.nonzero(labels[:, 0] == 0)[0]
        negative_indices = np.nonzero(labels[:, 0] == 1)[0]

    else:
        positive_indices = np.nonzero(labels == 1)[0]
        negative_indices = np.nonzero(labels == 0)[0]

    return positive_indices, negative_indices


def average_examples(model_type, data, labels):
    """ Calculates the average positive and negative examples given
    the train data and labels.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            abels -> np.array[int], list of labels associated with the data.
    Output: positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
    """

    if model_type == 'cnn':
    # For cnn student model
        positive_examples = tf.gather(data, np.nonzero(labels[:, 0] == 0)[0], axis=0)
        negative_examples = tf.gather(data, np.nonzero(labels[:, 0] == 1)[0], axis=0)
        return tf.constant(np.mean(positive_examples, axis=0)), tf.constant(np.mean(negative_examples, axis=0))

    else:
    # For svm student model
        positive_examples = data[np.nonzero(labels == 1)[0]]
        negative_examples = data[np.nonzero(labels == 0)[0]]
        return np.mean(positive_examples, axis=0), np.mean(negative_examples, axis=0)
