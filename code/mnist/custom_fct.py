#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from sklearn import svm


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


def find_examples(model_type, data, labels):
    """ Returns the positive and the negative examples given a data set
    and the associated labels.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], list of labels associated with the data.
    Output: positive_examples -> np.array[np.array[int]] or tf.tensor, list of
                the positive examples.
            negative_examples -> np.array[np.array[int]] or tf.tensor, list of
                the negative examples.
    """

    positive_indices, negative_indices = find_indices(model_type, labels)

    if model_type == 'cnn':
        # Extract positive and negative examples for the cnn model
        positive_examples = tf.gather(data, positive_indices, axis=0)
        negative_examples = tf.gather(data, negative_indices, axis=0)

    else:
        # Extract positive and negative examples for the svm model
        positive_examples = data[positive_indices]
        negative_examples = data[negative_indices]

    return positive_examples, negative_examples


def average_examples(model_type, data, labels):
    """ Calculates the average positive and negative examples given
    the train data and labels.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], list of labels associated with the data.
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


def sort_examples(model_type, data, labels):
    """ Extracts the positive and negative examples of the data using the labels and
    returns the sorted indices from closest to furthest to the positive and negative 
    average example respectively.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], list of labels associated with the data.
    Output: positive_sorted_indices -> np.array[int] or tf.tensor, sorted positive indices.
            negative_sorted_indices -> np.array[int] or tf.tensor, sorted negative indices.
    """

    positive_examples, negative_examples = find_examples(model_type, data, labels)  # Find positive and negative examples
    positive_average, negative_average = average_examples(model_type, data, labels) # Estimate the positive and negative average

    if model_type == 'cnn':
        # For neural netowrk
        positive_dist = tf.squeeze(tf.norm(positive_examples-positive_average, axis=(1, 2)))
        negative_dist = tf.squeeze(tf.norm(negative_examples-negative_average, axis=(1, 2)))

    else:
        # For svm
        positive_dist = np.linalg.norm(positive_examples-positive_average, axis=1)
        negative_dist = np.linalg.norm(negative_examples-negative_average, axis=1)

    return np.argsort(positive_dist, kind='heapsort'), np.argsort(negative_dist, kind='heapsort')
