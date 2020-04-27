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


def create_curriculum(model_type, data, labels):
    """ Creates the curriculum by sorting the examples from easiest to 
    hardest ones. 
    Input:  model_type -> str, {'svm', 'cnn'} model type for the learner.
            data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], list of labels associated with the data.
    Output: positive_distances -> np.array[float], list of distances for the positive
            examples.
            positive_sorted_indices -> np.array[int], list of sorted indices from the
            easiest to hardest positive example.
            negative_distances -> np.array[float], list of distances for the negative
            examples.
            negative_sorted_indices -> np.array[int], list of sorted indices from the
            easiest to hardest negative example.
    """
    
    positive_examples, negative_examples = find_examples(model_type, data, labels)  # Find positive and negative examples
    positive_average, negative_average = average_examples(model_type, data, labels) # Estimate the positive and negative average

    if model_type == 'cnn':
        # For neural network
        positive_pdist = tf.squeeze(tf.norm(positive_examples-positive_average, axis=(1, 2)))
        positive_ndist = tf.squeeze(tf.norm(positive_examples-negative_average, axis=(1, 2)))

        negative_ndist = tf.squeeze(tf.norm(negative_examples-negative_average, axis=(1, 2)))
        negative_pdist = tf.squeeze(tf.norm(negative_examples-positive_average, axis=(1, 2)))

    else:
        # For svm
        positive_pdist = np.linalg.norm(positive_examples-positive_average, axis=1)
        positive_ndist = np.linalg.norm(positive_examples-negative_average, axis=1)

        negative_ndist = np.linalg.norm(negative_examples-negative_average, axis=1)
        negative_pdist = np.linalg.norm(negative_examples-positive_average, axis=1)

    positive_distances = positive_ndist-positive_pdist
    negative_distances = negative_pdist-negative_ndist

    positive_sorted_indices = np.flip(np.argsort(positive_distances, kind='heapsort')) # Indices from easiest to hardest
    negative_sorted_indices = np.flip(np.argsort(negative_distances, kind='heapsort')) # Indices from easiest to hardest

    if model_type == 'cnn':
        return tf.gather(positive_distances, positive_sorted_indices), positive_sorted_indices, tf.gather(negative_distances, negative_sorted_indices), negative_sorted_indices

    else:
        return positive_distances[positive_sorted_indices], positive_sorted_indices, negative_distances[negative_sorted_indices], negative_sorted_indices


def continuous_training(model_type, train_data, train_labels, test_data, test_labels, iteration, batch_size=32, epochs=10):
    """ Trains the model by adding harder examples at each iteration.
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
            iteration -> int, number of iterations of training.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracies -> np.array[float], accuracy at each iteration. 
    """

    model = student_model(model_type)
    accuracies = np.zeros(iteration+1, dtype=np.float32)
    acc_hist = np.array([], dtype=np.float32)

    positive_distances, positive_sorted_indices, negative_distances, negative_sorted_indices = create_curriculum(model_type, train_data, train_labels)
    positive_examples, negative_examples = find_examples(model_type, train_data, train_labels)

    positive_percent = (len(positive_sorted_indices)-1)/iteration
    negative_percent = (len(negative_sorted_indices)-1)/iteration

    for i in range(1, iteration+1):
        if model_type == 'cnn':
            # For neural network
            positive_data = tf.gather(positive_examples, positive_sorted_indices[int((i-1)*positive_percent):int(i*positive_percent)])
            negative_data = tf.gather(negative_examples, negative_sorted_indices[int((i-1)*negative_percent):int(i*negative_percent)])
            data  = tf.concat([positive_data, negative_data], axis=0)
            labels = tf.one_hot(np.concatenate((np.ones(positive_data.shape[0], dtype=np.intc), np.zeros(negative_data.shape[0], dtype=np.intc))), 2)

            hist = model.fit(data, labels, batch_size=batch_size, epochs=epochs)
            acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0)
            accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
            accuracies[i] = accuracy[1] 

        else:
            # For svm
            positive_data = positive_examples[positive_sorted_indices[int((i-1)*positive_percent):int(i*positive_percent)]]
            negative_data = negative_examples[negative_sorted_indices[int((i-1)*negative_percent):int(i*negative_percent)]]
            data = np.concatenate((positive_data, negative_data), axis=0) 
            labels = np.concatenate((np.ones(positive_data.shape[0], dtype=np.intc), np.zeros(negative_data.shape[0], dtype=np.intc)))

            model.fit(data, labels.ravel())
            accuracies[i] = model.score(test_data, test_labels)
    
    return [acc_hist, accuracies]


def discret_training(model_type, train_data, train_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Trains the model in two or three steps by using all of the easy
    examples in the first batch and then harder examples in the other batches.
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
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: 
    """

    model = student_model(model_type)
    
    positive_distances, positive_sorted_indices, negative_distances, negative_sorted_indices = create_curriculum(model_type, train_data, train_labels)
    positive_examples, negative_examples = find_examples(model_type, train_data, train_labels)

    positive_distances = np.asarray(positive_distances)
    negative_distances = np.asarray(negative_distances)

    if model_type == 'cnn':
        # For neural network
        # Extract easy examples and create training set
        positive_easy = tf.gather(positive_examples, positive_sorted_indices[:500])
        negative_easy = tf.gather(negative_examples, negative_sorted_indices[:3000])
        easy_examples = tf.concat([positive_easy, negative_easy], axis=0)
        easy_labels = tf.one_hot(np.concatenate((np.ones(positive_easy.shape[0], dtype=np.intc), np.zeros(negative_easy.shape[0], dtype=np.intc))), 2)

        model.fit(easy_examples, easy_labels, batch_size=batch_size, epochs=epochs) 

        # Test the model
        accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

        print("Easy train", accuracy[1])

        # Extract hard examples and create training set
        positive_hard = tf.gather(positive_examples, positive_sorted_indices[500:])
        negative_hard = tf.gather(negative_examples, negative_sorted_indices[3000:])
        hard_examples = tf.concat([positive_hard, negative_hard], axis=0)
        hard_labels = tf.one_hot(np.concatenate((np.ones(positive_hard.shape[0], dtype=np.intc), np.zeros(negative_hard.shape[0], dtype=np.intc))), 2)

        model.fit(hard_examples, hard_labels, batch_size=batch_size, epochs=epochs)

        # Test the model
        accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

        print("Hard train", accuracy[1])

    else:
        # For svm
        # Extract easy examples and create training set
        positive_easy = positive_examples[positive_sorted_indices[:500]]
        negative_easy = negative_examples[negative_sorted_indices[:3000]]
        easy_examples = np.concatenate((positive_easy, negative_easy), axis=0) 
        easy_labels = np.concatenate((np.ones(positive_easy.shape[0], dtype=np.intc), np.zeros(negative_easy.shape[0], dtype=np.intc)))

        model.fit(easy_examples, easy_labels.ravel())
        print("Easy train", model.score(test_data, test_labels))

        # Extract hard examples and create training set
        positive_hard = positive_examples[positive_sorted_indices[500:]]
        negative_hard = negative_examples[negative_sorted_indices[3000:]]
        hard_examples = np.concatenate((positive_hard, negative_hard), axis=0)
        hard_labels = np.concatenate((np.ones(positive_hard.shape[0], dtype=np.intc), np.zeros(negative_hard.shape[0], dtype=np.intc)))

        model.fit(hard_examples, hard_labels.ravel())

        # Test the model
        print("Hard train", model.score(test_data, test_labels))

