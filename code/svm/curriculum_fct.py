#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function for curriculum learning. 
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
from custom_fct import *
from init_fct import *


def create_curriculum(data, labels):
    """ Creates the curriculum by sorting the examples from easiest to 
    hardest ones. 
    Input:  data -> np.array[np.array[int]], list of examples.
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
    
    positive_examples, negative_examples = find_examples(data, labels)  # Find positive and negative examples
    positive_average, negative_average = average_examples(data, labels) # Estimate the positive and negative average

    positive_pdist = np.linalg.norm(positive_examples-positive_average, axis=1)
    positive_ndist = np.linalg.norm(positive_examples-negative_average, axis=1)

    negative_ndist = np.linalg.norm(negative_examples-negative_average, axis=1)
    negative_pdist = np.linalg.norm(negative_examples-positive_average, axis=1)

    positive_distances = positive_ndist-positive_pdist
    negative_distances = negative_pdist-negative_ndist

    positive_sorted_indices = np.flip(np.argsort(positive_distances, kind='heapsort')) # Indices from easiest to hardest
    negative_sorted_indices = np.flip(np.argsort(negative_distances, kind='heapsort')) # Indices from easiest to hardest

    return positive_distances[positive_sorted_indices], positive_sorted_indices, negative_distances[negative_sorted_indices], negative_sorted_indices


def continuous_training(train_data, train_labels, test_data, test_labels, iteration):
    """ Trains the model by adding harder examples at each iteration.
    Input:  train_data -> np.array[np.array[int]], list of examples
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the
                train examples.
            test_data -> np.array[np.array[int]], list of examples
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with the 
                test examples.
            iteration -> int, number of iterations of training.
    Output: accuracies -> np.array[float], accuracy at each iteration. 
    """

    model = student_model()
    accuracies = np.zeros(iteration+1, dtype=np.float32)

    positive_distances, positive_sorted_indices, negative_distances, negative_sorted_indices = create_curriculum(train_data, train_labels)
    positive_examples, negative_examples = find_examples(train_data, train_labels)

    positive_percent = (len(positive_sorted_indices)-1)/iteration
    negative_percent = (len(negative_sorted_indices)-1)/iteration

    for i in range(1, iteration+1):
        positive_data = positive_examples[positive_sorted_indices[int((i-1)*positive_percent):int(i*positive_percent)]]
        negative_data = negative_examples[negative_sorted_indices[int((i-1)*negative_percent):int(i*negative_percent)]]
        data = np.concatenate((positive_data, negative_data), axis=0) 
        labels = np.concatenate((np.ones(positive_data.shape[0], dtype=np.intc), np.zeros(negative_data.shape[0], dtype=np.intc)))

        model.fit(data, labels.ravel())
        accuracies[i] = model.score(test_data, test_labels)
    
    return accuracies

