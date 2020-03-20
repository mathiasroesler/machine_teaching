#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the teacher algorithm.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
from numpy.random import default_rng 
from sklearn import svm
from scipy.spatial import distance

def create_teacher_set(train_set, test_set, lam_coef, set_limit, max_iter=100):
    """ Produces the optimal teaching set given the train_set and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  train_set -> np.array[np.array[int]], list of features with
                the last element being the label.
                First dimension number of examples.
                Second dimension features.
            test_set -> np.array[np.array[int]], list of features with
                the last element being the label. 
                First dimension number of examples.
                Second dimension features.
            lam_coef -> int, coefficiant for the exponential distribution
                for the thresholds.
            set_limit -> int, maximum number of examples to be put in the 
                teaching set.
            max_iter -> int, max number of iterations.
    Output: teaching_set -> np.array[np.array[int]], each row is the features 
                for an example with the last element being the label.
                First dimension number of examples.
                Second dimension features.
            accuracy -> np.array[int], accuracy of the model at each iteration.
            example_nb -> np.array[int], number of examples in teaching set at
                each iteration.
    """

    rng = default_rng() # Set seed 

    # Variables
    ite = 0
    nb_examples = train_set.shape[0]
    weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    example_nb = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    model = svm.LinearSVC() # SVM model

    dist_matrix, positive_examples, negative_examples = euclidean_dist(train_set)

    indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    teaching_set = np.vstack([positive_examples[indices[0]], negative_examples[indices[1]]])
    
    model.fit(teaching_set[:, :-1], teaching_set[:, -1])

    accuracy = np.append(accuracy, [model.score(test_set[:, :-1], test_set[:, -1])], axis=0)
    example_nb = np.append(example_nb, [len(teaching_set)], axis=0)

    """
    for i in range(1, len(train_set)):
        # Seek two examples of different classes
        if train_set[i][-1] != train_set[i-1][-1]:
            # If the two examples belong to different classes
            model.fit(train_set[i-1:i+1, :-1], train_set[i-1:i+1, -1]) # Intialize learner

            # Add examples to the teaching set
            teaching_set = np.vstack((train_set[i-1], train_set[i]))
            
            # Test the accuracy of the model
            accuracy = np.append(accuracy, [model.score(test_set[:, :-1], test_set[:, -1])], axis=0)
            example_nb = np.append(example_nb, [len(teaching_set)], axis=0)
             
            # Remove examples from train_set, weights and threshold
            train_set = np.delete(train_set, [i-1, i], axis=0)
            weights = np.delete(weights, [i-1, i], axis=0)
            thresholds = np.delete(thresholds, [i-1, i], axis=0)
            
            break
    """
    percent = np.append(np.linspace(0.99, 0.5, num=50), np.linspace(1.5, 1, num=50)) 

    #while len(teaching_set) != nb_examples and ite < max_iter and len(teaching_set) < set_limit:
    for i in range(1, len(percent)-1):
        # Exit if all of the examples are in the teaching set
        """
        missed_indices = np.array([], dtype=np.intc) # List of the indices of the missclassified examples
        added_indices = np.array([], dtype=np.intc)  # List of the indices of added examples to the teaching set
        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.where(model.predict(train_set[:, :-1]) != train_set[:, -1])[0]

        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        while np.sum(weights[missed_indices]) < 1:
            weights[missed_indices] = 2*weights[missed_indices] # Double weights of missed examples
            added_indices = np.where(weights[missed_indices] >= thresholds[missed_indices])[0] # Add indices of weights above threshold

        if max_iter == 100:
            max_cor_indices = correlation(teaching_set, train_set[added_indices])
            teaching_set = np.vstack((teaching_set, train_set[added_indices[max_cor_indices]])) # Add examples to the teaching set

        else:
            teaching_set = np.vstack((teaching_set, train_set[added_indices]))
        """
        if percent[i] < 1:
            indices = np.where((dist_matrix <= percent[i-1]*np.max(dist_matrix)) * (dist_matrix >= percent[i]*np.max(dist_matrix)))
            if len(indices[0]):
                rand_index = rng.integers(0, len(indices[0]))
                examples = np.array([positive_examples[indices[0][rand_index]], negative_examples[indices[1][rand_index]]])
                teaching_set = np.vstack((teaching_set, examples))

        elif percent[i] > 1:
            indices = np.where((dist_matrix <= percent[i]*np.min(dist_matrix)) * (dist_matrix >= percent[i+1]*np.min(dist_matrix)))
            if len(indices[0]):
                rand_index = rng.integers(0, len(indices[0]))
                examples = np.array([positive_examples[indices[0][rand_index]], negative_examples[indices[1][rand_index]]])
                teaching_set = np.vstack((teaching_set, examples))

        # Fit the model with the new teaching set
        model.fit(teaching_set[:, :-1], teaching_set[:, -1])
        
        # Test model accuracy
        accuracy = np.append(accuracy, [model.score(test_set[:, :-1], test_set[:, -1])], axis=0)
        example_nb = np.append(example_nb, [len(teaching_set)], axis=0)
        
        """
        # Remove train_set, weights and thresholds of examples in the teaching set
        train_set = np.delete(train_set, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)
        """
        ite += 1

    print("\nIteration number:", ite)

    return teaching_set, accuracy, example_nb


def correlation(teaching_set, missed_set):
    """ Calculates the correlation between the examples in the teaching set
    and the examples that were missed.
    Input:  teaching_set -> np.array[np.array[int]], each row is the features
                for an example with the last element being the label.
                First dimension number of examples.
                Second dimension features.
            missed_set -> np.array[np.array[int]], each row is the features
                for a missclassified example, the last element is the label.
                First dimension number of examples.
                Second dimension features.
    Output: example_indices -> np.array[int], list of indices of the examples
                that are the most similar to the ones in the teaching set. There
                is one for every example in the teaching set.
    """
    
    cor_matrix = np.zeros(shape=(teaching_set.shape[0], missed_set.shape[0]), dtype=np.float32)
    
    for i in range(teaching_set.shape[0]):
        for j in range(missed_set.shape[0]):
            cor_matrix[i][j] = np.correlate(teaching_set[i], missed_set[j])

    return np.argmin(cor_matrix, axis=1)


def euclidean_dist(train_set):
    """ Calculates the euclidean distance between all of the positive
    examples and the negative ones.
    Input:  train_set -> np.array[np.array[int]], list of features with
                the last element being the label.
                First dimension number of examples.
                Second dimension features.
    Output: dist_matrix -> np.array[np.array[int]], matrix of distances.
    """

    positive_examples = train_set[np.where(train_set[:, -1] != -1)]
    negative_examples = train_set[np.where(train_set[:, -1] == -1)]

    dist_matrix = distance.cdist(positive_examples[:, :-1], negative_examples[:, :-1], metric='canberra')
    
    return dist_matrix, positive_examples, negative_examples
