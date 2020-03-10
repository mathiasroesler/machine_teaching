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

def teacher_set(data, lam_coef, ite_max_nb):
    """ Produces the optimal teaching set given the data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  data -> np.array[list[int]], list of features with
                the last element being the label greater than 0.
            lam_coef -> int, coefficiant for the exponential distribution
                for the threshold.
            ite_max_nb -> int, maximal number of iterations.
    Output: teaching_set -> np.array[list[int]], each row is the features 
                for an example with the last element being the label.
    """

    # Check consitency
    if not isinstance(ite_max_nb, int):
        print("Error in function teacher_set: the number of iteration is not an integer")
        return None

    if ite_max_nb <= 0:
        print("Error in function teacher_set: the number of iteration must be a positive integer greater than 0")
        return None
    
    rng = default_rng(0) # Set seed

    # Variables
    ite = 0 # Iteration counter
    nb_examples = data.shape[0]
    weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example
    threshold = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    model = svm.LinearSVC() # SVM model

    for i in range(1, len(data)):
        # Seek two examples of different classes
        if data[i][-1] != data[i-1][-1]:
            # If the two examples belong to different classes
            model.fit(data[i-1:i+1, :-1], data[i-1:i+1, -1]) # Intialize learner

            # Add examples to the teaching set
            teaching_set = np.vstack((data[i-1], data[i]))
            break

    while len(teaching_set) != nb_examples or ite > ite_max_nb:
        # Exit if all of the examples are in the teaching set
        missed_indices = list() # List of the indices of the missclassified examples

        for i in range(len(data)):
            example = data[i]
            if model.predict(example[:-1].reshape(1, -1)) != example[-1]:
                missed_indices.append(i)
                
        if not missed_indices:
            # All examples are placed correctly
            break

        while weight_sum(weights, missed_indices) < 1:
            for index in missed_indices:
                # Double the value of each weight
                weights[index] = 2*weights[index] 

                if weights[index] > threshold[index]:
                    # If the weight exceeds the threshold add example to teaching set
                    teaching_set = np.vstack((teaching_set, data[index]))

        model.fit(teaching_set[:, :-1], teaching_set[:, -1])

    if ite > ite_max_nb:
        print("Unable to converge to a solution")
        return -1

    return teaching_set


def weight_sum(weights, indices):
    """ Calculates the sum of the missclassified weights.
    Input:  weights -> list[float]
            indices -> list[int]
    Output: total -> float
    """

    total = 0.

    for index in indices:
        total += weights[index]

    return total
