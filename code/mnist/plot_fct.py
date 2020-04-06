#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Plot functions for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from custom_fct import *


def plot_data(full_train_score, accuracy, example_nb, missed_len):
    """ Plots the data.
    Input:  full_train_score -> float, accuracy for the fully trained model.
            accuracy -> np.array[int], accuracy at each iteration for
                the optimally trained model.
            example_nb -> np.array[int], number of examples at each 
                iteration for the optimally trained model.
            missed_len -> np.array[int], number of misclassified examples at
                each iteration.
    Output:
    """

    # Plot accuracy as function of number of examples
    plt.subplot(1, 2, 1)
    plt.plot(example_nb, accuracy,'bo-', label="Optimally trained model") 
    plt.plot(example_nb, [full_train_score for i in range(len(example_nb))], 'ro-', label="Fully trained model") 

    plt.xlabel("Teaching set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    
    # Plot number of missclassified examples as a function of iteration
    plt.subplot(1, 2, 2)
    plt.plot(range(len(missed_len)), missed_len, 'ko-', label="Misclassified examples")

    plt.xlabel("Iteration number")
    plt.ylabel("Number of examples")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()
    

def plot_avg_dist(model_type, data, labels, positive_average, negative_average):
    """ Plots the distance of the examples in the teacher set to the average example.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], labels associated with the data.
            positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
    Output: 
    """

    if model_type == 'cnn':
    # For cnn student model
        positive_indices, negative_indices = find_indices(model_type, labels)

        # Extract positive and negative examples from data
        positive_examples = tf.gather(data, positive_indices, axis=0)
        negative_examples = tf.gather(data, negative_indices, axis=0)

        positive_dist = tf.norm(positive_examples-negative_average, axis=(1, 2))
        negative_dist = tf.norm(negative_examples-positive_average, axis=(1, 2))

        plt.plot(positive_dist, labels[positive_indices][1], 'r^', label="Positive examples")
        plt.plot(negative_dist, labels[negative_indices][1], 'bo', label="Negative examples")

    else:
    # For svm student model
        positive_indices, negative_indices = find_indices(model_type, labels)

        # Extract positive and negative examples from data
        positive_examples = data[positive_indices]
        negative_examples = data[negative_indices]
    
        positive_dist = np.linalg.norm(positive_examples-negative_average, axis=1)
        negative_dist = np.linalg.norm(negative_examples-positive_average, axis=1)

        plt.plot(positive_dist, labels[positive_indices], 'r^', label="Positive examples")
        plt.plot(negative_dist, labels[negative_indices], 'bo', label="Negative examples")


    plt.xlabel("Distance")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()
    
