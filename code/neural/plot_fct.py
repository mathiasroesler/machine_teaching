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
from matplotlib import colors
from custom_fct import *
from scipy.spatial import distance


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

    # Show plots
    plt.tight_layout()
    plt.show()
    

def plot_avg_dist(data, labels, positive_average, negative_average):
    """ Plots the distance of the examples in the teacher set to the average example.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
                First dimension number of examples.
                Second dimension features.
            positive_average -> tf.tensor[float32], average positive example.
            negative_average -> tf.tensor[float32], average negative example.
    Output: 
    """

    positive_indices, negative_indices = find_indices(labels)
    positive_examples, negative_examples = find_examples(data, labels)

    positive_dist = tf.norm(positive_examples-negative_average, axis=(1, 2))
    negative_dist = tf.norm(negative_examples-positive_average, axis=(1, 2))

    # Normalize data
    positive_dist = positive_dist/np.max(positive_dist)
    negative_dist = negative_dist/np.max(negative_dist)

    # Plot inital example
    plt.plot(positive_dist[0], labels[positive_indices[0]][1], 'gD', markersize=10, label='Initial example')
    plt.plot(negative_dist[0], labels[negative_indices[0]][1], 'gD', markersize=10)

    plt.plot(positive_dist[1:], tf.gather(labels, positive_indices)[1:, 1], 'r^', label="Positive examples")
    plt.plot(negative_dist[1:], tf.gather(labels, negative_indices)[1:, 1], 'bo', label="Negative examples")

    plt.xlabel("Distance")
    plt.ylabel("Label")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()
    

def plot_example_dist(data, labels):
    """ Plots the Euclidean distance between the different examples. The positive
    and negative examples are treated seperatly.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: 
    """

    positive_examples, negative_examples = find_examples(data, labels)

    positive_len = positive_examples.shape[0]
    negative_len = negative_examples.shape[0]

    positive_dist = np.zeros((positive_len, positive_len), dtype=np.float32)
    negative_dist = np.zeros((negative_len, negative_len), dtype=np.float32)
    data_dist = np.zeros((positive_len, negative_len), dtype=np.float32)

    for i in range(positive_len):
        positive_dist[:, i] = np.squeeze(tf.norm(positive_examples-positive_examples[i], axis=(1, 2)))

    for i in range(negative_len):
        negative_dist[:, i] = np.squeeze(tf.norm(negative_examples-negative_examples[i], axis=(1, 2)))
        data_dist[:, i] = np.squeeze(tf.norm(positive_examples-negative_examples[i], axis=(1, 2)))

    # Normalize data
    positive_dist = positive_dist/np.max(positive_dist)
    negative_dist = negative_dist/np.max(negative_dist)
    data_dist = data_dist/np.max(data_dist)

    # Plot the positive data
    plt.subplot(1, 3, 1)
    #plt.figure()
    plt.imshow(positive_dist, cmap='plasma') # Plot data
    plt.title('Distance between positive examples')
    plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()

    # Plot the negative data
    plt.subplot(1, 3, 2)
    #plt.figure()
    plt.imshow(negative_dist, cmap='plasma') # Plot data
    plt.title('Distance between negative examples')
    plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()

    # Plot the data
    plt.subplot(1, 3, 3)
    #plt.figure()
    plt.imshow(data_dist, cmap='plasma') # Plot data
    plt.xlabel('Negative examples')
    plt.ylabel('Positive examples')
    plt.title('Distance between positive and negative examples')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.show()


def plot_comp(acc_list, plot_types, plot_labels):
    """ Plot the comparaison between curriculum learning and machine
    teaching accuracies.
    Input:  acc_list -> list[np.array[float32]], list of accuracies for
                each strategy. The last element of each accuracy list must
                be the test accuracy.
            plot_types -> list[str], list of plot color and type for each
                strategy.
            plot_labels -> list[str], list of labels associated with each
                strategy.
    Output:
    """

    try:
        assert(len(acc_list) == len(plot_types) == len(plot_labels))

    except:
        print("Error in function plot_comp: the inputs must all have the same dimension.")
        exit(1)

    strategy_nb = len(acc_list)

    for i in range(strategy_nb):
        plt.plot(range(len(acc_list[i])-1), acc_list[i][:-1], plot_types[i], label=plot_labels[i])

    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    
    plt.show()
