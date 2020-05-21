#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for machine teaching.
Date: 21/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf


class SPLLoss:
    """ Custom loss for self-paced learning. """

    def __init__(self, threshold=np.finfo(np.float32).max, growth_factor=1.0, warm_up_ite=1):
        """ Initialize the class with a threshold and a
        growth factor.
        Input:  threshold -> float, value below which SPL
                    examples will be used for backpropagation.
                growth_factor -> float, multiplicative value to 
                    increase the threshold.
        Output:
        """

        try:
            assert(np.issubdtype(type(threshold), np.floating) or np.issubdtype(type(threshold), np.integer))
            assert(np.issubdtype(type(growth_factor), np.floating) or np.issubdtype(type(growth_factor), np.integer))

        except AssertionError:
            print("Error in SPLLoss init function: the type of the inputs are incorrect")
            exit(1)

        self.threshold = threshold
        self.growth_factor = growth_factor
        self.warm_up_ite = warm_up_ite 
        self.loss_ite = -1 # Counts the number of times the loss function has been called


    def loss(self, found_labels, true_labels):
        """ Calculates the SPL loss given the found and true labels.
        Input:  found_labels -> tf.tensor[tf.double], labels estimated
                    by the model.
                true_labels -> tf.tensor[tf.double], true labels for the
                    examples.
        Output: 
        """

        try:
            assert(tf.is_tensor(found_labels))
            assert(tf.is_tensor(true_labels))

        except AssertionError:
            # If not tensors
            found_labels = tf.convert_to_tensor(found_labels, dtype=tf.float32)
            true_labels = tf.convert_to_tensor(true_labels, dtype=tf.float32)

        loss_function = tf.keras.losses.CategoricalCrossentropy(reduction="none")
        self.loss_ite += 1

        if self.loss_ite <= self.warm_up_ite:
            # If the model is not "warmed up"
            return tf.reduce_mean(loss_function(found_labels, true_labels))

        else:
            custom_loss = loss_function(found_labels, true_labels) # Estimate loss
            v = tf.cast(custom_loss < self.threshold, dtype=tf.float32) # Find examples with a smaller loss then the threshold
            self.threshold *= self.growth_factor # Update the threshold
            return tf.reduce_mean(v*custom_loss) 


def is_empty(array):
    """ Checks if the elements in the array are empty or not.
    Input:  array -> np.array(np.array()), list of lists.
    Output: empty -> bool, True if all the lists in array are
                empty, False otherwise.
    """

    # Check that array is an array or a list
    try:
        assert(isinstance(array, (list, np.ndarray)))

    except AssertionError:
        print("Error in function is_empty: the input must be a list or an numpy array.")
        exit(1)

    for element in array: 
        # Check that element is an array or a list with exception
        try: 
            if len(element) != 0:
                return False
        except TypeError:
            print("Error in function is_empty: the elements of the input must be lists or numpy arrays.")
            exit(1)

    return True


def find_indices(labels):
    """ Returns the indices for the examples of each class given
    the labels. The label of the first class must be 0.
    Input:  labels -> tf.tensor[int] | np.array[int], list of labels 
                associated with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: indices -> list[np.array[int]], list of indices for each class. 
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        labels = np.argmax(labels, axis=1)

    nb_classes = np.max(labels)+1
    indices = [None]*nb_classes

    for i in range(nb_classes):
        # Find indices for each class
        indices[i] = np.where(labels == i)[0]

    return indices


def find_examples(data, labels):
    """ Returns the examples for each class given a data set
    and the associated labels. The label of the first class must be 0.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int] | np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: examples -> list[tf.tensor[float32]], list of examples.
                For each element in the list:
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        labels = np.argmax(labels, axis=1)

    indices = find_indices(labels)
    nb_classes = np.max(labels)+1
    examples = [None]*nb_classes

    for i in range(nb_classes):
        # Extract examples for each class from the data
        examples[i] = tf.gather(data, indices[i], axis=0)

    return examples


def find_class_nb(labels, multiclass=True):
    """ Finds the highest class number. The first class must be 0.
    Input:  labels -> tf.tensor[int] | np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
            multiclass -> bool, True if more than 2 classes.
    Output: class_nb -> int, number of class number.
    """

    if not multiclass:
        return 2

    if tf.is_tensor(labels):
        # If labels are one hot
        return np.max(np.argmax(labels, axis=1))+1

    else:
        return np.max(labels)+1


def average_examples(data, labels):
    """ Calculates the average examples for each class given
    the train data and labels. The label of the first class must be 0.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int] | np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: average_data -> tf.tensor[float32], average examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        labels = np.argmax(labels, axis=1)

    nb_classes = np.max(labels)+1
    averages = np.zeros((nb_classes, data[0].shape[0], data[0].shape[1], data[0].shape[2]), dtype=np.float32)
    examples = find_examples(data, labels)

    for i in range(nb_classes):
        # Calculate the mean for each class
        averages[i] = np.mean(examples[i], axis=0) 

    return tf.stack(averages)


def display(acc_list, display_labels, times):
    """ Displays the test accuracies and the times.
    Input:  acc_list -> list[np.array[float32]], list of accuracies for
                each strategy. The last element of each accuracy list must
                be the test accuracy.
            display_labels -> list[str], list of labels associated with each
                strategy.
            times -> np.array[float32], list of times associated with each
                strategy.
    Output:
    """

    try:
        assert(len(acc_list) == len(display_labels) == len(times))

    except AssertionError:
        print("Error in function display: the inputs must all have the same dimension.")
        exit(1)

    # Get longest label for justification
    if len(max(display_labels, key=len)) > len("strategy"):
        max_label_len = len(max(display_labels, key=len))

    else:
        max_label_len = len("strategy")

    print("\nStrategy".ljust(max_label_len, " "), " | Test accuracies | Times")

    for i in range(len(acc_list)):
        label_str = display_labels[i].ljust(max_label_len, " ") + " |" 
        acc_str = str(acc_list[i][-1])[:4].ljust(len("test accuracies"), " ") + " |"
        
        print(label_str, acc_str, "%.2f" % times[i], "s")
