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


def create_teacher_set(train_data, train_labels, test_data, test_labels, lam_coef, set_limit, batch_size=32, epochs=10):
    """ Produces the optimal teaching set given the train_data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
            test_data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
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
    model = student_model() # Declare student model

    positive_average, negative_average = average_examples(train_data, train_labels)

    positive_index, negative_index = rndm_init(train_labels)
    #positive_index, negative_index = min_avg_init(train_data, train_labels, positive_average, negative_average)

    model, teaching_data, teaching_labels = teacher_initialization(model, train_data, train_labels, positive_index, negative_index, batch_size=batch_size, epochs=epochs)

    # Remove inital examples from the data and labels
    train_data = np.delete(train_data, [positive_index, negative_index], axis=0)
    train_labels = np.delete(train_labels, [positive_index, negative_index], axis=0)

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
        #added_indices = select_min_avg_dist(missed_indices, 200, train_data, train_labels, positive_average, negative_average)
        #added_indices = select_curriculum_examples(200, train_data, train_labels, ite-1)

        teaching_data, teaching_labels = update_teaching_set(teaching_data, teaching_labels, train_data, train_labels, added_indices)
        
        curr_accuracy = update_model(model, teaching_data, teaching_labels, test_data, test_labels, batch_size=batch_size, epochs=epochs)

        # Reset the weights for the cnn model
        model = student_model()

        # Test model accuracy
        accuracy = np.concatenate((accuracy, [curr_accuracy]), axis=0)
        teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)
    
         
        if abs((accuracy[-1] - accuracy[-2])) < 0.001: 
            # If the performances don't change much
            break
         

        # Remove train data and labels, weights and thresholds of examples in the teaching set
        #train_data = np.delete(train_data, added_indices, axis=0)
        #train_labels = np.delete(train_labels, added_indices, axis=0)
        #weights = np.delete(weights, added_indices, axis=0)
        #thresholds = np.delete(thresholds, added_indices, axis=0)
        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, accuracy, teaching_set_len, missed_set_len


def train_student_model(train_data, train_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Trains a student model fitted with the train set and
    tested with the test set. Returns None if an error occured.
    Input:  train_data -> tf.tensor[float32], list
                of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
            test_data -> tf.tensor[float32], list
                of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[float32], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: test_score -> float, score obtained with the test set. 
    """

    model = student_model() # Declare student model
    
    if model == None:
        print("Error in function train_student_model: the model was not created.")
        return None

    print("\nSet length", len(train_data))

    hist = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print("\nTest score", test_score[1])
    return [np.array(hist.history.get('accuracy'), dtype=np.float32), test_score[1]]


def update_teaching_set(teaching_data, teaching_labels, data, labels, added_indices):
    """ Updates the teaching data and labels using the train data and labels and the indices of the examples
    to be added depending on the model type.
    Input:  teaching_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            teaching_labels -> tf.tensor[int], list of labels associated
                with the teaching data.
                First dimension, number of examples.
                Second dimension, one hot label.
            data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
            added_indices -> np.array[int], list of indices of examples to be added to the
                teaching set.
    Output: teaching_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            teaching_labels -> tf.tensor[int], list of labels associated
                with the teaching data.
                First dimension, number of examples.
                Second dimension, one hot label.
    """

    teaching_data = tf.concat([teaching_data, tf.gather(data, added_indices)], axis=0)
    teaching_labels = tf.concat([teaching_labels, tf.gather(labels, added_indices)], axis=0)

    return teaching_data, teaching_labels


def update_model(model, teaching_data, teaching_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Updates the student model using the teaching data and labels and evaluates the new 
    performances with the test data and labels.
    Input:  model -> cnn model, student model.
            teaching_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            teaching_labels -> tf.tensor[int], list of labels associated
                with the teaching data.
                First dimension, number of examples.
                Second dimension, one hot label.
            test_data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracy -> float, score obtained with the updated model on the test data.
    """

    # Update model
    model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 

    # Test the updated model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return accuracy[1]
