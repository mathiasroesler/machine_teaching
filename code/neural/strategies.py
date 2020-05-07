#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the main functions for different strategies. 
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from data_fct import prep_data
from custom_fct import *
from selection_fct import *
from init_fct import *


def classic_training(train_data, train_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Trains a student model fitted with the train set and
    tested with the test set. 
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
    Output: accuracies -> np.array[float32], accuracies for the training and the test. 
    """

    model = model_init(train_data[0].shape) # Declare student model
    
    hist = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.array(np.append(hist.history.get('accuracy'), [test_score[1]]), dtype=np.float32)


### MACHINE TEACHING ###


def create_teacher_set(train_data, train_labels, lam_coef, set_limit, batch_size=32, epochs=10):
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
            lam_coef -> int, coefficiant for the exponential distribution
                for the thresholds.
            set_limit -> int, maximum number of examples to be put in the 
                teaching set.
    Output: teaching_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], list of labels associated with the
                teaching data.
            teaching_set_len -> np.array[int], number of examples in teaching set at
                each iteration.
    """

    rng = default_rng() # Set seed 

    # Variables
    ite = 1
    old_acc = 0
    eps = 10e-3
    nb_examples = train_data.shape[0]
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    model = model_init(train_data[0].shape) # Declare student model

    # Recuperate initial examples
    init_indices = rndm_init(train_labels)
    teaching_data = tf.gather(train_data, init_indices)
    teaching_labels = tf.gather(train_labels, init_indices)

    # Initialize the model
    model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs)

    while len(teaching_data) != nb_examples and len(teaching_data) < set_limit:
        # Exit if all of the examples are in the teaching set or enough examples in the teaching set
        missed_indices = np.array([], dtype=np.intc) # List of the indices of the missclassified examples
        added_indices = np.array([], dtype=np.intc)  # List of the indices of added examples to the teaching set
        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.unique(np.nonzero(np.round(model.predict(train_data)) != train_labels)[0])
    
        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        #added_indices = select_rndm_examples(missed_indices, 200)
        added_indices = select_examples(missed_indices, thresholds, weights)
        #added_indices = select_min_avg_dist(missed_indices, 200, train_data, train_labels, positive_average, negative_average)

        # Update the teacher set and add length to list
        teaching_data = tf.concat([teaching_data, tf.gather(train_data, added_indices)], axis=0)
        teaching_labels = tf.concat([teaching_labels, tf.gather(train_labels, added_indices)], axis=0)
        teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)
    
        # Update model
        hist = model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 
        new_acc = np.sum(hist.history.get('accuracy'))/epochs

        if (abs(new_acc-old_acc) < eps):
            # Avoid overlearning
            break

        old_acc = new_acc

        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, teaching_set_len


### CURRICULUM ###


def two_step_curriculum(data, labels):
    """ Creates a curriculum dividing the data into easy and
    hard examples taking into account the classes.
    Input:  data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, label.
    Output: easy_indices -> np.array[int], list indices associated
                with the easy examples of the data.
            hard_indices -> np.array[int], list of hard indices associatied
                with the hard examples of the data.
    """

    max_class_nb = np.max(labels) # Highest class number
    easy_indices = np.array([], dtype=np.intc)
    hard_indices = np.array([], dtype=np.intc)
    
    classes = np.random.choice(range(max_class_nb+1), max_class_nb+1, replace=False)
    averages = average_examples(data, labels)  # List of average example for each class
    examples = find_examples(data, labels)     # List of examples for each class
    
    for i in classes:
        dist = tf.norm(averages[i]-examples[i], axis=(1, 2)) # Estimate distance to average
        dist = dist/np.max(dist)    # Normalize distances

        easy_indices = np.concatenate([easy_indices, np.where(dist <= np.mean(dist))[0]], axis=0)
        hard_indices = np.concatenate([hard_indices, np.where(dist > np.mean(dist))[0]], axis=0)
        
    return easy_indices, hard_indices


def curriculum_training(train_data, train_labels, test_data, test_labels, class_nb, batch_size=32, epochs=10):
    """ Trains the model using a two step curriculum. 
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, label.
            test_data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_labels -> tf.tensor[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
            class_nb -> int, class selected for the one vs all.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: accuracies -> np.array[float], accuracy at each epoch of training and
                for the test set.
    """

    max_class_nb = np.max(train_labels) # Highest class number

    try:
        assert(isinstance(class_nb, int))
        assert(class_nb >= 0)
        assert(max_class_nb >= class_nb)

    except AssertionError:
        print("Error in function class_training: class_nb must be an integer smaller than the number of classes.")
        exit(1)

    model = model_init(train_data[0].shape) # Student model to train
    acc_hist = np.array([], dtype=np.float32)  # List for the accuracies
    easy_indices, hard_indices = two_step_curriculum(train_data, train_labels)

    # Convert train labels to one hot labels
    train_labels = prep_data(train_labels, class_nb)

    # Train model with easy then hard examples
    hist = model.fit(tf.gather(train_data, easy_indices), tf.gather(train_labels, easy_indices), batch_size=batch_size, epochs=epochs//2)
    acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 
    hist = model.fit(tf.gather(train_data, hard_indices), tf.gather(train_labels, hard_indices), batch_size=batch_size, epochs=epochs//2)
    acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 

    # Test the model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.concatenate((acc_hist, [accuracy[1]]), axis=0)
