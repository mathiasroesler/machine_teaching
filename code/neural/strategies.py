#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the main functions for different strategies. 
Date: 21/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from data_fct import prep_data
from custom_fct import *
from selection_fct import *
from init_fct import *


def classic_training(train_data, train_labels, test_data, test_labels, class_nb=0,  batch_size=32, epochs=10, multiclass=False, threshold=np.finfo(np.float32).max, growth_rate=1.0):
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
            class_nb -> int, class selected for the one vs all.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
            multiclass -> bool, True if more than 2 classes.
            threshold -> float, value below which SPL
                examples will be used for backpropagation.
            growth_factor -> float, multiplicative value to 
                increase the threshold.
    Output: accuracies -> np.array[float32], accuracies for the training and the test. 
    """
 
    # Get number of classes
    max_class_nb = find_class_nb(train_labels, multiclass)

    try:
        assert(np.issubdtype(type(class_nb), np.integer))
        assert(class_nb >= 0)

    except AssertionError:
        print("Error in function classic_training: class_nb must be an integer smaller than the number of classes.")
        exit(1)

    # Convert labels to one hot and declare model
    train_labels = prep_data(train_labels, class_nb, multiclass)
    model = model_init(train_data[0].shape, max_class_nb, threshold, growth_rate)
    
    hist = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.array(np.append(hist.history.get('accuracy'), [test_score[1]]), dtype=np.float32)


### MACHINE TEACHING ###


def create_teacher_set(train_data, train_labels, exp_rate, target_acc, class_nb=0, batch_size=32, epochs=10, multiclass=False):
    """ Produces the optimal teaching set given the train_data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int] | tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, label | one hot label.
            exp_rate -> int, coefficiant for the exponential distribution
                for the thresholds.
            target_acc -> float, accuracy at which to stop the algorithm. 
            class_nb -> int, class selected for the one vs all.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
            multiclass -> bool, True if more than 2 classes.
    Output: teaching_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], list of labels associated with the
                teaching data.
            teaching_set_len -> np.array[int], number of examples in teaching set at
                each iteration.
    """

    rng = default_rng() # Set seed 

    # Get number of classes
    max_class_nb = find_class_nb(train_labels, multiclass)

    try:
        assert(np.issubdtype(type(class_nb), np.integer))
        assert(class_nb >= 0)

    except AssertionError:
        print("Error in function create_teacher_set: class_nb must be an integer smaller than the number of classes.")
        exit(1)

    # Variables
    ite = 1
    nb_examples = train_data.shape[0]
    accuracy = 0
    thresholds = rng.exponential(1/exp_rate, size=(nb_examples)) # Threshold for each example
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    added_indices = np.array([], dtype=np.intc) # List to keep track of indices of examples already added 

    # Initial example indices
    #init_indices = rndm_init(train_labels)
    init_indices = nearest_avg_init(train_data, train_labels)

    # Create model and convert labels to one hot
    train_labels = prep_data(train_labels, class_nb, multiclass)
    model = model_init(train_data[0].shape, max_class_nb)

    # Initialize teaching data and labels
    teaching_data = tf.gather(train_data, init_indices)
    teaching_labels = tf.gather(train_labels, init_indices)
    added_indices = np.concatenate([added_indices, init_indices], axis=0)

    # Initialize the model
    hist = model.fit(teaching_data, teaching_labels, batch_size=2, epochs=epochs)
    accuracy = hist.history.get('accuracy')[-1]

    while len(teaching_data) != len(train_data):
        # Exit if all of the examples are in the teaching set or enough examples in the teaching set

        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.where(tf.norm(np.round(model.predict(train_data), 1)-train_labels, axis=1) != 0)[0]

        if missed_indices.size == 0 or accuracy >= target_acc:
            # All examples are placed correctly or sufficiently precise
            break

        # Find indices of examples that could be added
        new_indices = select_examples(missed_indices, thresholds, weights)
        #new_indices = select_rndm_examples(missed_indices, 200)
        #new_indices = select_min_avg_dist(missed_indices, 200, train_data, train_labels, positive_average, negative_average)

        # Find the indices of the examples already present and remove them from the new ones
        new_indices = np.setdiff1d(new_indices, added_indices)
        added_indices = np.concatenate([added_indices, new_indices], axis=0)

        if len(new_indices) != 0: 
            # Update the data for the model 
            data = tf.gather(train_data, new_indices)
            labels = tf.gather(train_labels, new_indices)

            # Add data and labels to teacher set and set length to list
            teaching_data = tf.concat([teaching_data, data], axis=0)
            teaching_labels = tf.concat([teaching_labels, labels], axis=0)
            teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)

        hist = model.fit(data, labels, batch_size=batch_size, epochs=epochs) 
        accuracy = hist.history.get('accuracy')[-1]

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
            labels -> tf.tensor[int] np.array[int], list of labels 
                associated with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: curriculum_indices -> list[np.array[int]], list of indices 
                sorted from easy to hard.
    """

    # Get number of classes
    max_class_nb = find_class_nb(labels)

    easy_indices = np.array([], dtype=np.intc)
    hard_indices = np.array([], dtype=np.intc)
    
    classes = np.random.choice(range(max_class_nb), max_class_nb, replace=False)
    averages = average_examples(data, labels)  # List of average example for each class
    indices = find_indices(labels)       # List of indices of examples for each class
    examples = find_examples(data, labels)     # List of examples for each class

    for i in classes:
        dist = tf.norm(averages[i]-examples[i], axis=(1, 2)) # Estimate distance to average
        score = tf.norm(dist-np.mean(dist, axis=0), axis=1)
        easy_indices = np.concatenate([easy_indices, indices[i][np.where(score <= np.median(score))[0]]], axis=0)
        hard_indices = np.concatenate([hard_indices, indices[i][np.where(score > np.median(score))[0]]], axis=0)
        
    return [easy_indices, hard_indices]


def curriculum_training(train_data, train_labels, test_data, test_labels, class_nb=0, batch_size=32, epochs=10, multiclass=False):
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
            multiclass -> bool, True if more than 2 classes.
    Output: accuracies -> np.array[float], accuracy at each epoch of training and
                for the test set.
    """

    # Get number of classes
    max_class_nb = find_class_nb(train_labels, multiclass)

    try:
        assert(np.issubdtype(type(class_nb), np.integer))
        assert(class_nb >= 0)

    except AssertionError:
        print("Error in function curriculum_training: class_nb must be an integer smaller than the number of classes.")
        exit(1)


    model = model_init(train_data[0].shape, max_class_nb) # Declare model
    acc_hist = np.array([], dtype=np.float32)  # List for the accuracies
    curriculum_indices = two_step_curriculum(train_data, train_labels)

    # Convert train labels to one hot labels
    train_labels = prep_data(train_labels, class_nb, multiclass)

    # Train model with easy then hard examples
    for i in range(len(curriculum_indices)):
        hist = model.fit(tf.gather(train_data, curriculum_indices[i]), tf.gather(train_labels, curriculum_indices[i]), batch_size=batch_size, epochs=epochs//2)
        acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 

    # Test the model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.concatenate((acc_hist, [accuracy[1]]), axis=0)

