#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the main functions for different strategies. 
Date: 7/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from data_fct import prep_data
from custom_fct import *
from selection_fct import *
from init_fct import *


def classic_training(train_data, train_labels, test_data, test_labels, class_nb=0,  batch_size=32, epochs=10, multiclass=False, shuffle=True):
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
            shuffle -> bool, if True the train data is shuffle at train.
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
    model = model_init(train_data[0].shape, max_class_nb)
    
    hist = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.array(np.append(hist.history.get('accuracy'), [test_score[1]]), dtype=np.float32)


### MACHINE TEACHING ###


def create_teacher_set(train_data, train_labels, lam_coef, set_limit, class_nb=0, batch_size=32, epochs=10, multiclass=False):
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
            lam_coef -> int, coefficiant for the exponential distribution
                for the thresholds.
            set_limit -> int, maximum number of examples to be put in the 
                teaching set.
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
    old_acc = 0
    eps = 10e-3
    nb_examples = train_data.shape[0]
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0

    # Initial example indices
    #init_indices = rndm_init(train_labels)
    init_indices = nearest_avg_init(train_data, train_labels)

    # Create model and convert labels to one hot
    train_labels = prep_data(train_labels, class_nb, multiclass)
    model = model_init(train_data[0].shape, max_class_nb)

    # Initialize teaching data and labels
    teaching_data = tf.gather(train_data, init_indices)
    teaching_labels = tf.gather(train_labels, init_indices)

    # Initialize the model
    model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs)

    while len(teaching_data) != nb_examples and len(teaching_data) < set_limit:
        # Exit if all of the examples are in the teaching set or enough examples in the teaching set

        # Find all the missed examples indices
        missed_indices = np.where(tf.norm(model.predict(train_data)-train_labels, axis=1) == 0)[0]
    
        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        # Find indices of examples to add
        #added_indices = select_rndm_examples(missed_indices, 200)
        added_indices = select_examples(missed_indices, thresholds, weights)
        #added_indices = select_min_avg_dist(missed_indices, 200, train_data, train_labels, positive_average, negative_average)
        data = tf.gather(train_data, added_indices)
        labels = tf.gather(train_labels, added_indices)

        # Update the teacher set and add length to list
        teaching_data = tf.concat([teaching_data, data], axis=0)
        teaching_labels = tf.concat([teaching_labels, labels], axis=0)
        teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)
    
        # Update the model 
        hist = model.fit(data, labels, batch_size=batch_size, epochs=epochs) 
        new_acc = np.sum(hist.history.get('accuracy'))/epochs

        # Remove used data and labels
        train_data = tf.convert_to_tensor(np.delete(train_data, added_indices, axis=0))
        train_labels = tf.convert_to_tensor(np.delete(train_labels, added_indices, axis=0))
        thresholds = np.delete(thresholds, added_indices, axis=0)
        weights = np.ones(shape=len(train_data))/len(train_data) # Reset weights

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
    Output: easy_indices -> np.array[int], list indices associated
                with the easy examples of the data.
            hard_indices -> np.array[int], list of hard indices associatied
                with the hard examples of the data.
    """

    # Get number of classes
    max_class_nb = find_class_nb(labels)

    easy_indices = np.array([], dtype=np.intc)
    hard_indices = np.array([], dtype=np.intc)
    
    classes = np.random.choice(range(max_class_nb), max_class_nb, replace=False)
    averages = average_examples(data, labels)  # List of average example for each class
    examples = find_examples(data, labels)     # List of examples for each class

    for i in classes:
        dist = tf.norm(averages[i]-examples[i], axis=(1, 2)) # Estimate distance to average

        easy_indices = np.concatenate([easy_indices, np.where(dist <= np.mean(dist))[0]], axis=0)
        hard_indices = np.concatenate([hard_indices, np.where(dist > np.mean(dist))[0]], axis=0)
        
    return easy_indices, hard_indices


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
    easy_indices, hard_indices = two_step_curriculum(train_data, train_labels)

    # Convert train labels to one hot labels
    train_labels = prep_data(train_labels, class_nb, multiclass)

    # Train model with easy then hard examples
    hist = model.fit(tf.gather(train_data, easy_indices), tf.gather(train_labels, easy_indices), batch_size=batch_size, epochs=epochs//2)
    acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 
    hist = model.fit(tf.gather(train_data, hard_indices), tf.gather(train_labels, hard_indices), batch_size=batch_size, epochs=epochs//2)
    acc_hist = np.concatenate((acc_hist, hist.history.get('accuracy')), axis=0) 

    # Test the model
    accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return np.concatenate((acc_hist, [accuracy[1]]), axis=0)


### SELF-PACED CURRICULUM ###


def create_spc_set(train_data, train_labels, loop_ite=1, class_nb=0,  batch_size=32, epochs=10, multiclass=False):
    """ Creates the data set for self-paced curriculum learning.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, label.
            loop_ite -> int, number of loop iterations for selecting examples.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
            class_nb -> int, class selected for the one vs all.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
            multiclass -> bool, True if more than 2 classes.
    Output: data -> tf.tensor[float32], list of examples for training.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, label.
    """

    ite = 0

    # Get number of classes
    max_class_nb = find_class_nb(train_labels, multiclass)

    try:
        assert(np.issubdtype(type(class_nb), np.integer))
        assert(class_nb >= 0)

    except AssertionError:
        print("Error in function spc_training: class_nb must be an integer smaller than the number of classes.")
        exit(1)

    # Extract inital examples indices
    init_indices = nearest_avg_init(train_data, train_labels)

    # Create model and convert labels to one hot
    train_labels = prep_data(train_labels, class_nb, multiclass)
    model = model_init(train_data[0].shape, max_class_nb)
    
    # Initialize data and labels
    data = tf.gather(train_data, init_indices)
    labels = tf.gather(train_labels, init_indices)

    # Remove used examples from train data and labels
    train_data = tf.convert_to_tensor(np.delete(train_data, init_indices, axis=0))
    train_labels = tf.convert_to_tensor(np.delete(train_labels, init_indices, axis=0))

    # Seperate data using two step curriculum
    easy_indices, hard_indices = two_step_curriculum(train_data, train_labels) 
    easy_data = tf.gather(train_data, easy_indices)
    easy_labels = tf.gather(train_labels, easy_indices)
    hard_data = tf.gather(train_data, hard_indices)
    hard_labels = tf.gather(train_labels, hard_indices)

    # Initialize the model
    model.fit(data, labels, batch_size=2, epochs=epochs)

    while ite != loop_ite:
        # Find indices to add
        easy_added_indices = np.where(tf.norm(model.predict(easy_data)-easy_labels, axis=1) == 0)[0]
        easy_added_data = tf.gather(easy_data, easy_added_indices)
        easy_added_labels = tf.gather(easy_labels, easy_added_indices)

        hard_added_indices = np.where(tf.norm(model.predict(hard_data)-hard_labels, axis=1) == 0)[0]
        hard_added_data = tf.gather(hard_data, hard_added_indices)
        hard_added_labels = tf.gather(hard_labels, hard_added_indices)

        added_data = tf.concat([easy_added_data, hard_added_data], axis=0)
        added_labels = tf.concat([easy_added_labels, hard_added_labels], axis=0) 

        # Update the data and labels
        data = tf.concat([data, added_data], axis=0)
        labels = tf.concat([labels, added_labels], axis=0)

        # Update model
        model.fit(added_data, added_labels, batch_size=batch_size, epochs=epochs)

        # Remove used data and labels
        easy_data = tf.convert_to_tensor(np.delete(easy_data, easy_added_indices, axis=0))
        easy_labels = tf.convert_to_tensor(np.delete(easy_labels, easy_added_indices, axis=0))
        hard_data = tf.convert_to_tensor(np.delete(hard_data, hard_added_indices, axis=0))
        hard_labels = tf.convert_to_tensor(np.delete(hard_labels, hard_added_indices, axis=0))

        ite+=1

    # Add remaining data and labels
    data = tf.concat([data, easy_data, hard_data], axis=0)
    labels = tf.concat([labels, easy_labels, hard_labels], axis=0)

    return data, labels

