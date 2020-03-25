#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the teacher algorithm.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D, Flatten
from numpy.random import default_rng 
from sklearn import svm

def create_teacher_set(model_type, train_data, train_labels, test_data, test_labels, lam_coef, set_limit, batch_size=32, epochs=10):
    """ Produces the optimal teaching set given the train_data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  model_type -> str, {'svm', 'cnn'} type of model used for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
            test_data -> np.array[np.array[int]] or tf.tensor, list of examples. 
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with the test data.
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
    ite = 0
    nb_examples = train_data.shape[0]
    weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0

    model = student_model(model_type) # Declare student model

    if model == None:
        # If the model was not created
        print("Error in function teacher_initialization: the model was not created.")
        exit(1)

    model, teaching_data, teaching_labels = teacher_initialization(model, model_type, train_data, train_labels, batch_size=batch_size, epochs=epochs)

    while len(teaching_data) != nb_examples and len(teaching_data) < set_limit:
    # Exit if all of the examples are in the teaching set
        missed_indices = np.array([], dtype=np.intc) # List of the indices of the missclassified examples
        added_indices = np.array([], dtype=np.intc)  # List of the indices of added examples to the teaching set
        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.where(model.predict(train_data) != train_labels)[0]

        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        if len(missed_indices) > batch_size:
            added_indices = rng.choice(missed_indices, batch_size, replace=False)

        else:
            added_indices = missed_indices

        teaching_data = np.concatenate((teaching_data, train_data[added_indices]), axis=0)
        teaching_labels = np.concatenate((teaching_labels, train_labels[added_indices]), axis=0)

        # Fit the model with the new teaching set
        model.fit(teaching_data, teaching_labels.ravel())
        
        # Test model accuracy
        accuracy = np.append(accuracy, [model.score(test_data, test_labels)], axis=0)
        teaching_set_len = np.append(teaching_set_len, [len(teaching_data)], axis=0)

        # Remove train data and labels, weights and thresholds of examples in the teaching set
        train_data = np.delete(train_data, added_indices, axis=0)
        train_labels = np.delete(train_labels, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)
        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, accuracy, teaching_set_len


def student_model(model_type, max_iter=5000, dual=False):
    """ Returns the student model used. If an error occurs, the function
    returns None.
    Input:  model_type -> str, {'svm', 'cnn'} type of model used for the student.
            max_iter -> int, maximum number of iteration for the SVM if no convergence
                is found.
            dual -> bool, selects dual or primal optimization problem for the SVM.
    Output: model -> SVM or CNN model
    """

    if model_type == 'svm':
        return svm.LinearSVC(dual=dual, max_iter=max_iter) # SVM model

    elif model_type == 'cnn':
        model = tf.keras.models.Sequential() # Sequential neural network

        # Add layers to model
        model.add(ZeroPadding2D(2, input_shape=(28, 28, 1)))
        model.add(Conv2D(6, (5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (5, 5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten(data_format='channels_last'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                      metrics=['accuracy']
                      )

        return model

    print("Error in function student_model: the input argument must be svm or cnn.")
    return None


def teacher_initialization(model, model_type, train_data, train_labels, batch_size=32, epochs=10):
    """ Initializes the student model and the teaching set.
    Input:  model -> svm or cnn model, student model.
            model_type -> str, {'svm', 'cnn'} model used for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with
                the train data.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: model -> svm or cnn model, student model fitted with the train data.
            teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], labels associated with the teaching data.
    """

    rng = default_rng() # Set seed 
    
    if model_type == 'cnn':
    # The data is one_hot
        positive_indices = np.where(train_labels == [0, 1])[0]
        negative_indices = np.where(train_labels == [1, 0])[0]

    else:
        positive_indices = np.where(train_labels == 1)[0]
        negative_indices = np.where(train_labels == -1)[0]

    # Find a random positive example 
    positive_index = rng.choice(positive_indices)
    positive_example = train_data[rng.choice(positive_indices)] 
    
    # Find a random negative example
    negative_index = rng.choice(negative_indices)
    negative_example = train_data[rng.choice(negative_indices)]

    # Add labels to teaching labels
    teaching_labels = np.concatenate(([train_labels[positive_index]], [train_labels[negative_index]]), axis=0)

    if model_type == 'cnn':
        # Reshape examples to concatenate them
        positive_example = tf.reshape(positive_example, shape=(1, 28, 28, 1))
        negative_example = tf.reshape(negative_example, shape=(1, 28, 28, 1))

        # Concatenate examples
        teaching_data = tf.concat([positive_example, negative_example], axis=0)   
        model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 

    else:
        teaching_data = np.concatenate(([positive_example], [negative_example]), axis=0)
        model.fit(teaching_data, teaching_labels.ravel())

    return model, teaching_data, teaching_labels
