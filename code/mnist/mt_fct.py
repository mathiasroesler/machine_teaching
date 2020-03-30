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
from scipy.spatial import distance


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
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    teaching_set_len = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    missed_set_len = np.array([], dtype=np.intc) # List of number of misclassified examples at each iteration

    model = student_model(model_type) # Declare student model

    if model == None:
        # If the model was not created
        print("Error in function teacher_initialization: the model was not created.")
        exit(1)

    model, teaching_data, teaching_labels = teacher_initialization(model, model_type, train_data, train_labels, batch_size=batch_size, epochs=epochs)
    positive_average, negative_average = average_examples(model_type, train_data, train_labels)

    while len(teaching_data) != nb_examples and len(teaching_data) < set_limit:
        # Exit if all of the examples are in the teaching set or enough examples in the teaching set
        missed_indices = np.array([], dtype=np.intc) # List of the indices of the missclassified examples
        added_indices = np.array([], dtype=np.intc)  # List of the indices of added examples to the teaching set
        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.unique(np.nonzero(model.predict(train_data) != train_labels)[0])
        missed_set_len = np.concatenate((missed_set_len, [len(missed_indices)]), axis=0)

        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        added_indices = select_rndm_examples(missed_indices, 20)
        #added_indices = select_examples(missed_indices, thresholds, weights)
        #added_indices = select_min_avg_dist(model_type, missed_indices, 200, train_data, train_labels, positive_average, negative_average)

        teaching_data, teaching_labels = update_teaching_set(model_type, teaching_data, teaching_labels, train_data, train_labels, added_indices)
        
        model, curr_accuracy = update_model(model, model_type, teaching_data, teaching_labels, test_data, test_labels, batch_size=batch_size, epochs=epochs)

        # Test model accuracy
        accuracy = np.concatenate((accuracy, [curr_accuracy]), axis=0)
        teaching_set_len = np.concatenate((teaching_set_len, [len(teaching_data)]), axis=0)

        # Remove train data and labels, weights and thresholds of examples in the teaching set
        train_data = np.delete(train_data, added_indices, axis=0)
        train_labels = np.delete(train_labels, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)
        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, accuracy, teaching_set_len, missed_set_len


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
        positive_indices = np.nonzero(train_labels[:, 0] == 0)[0]
        negative_indices = np.nonzero(train_labels[:, 0] == 1)[0]

    else:
        positive_indices = np.nonzero(train_labels == 1)[0]
        negative_indices = np.nonzero(train_labels == 0)[0]

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


def update_teaching_set(model_type, teaching_data, teaching_labels, train_data, train_labels, added_indices):
    """ Updates the teaching data and labels using the train data and labels and the indices of the examples
    to be added depending on the model type.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int] or tf.one_hot, list of labels associated with
                the teaching data.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int] or tf.one_hot, list of labels associated with
                the train data.
            added_indices -> np.array[int], list of indices of examples to be added to the
                teaching set.
    Output: teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int] or tf.one_hot, list of labels associated with
                the teaching data.
    """

    if model_type == 'cnn':
        # For the neural network
        teaching_data = tf.concat([teaching_data, tf.gather(train_data, added_indices)], axis=0)
        teaching_labels = tf.concat([teaching_labels, tf.gather(train_labels, added_indices)], axis=0)

    else:
        # For the svm
        teaching_data = np.concatenate((teaching_data, train_data[added_indices]), axis=0)
        teaching_labels = np.concatenate((teaching_labels, train_labels[added_indices]), axis=0)

    return teaching_data, teaching_labels


def update_model(model, model_type, teaching_data, teaching_labels, test_data, test_labels, batch_size=32, epochs=10):
    """ Updates the student model using the teaching data and labels and evaluates the new 
    performances with the test data and labels.
    Input:  model -> svm or cnn model, student model.
            model_type -> str, {'svm', 'cnn'} model used for the student.
            teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], list of labels associated with
                the teaching data.
            test_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with
                the test data.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: model -> svm or cnn model, student model fitted with the train data.
            accuracy -> float, score obtained with the updated model on the test data.
    """

    if model_type == 'cnn':
        # Update model
        model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 

        # Test the updated model
        accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)

        return model, accuracy[1]

    else:
        # Update model
        model.fit(teaching_data, teaching_labels.ravel())

        # Test the updated model
        accuracy = model.score(test_data, test_labels)

        return model, accuracy


def select_examples(missed_indices, thresholds, weights):
    """ Selects the indices of the examples to be added to the teaching set where
    the weight of the example is greater than its threshold.
    Input:  missed_indices -> np.array[int], list of indices of missclassified
                examples.
            thresholes -> np.array[int], threshold for each example.
            weights -> np.array[int], weight of each example.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    while np.sum(weights[missed_indices]) < 1:
        weights[missed_indices] = 2*weights[missed_indices]
        added_indices = np.nonzero(weights[missed_indices] >= thresholds[missed_indices])[0]

    return added_indices


def select_rndm_examples(missed_indices, max_nb):
    """ Selects randomly the indices of the examples to be added to the teaching set.
    Input:  missed_indices -> np.array[int], list of indices of missclassified
                examples.
            max_nb -> int, maximal number of examples to add.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    rng = default_rng() # Set random seed

    if len(missed_indices) > max_nb:
        added_indices = rng.choice(missed_indices, max_nb, replace=False)

    else:
        added_indices = missed_indices

    return added_indices


def select_min_avg_dist(model_type, missed_indices, max_nb, train_data, train_labels, positive_average, negative_average):
    """ Selects the max_nb nearest examples to the averages. 
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            missed_indices -> np.array[int], list of indices of missclassified
                examples.
            max_nb -> int, maximal number of examples to add.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
            positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """
   
    if model_type == 'cnn':
    # For cnn student model
        # Extract misclassified examples and labels 
        missed_data = tf.gather(train_data, missed_indices, axis=0)
        missed_labels = tf.gather(train_labels, missed_indices, axis=0)

        # Extract indices of positive and negative misclassified examples
        positive_indices = np.nonzero(missed_labels[:, 0] == 0)[0]
        negative_indices = np.nonzero(missed_labels[:, 0] == 1)[0]

        # Extract positive and negative missclassified examples
        positive_examples = tf.gather(missed_data, positive_indices, axis=0)
        negative_examples = tf.gather(missed_data, negative_indices, axis=0)

        if max_nb//2 < positive_indices.shape[0]:
            positive_dist = tf.norm(positive_examples-negative_average, axis=(1, 2))
            positive_indices = np.argpartition(positive_dist, max_nb//2, axis=0)[:max_nb//2]

        if max_nb//2 < negative_indices.shape[0]:
            negative_dist = tf.norm(negative_examples-positive_average, axis=(1, 2))
            negative_indices = np.argpartition(negative_dist, max_nb//2, axis=0)[:max_nb//2]

    else:
    # For svm student model
        # Extract misclassified examples and labels 
        missed_data = train_data[missed_indices]
        missed_labels = train_labels[missed_indices]

        # Extract indices of positive and negative misclassified examples
        positive_indices = np.nonzero(missed_labels == 1)[0]
        negative_indices = np.nonzero(missed_labels == 0)[0]

        # Extract positive and negative missclassified examples
        positive_examples = missed_data[positive_indices]
        negative_examples = missed_data[negative_indices]
    
        if max_nb//2 < positive_indices.shape[0]:
            positive_dist = np.linalg.norm(positive_examples-negative_average, axis=1)
            positive_indices = np.argpartition(positive_dist, max_nb//2, axis=0)[:max_nb//2]

        if max_nb//2 < negative_indices.shape[0]:
            negative_dist = np.linalg.norm(negative_examples-positive_average, axis=1)
            negative_indices = np.argpartition(negative_dist, max_nb//2, axis=0)[:max_nb//2]

    return np.concatenate((np.squeeze(positive_indices), np.squeeze(negative_indices)), axis=0)


def average_examples(model_type, train_data, train_labels):
    """ Calculates the average positive and negative examples given
    the train data and labels.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
    Output: positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
    """

    if model_type == 'cnn':
    # For cnn student model
        positive_examples = tf.gather(train_data, np.nonzero(train_labels[:, 0] == 0)[0], axis=0)
        negative_examples = tf.gather(train_data, np.nonzero(train_labels[:, 0] == 1)[0], axis=0)
        return tf.constant(np.mean(positive_examples, axis=0)), tf.constant(np.mean(negative_examples, axis=0))

    else:
    # For svm student model
        positive_examples = train_data[np.nonzero(train_labels == 1)[0]]
        negative_examples = train_data[np.nonzero(train_labels == 0)[0]]
        return np.mean(positive_examples, axis=0), np.mean(negative_examples, axis=0)