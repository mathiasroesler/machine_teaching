#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the initialization of the teacher.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D, Flatten
from numpy.random import default_rng 
from sklearn import svm


def student_model(model_type, max_iter=5000, dual=False):
    """ Returns the student model used. The default is svm model. 
    Input:  model_type -> str, {'svm', 'cnn'} type of model used for the student.
            max_iter -> int, maximum number of iteration for the SVM if no convergence
                is found.
            dual -> bool, selects dual or primal optimization problem for the SVM.
    Output: model -> SVM or CNN model
    """

    if model_type == 'cnn':
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

    return svm.LinearSVC(dual=dual, max_iter=max_iter) # SVM model


def teacher_initialization(model, model_type, train_data, train_labels, positive_index, negative_index, batch_size=32, epochs=10):
    """ Initializes the student model and the teaching set.
    Input:  model -> svm or cnn model, student model.
            model_type -> str, {'svm', 'cnn'} model used for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with
                the train data.
            positive_index -> int, index of the positive example to initialize the
                teacher set.
            negative_index -> int, index of the negative example to initialize the
                teacher set.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: model -> svm or cnn model, student model fitted with the train data.
            teaching_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], labels associated with the teaching data.
    """

    positive_example = train_data[positive_index] 
    negative_example = train_data[negative_index]

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


def rndm_init(positive_indices, negative_indices):
    """ Randomly selects an index from the positive and the negative
    index pool.
    Input:  positive_indices -> np.array[int], list of indices for the 
                positive examples in a data set.
            negative_indices -> np.array[int], list of indices for the
                negative examples in a data set.
    Output: positive_index -> int, selected positive example index.
            negative_index -> int, selected negative example index.
    """

    rng = default_rng() # Set seed 

    # Find a random positive example 
    positive_index = rng.choice(positive_indices)
    
    # Find a random negative example
    negative_index = rng.choice(negative_indices)

    return positive_index, negative_index


def min_avg_init(model_type, positive_indices, negative_indices, positive_average, negative_average, train_data):
    """ Selects the index of the positive example closest to the negative average and the 
    index of the negative example closest to the positive average.
    Input:  model_type -> str, {'svm', 'cnn'} type of model used for the student.
            positive_indices -> np.array[int], list of indices for the 
                positive examples in a data set.
            negative_indices -> np.array[int], list of indices for the
                negative examples in a data set.
            positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
    Output: positive_index -> int, selected positive example index.
            negative_index -> int, selected negative example index.
    """

    if model_type == 'cnn':
        # For the neural network
        positive_examples = tf.gather(train_data, positive_indices, axis=0)
        negative_examples = tf.gather(train_data, negative_indices, axis=0)

        positive_dist = tf.norm(positive_examples-negative_average, axis=(1, 2))
        negative_dist = tf.norm(negative_examples-positive_average, axis=(1, 2))

    else:
        # For the svm
        positive_examples = train_data[positive_indices]
        negative_examples = train_data[negative_indices]

        positive_dist = np.linalg.norm(positive_examples-negative_average, axis=1)
        negative_dist = np.linalg.norm(negative_examples-positive_average, axis=1)

    return positive_indices[np.argmin(positive_dist)], negative_indices[np.argmin(negative_dist)]
