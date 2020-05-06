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
from custom_fct import *


def student_model(data_shape):
    """ Returns the student model used.
    Input:  data_shape -> tuple[int], shape of the input data. 
    Output: model -> CNN model
    """

    model = tf.keras.models.Sequential() # Sequential neural network

    # Add layers to model
    if (data_shape == (28, 28, 1)):
        # Pad the input to be 32x32
        model.add(ZeroPadding2D(2, input_shape=data_shape))

    model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten(data_format='channels_last'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy']
                )

    return model


def teacher_initialization(model, data, labels, positive_index, negative_index, batch_size=32, epochs=10):
    """ Initializes the student model and the teaching set.
    Input:  model -> cnn model, student model.
            data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
            positive_index -> int, index of the positive example to initialize the
                teacher set.
            negative_index -> int, index of the negative example to initialize the
                teacher set.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: model -> cnn model, student model fitted with the data.
            teaching_data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            teaching_labels -> tf.tensor[int], list of labels associated
                with the teaching data.
                First dimension, number of examples.
                Second dimension, one hot label.
    """

    data_shape = data[0].shape
    positive_example = data[positive_index] 
    negative_example = data[negative_index]

    # Add labels to teaching labels
    teaching_labels = np.concatenate(([labels[positive_index]], [labels[negative_index]]), axis=0)

    # Reshape examples to concatenate them
    positive_example = tf.reshape(positive_example, shape=(1, data_shape[0], data_shape[1], 1))
    negative_example = tf.reshape(negative_example, shape=(1, data_shape[0], data_shape[1], 1))

    # Concatenate examples
    teaching_data = tf.concat([positive_example, negative_example], axis=0)   
    model.fit(teaching_data, teaching_labels, batch_size=batch_size, epochs=epochs) 

    return model, teaching_data, teaching_labels


def rndm_init(labels):
    """ Randomly selects an index from the positive and the negative
    index pool.
    Input:  labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: positive_index -> int, selected positive example index.
            negative_index -> int, selected negative example index.
    """

    positive_indices, negative_indices = find_indices(labels)

    rng = default_rng() # Set seed 

    # Find a random positive example 
    positive_index = rng.choice(positive_indices)
    
    # Find a random negative example
    negative_index = rng.choice(negative_indices)

    return positive_index, negative_index
