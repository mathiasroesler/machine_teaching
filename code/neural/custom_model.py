#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom tensorflow neural network model.
Date: 25/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D, Flatten
from tensorflow.keras.utils import Progbar
from strategies import *
from misc_fct import *
from data_fct import *


class CustomModel:
    """ Custom neural network class. """


    def __init__(self, data_shape, class_nb, multiclass=False):
        """ Initializes the model.
        Input:  data_shape -> tuple[int], shape of the input data. 
                class_nb -> int, number of classes.
                threshold -> float, value below which SPL
                    examples will be used for backpropagation.
                growth_factor -> float, multiplicative value to 
                    increase the threshold.
        Output: 
        """

        try:
            assert(np.issubdtype(type(class_nb), np.integer))
            assert(class_nb > 1)

        except AssertionError:
            print("Error in init function of CustomModel: class_nb must be an integer greater or equal to 2.")
            exit(1)

        try:
            assert(len(data_shape) == 3)
            assert(data_shape[0] == data_shape[1])
            input_shape = data_shape

        except AssertionError:
            print("Error in init function of CustomModel: data_shape must have 3 elements with the two first ones equal.")
            exit(1)

        # Class attributes
        self.model = tf.keras.models.Sequential() # Sequential neural network
        self.class_nb = class_nb
        self.multiclass = multiclass

        # Loss attributes
        self.threshold = np.finfo(np.float32).max
        self.growth_factor = 1

        # Accuracy attributes
        self.train_acc = np.array([], dtype=np.float32)
        self.test_acc = 0.0

        # Model attributes
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        # Add layers to model
        if (data_shape[0] == 28):
            # Pad the input to be 32x32
            self.model.add(ZeroPadding2D(2, input_shape=input_shape))
            input_shape = (input_shape[0]+4, input_shape[1]+4, input_shape[2])

        self.model.add(Conv2D(6, (5, 5), activation='relu', input_shape=input_shape))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(16, (5, 5), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten(data_format='channels_last'))
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(self.class_nb, activation='softmax'))
    

    def loss(self, data, labels, threshold, growth_factor, warm_up):
        """ Calculates the loss for SPL training given the data and the labels.
        Input:  data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                labels -> np.array[int], list of labels associated
                    with the data.
                threshold -> float, value below which SPL
                    examples will be used for backpropagation.
                growth_factor -> float, multiplicative value to 
                    increase the threshold.
                warm_up -> bool, when True use SP loss.
        Output: loss_value -> tf.tensor[float32], calculated loss.
        """

        predicted_labels = self.model(data)

        loss_function = tf.keras.losses.CategoricalCrossentropy()

        return loss_function(y_true=labels, y_pred=predicted_labels)

        if warm_up:
            # If the model is still warming up
            return tf.reduce_mean(loss_function(y_true=labels, y_pred=predicted_labels))

        else:
            loss_value = loss_function(y_true=labels, y_pred=predicted_labels) # Estimate loss
            v = tf.cast(loss_value < self.threshold, dtype=tf.float32) # Find examples with a smaller loss then the threshold
            self.threshold *= self.growth_factor # Update the threshold
            return tf.reduce_mean(v*loss_value) 


    def gradient(self, data, labels, threshold, growth_factor, warm_up):
        """ Calculates the gradients used to optimize the model.
        Input:  data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                labels -> np.array[int], list of labels associated
                    with the data.
                threshold -> float, value below which SPL
                    examples will be used for backpropagation.
                growth_factor -> float, multiplicative value to 
                    increase the threshold.
                warm_up -> bool, when True use SP loss.
        Output: loss_value -> tf.tensor[float32], calculated loss.
                gradients -> list(), list of gradients.
        """

        with tf.GradientTape() as tape:
            loss_value = self.loss(data, labels, threshold, growth_factor, warm_up)

        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
        

    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        """ Trains the model with the given inputs.
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> np.array[int], list of labels associated
                    with the train data.
                batch_size -> int, number of examples used in a batch for the neural
                    network.
                epochs -> int, number of epochs for the training of the neural network.
        Output:
        """    

        # Compile model
        self.model.compile(loss=self.loss_function,
                    optimizer=self.optimizer,
                    metrics=['accuracy']
                    )
        
        hist = self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
    
        self.train_acc = hist.history.get('accuracy') 


    def CL_train(self, train_data, train_labels, epochs=10, batch_size=32):
        """ Trains the model using a curriculum and the given data.
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> np.array[int], list of labels associated
                    with the train data.
                batch_size -> int, number of examples used in a batch for the neural
                    network.
                epochs -> int, number of epochs for the training of the neural network.
        Output:
        """    

        try:
            assert(epochs > 1)
            assert(np.issubdtype(type(epochs), np.integer))

        except AssertionError:
            print("Error in function CL_train of CustomModel: the number of epochs must be an integer greater than 1")
            exit(1)

        # Reset train accuracy
        self.train_acc = np.array([], dtype=np.float32)

        # Compile model
        self.model.compile(loss=self.loss_function,
                    optimizer=self.optimizer,
                    metrics=['accuracy']
                    )
        
        curriculum_indices = two_step_curriculum(train_data, train_labels)

        # Train model with easy then hard examples
        for i in range(len(curriculum_indices)):
            hist = self.model.fit(tf.gather(train_data, curriculum_indices[i]), tf.gather(train_labels, curriculum_indices[i]), batch_size=batch_size, epochs=epochs//2)
            self.train_acc = np.concatenate((self.train_acc, hist.history.get('accuracy')), axis=0) 


    def SPL_train(self, train_data, train_labels, threshold, growth_factor, epochs=10, batch_size=32):
        """ Trains the model using self-paced training and the given inputs. 
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> np.array[int], list of labels associated
                    with the train data.
                batch_size -> int, number of examples used in a batch for the neural
                    network.
                epochs -> int, number of epochs for the training of the neural network.
        Output:
        """    

        # Reset train accuracy history
        self.train_acc = np.array([], dtype=np.float32)

        warm_up_ite = 1000 # Iteration after which SP gradient is applied
        warm_up = True  # Indicates if the model is warming up or not

        for epoch in range(epochs):
            # For each epoch
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

            ite = 0
            print("Epoch {}/{}".format(epoch+1, epochs))

            bar = Progbar(train_data.shape[0]//batch_size, stateful_metrics=["loss", "accuracy"])

            for i in range(batch_size, train_data.shape[0]+batch_size-1, batch_size):
                # Go through batches of data of size batch_size
                data = train_data[ite:i]
                labels = train_labels[ite:i]

                if i > warm_up_ite*batch_size:
                    # When the model is warmed up
                    warm_up = False

                # Optimize the model
                loss_value, gradients = self.gradient(data, labels, threshold, growth_factor, warm_up) 
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)    # Add current batch loss
                epoch_accuracy.update_state(labels, self.model(data)) # Compare predicted labels with true labels
                ite = i
                
                # Display
                values = [("loss", epoch_loss_avg.result()), ("accuracy", epoch_accuracy.result())]
                bar.update(i//batch_size, values=values)

            # End epoch 
            self.train_acc = np.append(self.train_acc, epoch_accuracy.result())


    def test(self, test_data, test_labels, batch_size=32):
        """ Tests the model with the given data.
            Input:  test_data -> tf.tensor[float32], list
                        of examples.
                        First dimension, number of examples.
                        Second and third dimensions, image. 
                        Fourth dimension, color channel. 
                    test_labels -> np.array[int], list of labels associated
                        with the test data.
                    batch_size -> int, number of examples used in a batch for the neural
                        network.
        """

        score = self.model.evaluate(test_data, test_labels, batch_size=batch_size)
        self.test_acc = score[1]


    def custom_test(self, test_data, test_labels, batch_size=32):
        """ Tests the model when using SPL training.
            Input:  test_data -> tf.tensor[float32], list
                        of examples.
                        First dimension, number of examples.
                        Second and third dimensions, image. 
                        Fourth dimension, color channel. 
                    test_labels -> np.array[int], list of labels associated
                        with the test data.
                    batch_size -> int, number of examples used in a batch for the neural
                        network.
        """

        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        ite = 0

        for i in range(batch_size, test_data.shape[0]+batch_size-1, batch_size):
            logits = self.model(test_data[ite:i])
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            test_accuracy(prediction, test_labels[ite:i])

            ite = i

        self.test_acc = test_accuracy.result()

