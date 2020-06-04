#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom tensorflow neural network model and
extra functions used for different strategies.
Date: 04/6/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D, Flatten
from tensorflow.keras.utils import Progbar
from data_fct import prep_data
from misc_fct import *
from selection_fct import *
from init_fct import *


class CustomModel(tf.keras.Model):
    """ Custom neural network class. """


    def __init__(self, data_shape, class_nb):
        """ Initializes the model.
        Input:  data_shape -> tuple[int], shape of the input data. 
                class_nb -> int, number of classes.
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

        super(CustomModel, self).__init__()

        # Class attributes
        self.model = tf.keras.models.Sequential() # Sequential neural network
        self.class_nb = class_nb

        # Loss attributes
        self.threshold = np.finfo(np.float32).max
        self.growth_rate = 1

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
    

    def loss(self, data, labels, threshold, growth_rate, warm_up):
        """ Calculates the loss for SPL training given the data and the labels.
        Input:  data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                labels -> np.array[int], list of labels associated
                    with the data.
                threshold -> float, value below which SPL
                    examples will be used for backpropagation.
                growth_rate -> float, multiplicative value to 
                    increase the threshold.
                warm_up -> bool, when True use SP loss.
        Output: loss_value -> tf.tensor[float32], calculated loss.
        """

        predicted_labels = self.model(data)

        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        if warm_up:
            # If the model is still warming up
            return tf.reduce_mean(loss_function(y_true=labels, y_pred=predicted_labels))

        else:
            loss_value = loss_function(y_true=labels, y_pred=predicted_labels) # Estimate loss
            v = tf.cast(loss_value < self.threshold, dtype=tf.float32) # Find examples with a smaller loss then the threshold
            self.threshold *= self.growth_rate # Update the threshold
            return tf.reduce_mean(v*loss_value) 


    def gradient(self, data, labels, threshold, growth_rate, warm_up):
        """ Calculates the gradients used to optimize the model.
        Input:  data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                labels -> np.array[int], list of labels associated
                    with the data.
                threshold -> float, value below which SPL
                    examples will be used for backpropagation.
                growth_rate -> float, multiplicative value to 
                    increase the threshold.
                warm_up -> bool, when True use SP loss.
        Output: loss_value -> tf.tensor[float32], calculated loss.
                gradients -> list(), list of gradients.
        """

        with tf.GradientTape() as tape:
            loss_value = self.loss(data, labels, threshold, growth_rate, warm_up)

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
    
        self.train_acc = np.array(hist.history.get('accuracy'))


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


    def SPL_train(self, train_data, train_labels, threshold, growth_rate, epochs=10, batch_size=32):
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

        warm_up_ite = 100 # Iteration after which SP gradient is applied
        warm_up = True  # Indicates if the model is warming up or not

        for epoch in range(epochs):
            # For each epoch
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

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
                loss_value, gradients = self.gradient(data, labels, threshold, growth_rate, warm_up) 
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

        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        ite = 0

        for i in range(batch_size, test_data.shape[0]+batch_size-1, batch_size):
            logits = self.model(test_data[ite:i])
            test_accuracy(y_true=test_labels[ite:i], y_pred=logits)

            ite = i

        self.test_acc = test_accuracy.result()


def create_teacher_set(train_data, train_labels, exp_rate, target_acc=0.9, batch_size=32, epochs=10):
    """ Produces the optimal teaching set given the train_data and a lambda coefficiant. 
    The search stops if the accuracy of the model is greater than target_acc.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int], list of labels associated
                with the train data.
            exp_rate -> int, coefficiant for the exponential distribution
                for the thresholds.
            target_acc -> float, accuracy at which to stop the algorithm. 
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
    max_class_nb = find_class_nb(train_labels)

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

    # Declare model
    model = CustomModel(train_data[0].shape, max_class_nb)

    # Initialize teaching data and labels
    teaching_data = tf.gather(train_data, init_indices)
    teaching_labels = tf.gather(train_labels, init_indices)
    added_indices = np.concatenate([added_indices, init_indices], axis=0)

    # Initialize the model
    model.train(teaching_data, teaching_labels, epochs=epochs, batch_size=2)

    while len(teaching_data) != len(train_data):
        # Exit if all of the examples are in the teaching set or enough examples in the teaching set

        weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example

        # Find all the missed examples indices
        missed_indices = np.where(np.argmax(model.model.predict(train_data), axis=1)-train_labels != 0)[0]

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

        model.train(data, labels, batch_size=batch_size, epochs=epochs) 
        accuracy = model.train_acc[-1]

        ite += 1

    print("\nIteration number:", ite)

    return teaching_data, teaching_labels, teaching_set_len


def two_step_curriculum(data, labels):
    """ Creates a curriculum dividing the data into easy and
    hard examples taking into account the classes.
    Input:  data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> np.array[int], list of labels 
                associated with the data.
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
