#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom tensorflow neural network model and
extra functions used for different strategies.
Date: 11/6/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D, Flatten, Softmax, GlobalAveragePooling2D, Dropout
from data_fct import prep_data
from misc_fct import *
from selection_fct import *
from init_fct import *


class CustomModel(tf.keras.Model):
    """ Custom neural network class. """


    def __init__(self, data_shape, class_nb, archi_type=1, warm_up=5, threshold_value=0.4, growth_rate_value=1.1):
        """ Initializes the model.
        Input:  data_shape -> tuple[int], shape of the input data. 
                class_nb -> int, number of classes.
                archi_type -> int, architecture type: 1 for LeNet5, 2 for All-CNN.
                warm_up -> int, batch number after which the model is "warmed up".
                threshold_value -> float32, initial value for SPL threshold.
                growth_rate_value -> float32, initial value for SPL growth rate.
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

        except AssertionError:
            print("Error in init function of CustomModel: data_shape must have 3 elements with the two first ones equal.")
            exit(1)

        super(CustomModel, self).__init__()

        # Class attributes
        self.class_nb = class_nb
        self.model = self.set_model(data_shape, archi_type=archi_type) # Add layers to model

        # Model attributes
        if archi_type == 1:
            # If the LeNet 5 architecture is used
            self.optimizer = tf.keras.optimizers.Adam()
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        else:
            # If the All-CNN architecture is used
            self.optimizer = tf.keras.optimizers.Adam(1e-5)
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        # SPL loss attributes
        self.warm_up = 5
        self.threshold_value = threshold_value
        self.growth_rate_value = growth_rate_value
        self.threshold = tf.Variable(np.finfo(np.float32).max, trainable=False, dtype=tf.float32)
        self.growth_rate = tf.Variable(1, trainable=False, dtype=tf.float32)
        self.v = tf.Variable(np.zeros(shape=(32,)), trainable=False, dtype=tf.float32)

        # Accuracy attributes
        self.train_acc = np.array([], dtype=np.float32)
        self.test_acc = np.array([], dtype=np.float32)


    def set_model(self, input_shape, archi_type=1):
        """ Creates a model with a different architecture depending on the 
        type.
        Input:  input_shape -> tuple[int], shape of the input data. 
                archi_type -> int, architecture type: 1 for LeCun, 2 for full convolution.
        Output: model -> tf.keras.models.Sequential, structured model.
        """

        try:
            assert(np.issubdtype(type(archi_type), np.integer))
            assert(archi_type != 1 or archi_type != 2)

        except AssertionError:
            print("Error in set_model function of CustomModel: archi_type must be 1 or 2") 
            print("Defaulted value to 1")
            archi_type = 1

        model = tf.keras.models.Sequential() # Sequential neural network

        if archi_type == 1:
            # Add layers to model LeNet5
            if (input_shape[0] == 28):
                # Pad the input to be 32x32
                model.add(ZeroPadding2D(2, input_shape=input_shape))
                input_shape = (input_shape[0]+4, input_shape[1]+4, input_shape[2])

            model.add(Conv2D(6, (5, 5), activation='relu', input_shape=input_shape, kernel_initializer='random_normal'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(16, (5, 5), activation='relu', kernel_initializer='random_normal'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten(data_format='channels_last'))
            model.add(Dense(120, activation='relu', kernel_initializer='random_normal'))
            model.add(Dense(84, activation='relu', kernel_initializer='random_normal'))
            model.add(Dense(self.class_nb, activation='softmax', kernel_initializer='random_normal'))

        if archi_type == 2:
            # Add layers to model All-CNN
            if (input_shape[0] == 28):
                # Pad the input to be 32x32
                model.add(ZeroPadding2D(2, input_shape=input_shape))
                input_shape = (input_shape[0]+4, input_shape[1]+4, input_shape[2])

            model.add(Dropout(0.2))
            model.add(Conv2D(96, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer='random_normal'))
            model.add(Conv2D(96, (3, 3), activation='relu', kernel_initializer='random_normal'))
            model.add(Conv2D(96, (3, 3), 2, activation='relu', kernel_initializer='random_normal'))
            model.add(Dropout(0.5))
            model.add(Conv2D(192, (3, 3), activation='relu', kernel_initializer='random_normal'))
            model.add(Conv2D(192, (3, 3), activation='relu', kernel_initializer='random_normal'))
            model.add(Conv2D(192, (3, 3), 2, activation='relu', kernel_initializer='random_normal'))
            model.add(Dropout(0.5))
            model.add(Conv2D(192, (3, 3), activation='relu', kernel_initializer='random_normal'))
            model.add(Conv2D(192, (1, 1), activation='relu', kernel_initializer='random_normal'))
            model.add(Conv2D(10, (1, 1), activation='relu', kernel_initializer='random_normal'))
            model.add(GlobalAveragePooling2D())
            model.add(Softmax()) 

        return model


    def reset_model(self, input_shape, archi_type=1):
        """ Resets the weights of the model.
        Input:  input_shape -> tuple[int], shape of the input data. 
                archi_type -> int, architecture type: 1 for LeCun, 2 for full convolution.
        Output:
        """

        self.model = self.set_model(input_shape, archi_type)
    

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

        try:
            assert(epochs > 1)
            assert(np.issubdtype(type(epochs), np.integer))

        except AssertionError:
            print("Error in function train of CustomModel: the number of epochs must be an integer greater than 1")
            exit(1)


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


    def SPL_train(self, train_data, train_labels, epochs=10, batch_size=32):
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

        try:
            assert(epochs > 1)
            assert(np.issubdtype(type(epochs), np.integer))

        except AssertionError:
            print("Error in function SPL_train of CustomModel: the number of epochs must be an integer greater than 1")
            exit(1)

        # Compile model
        self.model.compile(loss=self.SPL_loss,
                    optimizer=self.optimizer,
                    metrics=['accuracy']
                    )

        # Define callback functions
        batch_callback = tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: self.assign(batch)
                )
        epoch_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.assign(epoch, reset=True)
                )

        hist = self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,
                callbacks=[batch_callback, epoch_callback])
    
        self.train_acc = np.array(hist.history.get('accuracy'))
        

    def SPL_loss(self, labels, predicted_labels):
        """ Calculates the loss for SPL training given the data and the labels.
        Input:  labels -> np.array[int], list of labels associated
                    with the data.
                predicted_labels -> np.array[int], list of labels estimated
                    by the model.
        Output: loss_value -> tf.tensor[float32], calculated loss.
        """

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        loss_value = loss_object(y_true=labels, y_pred=predicted_labels) # Estimate loss
        self.v.assign(tf.cast(loss_value < self.threshold, dtype=tf.float32)) # Find examples with a smaller loss then the threshold
        self.threshold.assign(self.threshold*self.growth_rate) # Update the threshold
        return tf.reduce_mean(v*loss_value) 

    
    def assign(self, batch, reset=False):
        """ Assigns the values to the threshold and the growth
        rate for the SPL training or resets them.
        Input:  batch -> tf.int32, batch number.
                reset -> bool, resets the threshold and the 
                    growth rate if True.
        Output: 
        """

        if reset == True:
            # At the end of each epoch reset values
            self.threshold.assign(self.threshold_value)
            self.growth_rate.assign(self.growth_rate_value)

        elif batch == self.warm_up and self.threshold.numpy() >= np.finfo(np.float32).max:
            # After warm up change threshold and growth rate values 
            self.threshold.assign(self.threshold_value)
            self.growth_rate.assign(self.growth_rate_value)

       
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
        self.test_acc = np.append(self.test_acc, score[1])


def create_teacher_set(train_data, train_labels, exp_rate, target_acc=0.9, batch_size=32, epochs=10):
    """ Produces the optimal teaching set given the train_data and a lambda coefficiant. 
    The search stops if the accuracy of the model is greater than target_acc. The indices
    are saved to a file named by the user and returned as well.
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
    Output: added_indices -> np.array[int], list of indices of selected  examples.
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
    print("Select name to save data to:")
    data_name = input().rstrip()

    np.save(data_name, added_indices)

    return added_indices


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
