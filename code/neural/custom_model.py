#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom tensorflow neural network model and
extra functions used for different strategies.
Date: 22/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D
from tensorflow.keras.layers import Flatten, Softmax, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from data_fct import prep_labels
from misc_fct import *
from selection_fct import *
from init_fct import *


class CustomModel(object):
    """ Custom neural network class. """


    def __init__(self, data_shape, class_nb, archi_type=1, warm_up=50,
            threshold_value=0.4, growth_rate_value=1.1):
        """ Initializes the model.

        The architecture depends on the variable archi_type.
        1 for LeNet5, 2 for All-CNN, 3 for CNN.
        Input:  data_shape -> tuple[int], shape of the input data. 
                class_nb -> int, number of classes.
                archi_type -> int, selects the architecture,
                    default value 1.
                warm_up -> int, batch number after which the model is
                    "warmed up", default value 50.
                threshold_value -> float32, initial value for SPL 
                    threshold, default value 0.4.
                growth_rate_value -> float32, initial value for SPL 
                    growth rate, default value 1,1.
        Output: 

        """
        try:
            assert(np.issubdtype(type(class_nb), np.integer))
            assert(class_nb > 1)

        except AssertionError:
            print("Error in init function of CustomModel: class_nb must be an "
                    "integer greater or equal to 2.")
            exit(1)

        try:
            assert(len(data_shape) == 3)
            assert(data_shape[0] == data_shape[1])

        except AssertionError:
            print("Error in init function of CustomModel: data_shape must have "
                    " 3 elements with the two first ones equal.")
            exit(1)

        # Class attributes
        self.class_nb = class_nb
        self.model = self.set_model(data_shape, archi_type=archi_type) 

        # Model attributes
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()

        # SPL loss attributes
        self.warm_up = warm_up
        self.threshold_value = threshold_value
        self.growth_rate_value = growth_rate_value
        self.threshold = tf.Variable(np.finfo(np.float32).max,
                trainable=False, dtype=tf.float32)
        self.growth_rate = tf.Variable(1, trainable=False, dtype=tf.float32)

        # Accuracy attributes
        self.train_acc = np.array([], dtype=np.float32)
        self.test_acc = np.array([], dtype=np.float32)
        self.val_acc = np.array([], dtype=np.float32)
        
        # Loss attributes
        self.train_loss = np.array([], dtype=np.float32)
        self.val_loss = np.array([], dtype=np.float32)


    def set_model(self, input_shape, archi_type=1):
        """ Creates a model.

        The architecture depends on the variable archi_type.
        1 for LeNet5, 2 for All-CNN, 3 for CNN.
        Input:  input_shape -> tuple[int], shape of the input data. 
                archi_type -> int, selects the architecture,
                    default value 1. 
        Output: model -> tf.keras.models.Sequential, structured model.

        """
        try:
            assert(np.issubdtype(type(archi_type), np.integer))
            assert(archi_type != 1 or archi_type != 2 or archi_type != 3)

        except AssertionError:
            print("Error in set_model function of CustomModel: archi_type must "
                    "be 1, 2 or 3") 
            print("Defaulted value to 1")
            archi_type = 1

        model = tf.keras.models.Sequential() # Sequential neural network

        if archi_type == 1:
            # Add layers to model LeNet5
            if (input_shape[0] == 28):
                # Pad the input to be 32x32
                model.add(ZeroPadding2D(2, input_shape=input_shape))
                input_shape = (input_shape[0]+4, 
                        input_shape[1]+4, input_shape[2])

            model.add(Conv2D(6, (5, 5), activation='relu',
                input_shape=input_shape))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(16, (5, 5), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten(data_format='channels_last'))
            model.add(Dense(120, activation='relu'))
            model.add(Dense(84, activation='relu'))
            model.add(Dense(self.class_nb, activation='softmax'))

        if archi_type == 2:
            # Add layers to model All-CNN
            if (input_shape[0] == 28):
                # Pad the input to be 32x32
                model.add(ZeroPadding2D(2, input_shape=input_shape))
                input_shape = (input_shape[0]+4, input_shape[1]+4,
                        input_shape[2])

            model.add(Dropout(0.2))
            model.add(Conv2D(48, (3, 3), activation='relu',
                input_shape=input_shape))
            model.add(Conv2D(48, (3, 3), activation='relu'))
            model.add(Conv2D(48, (3, 3), 2, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Conv2D(96, (3, 3), activation='relu'))
            model.add(Conv2D(96, (3, 3), activation='relu'))
            model.add(Conv2D(96, (3, 3), 2, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Conv2D(96, (3, 3), activation='relu'))
            model.add(Conv2D(96, (1, 1), activation='relu'))
            model.add(Conv2D(self.class_nb, (1, 1), activation='relu'))
            model.add(GlobalAveragePooling2D())
            model.add(Softmax()) 

        if archi_type == 3:
            # Add layers to model CNN
            if (input_shape[0] == 28):
                # Pad the input to be 32x32
                model.add(ZeroPadding2D(2, input_shape=input_shape))
                input_shape = (input_shape[0]+4, input_shape[1]+4,
                        input_shape[2])

            model.add(Conv2D(32, (3, 3), activation='relu',
                input_shape=input_shape))
            model.add(MaxPool2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPool2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.class_nb, activation='softmax'))

        return model


    def reset_model(self, input_shape, archi_type=1):
        """ Resets the weights of the model.

        The architecture depends on the variable archi_type.
        1 for LeNet5, 2 for All-CNN, 3 for CNN.
        Input:  input_shape -> tuple[int], shape of the input data. 
                archi_type -> int, selects the architecture,
                    default value 1.
        Output:

        """
        # Reset weigths
        self.model = self.set_model(input_shape, archi_type)


    def train(self, train_data, train_labels, strategy, val_set=None, 
            epochs=10, batch_size=32, verbose=0):
        """ Calls the training function.

        The strategy depends on the variable strategy.
        Full -> classic training,
        MT -> classic training with the optimal set,
        CL -> curriculum training,
        SPL -> self-paced training.
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> tf.tensor[int], list of one hot labels
                    associated with the train data.
                strategy -> {'Full', 'MT', 'CL', 'SPL'}, strategy used. 
                val_set -> tuple(tf.tensor[float32], tf.tensor[int]),
                    validation set.
                    First dimension, examples.
                    Second dimension, one hot labels.
                    Default value None.
                epochs -> int, number of epochs for the training of the
                    neural network, default value 10.
                batch_size -> int, number of examples used in a batch 
                    for the neural network, default value 32.
                verbose -> int, amount of printing for the training,
                    default value 0, see Tensorflow for details.
        Output:

        """    
        try:
            assert(isinstance(strategy, str))
            assert(strategy == "MT" or strategy == "CL" or 
                    strategy == "Full" or strategy == "SPL")

        except AssertionError:
            print("Error in function train of CustomModel: the strategy must "
                    "be a string, either Full, MT, CL or SPL")
            exit(1)

        try:
            assert(len(train_labels.shape) != 1)
            assert(train_labels.shape[1] == 10 or train_labels.shape[1] == 2)

        except AssertionError:
            # If the labels are not one hot
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)

        # Initialize arrays
        self.train_acc = np.zeros(epochs, dtype=np.float32)
        self.train_loss = np.zeros(epochs, dtype=np.float32)
        self.val_acc = np.zeros(epochs, dtype=np.float32)
        self.val_loss = np.zeros(epochs, dtype=np.float32)

        if strategy == "MT" or strategy == "Full":
            self.simple_train(train_data, train_labels, val_set, epochs,
                    batch_size, verbose)

        elif strategy == "CL":
            self.CL_train(train_data, train_labels, val_set, epochs, batch_size,
                    verbose)

        elif strategy == "SPL":
            self.SPL_train(train_data, train_labels, val_set, epochs,
                    batch_size, verbose)


    def simple_train(self, train_data, train_labels, val_set, epochs=10, 
            batch_size=32, verbose=0):
        """ Trains the model.
        
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> tf.tensor[int], list of one hot labels 
                    associated with the train data.
                val_set -> tuple(tf.tensor[float32], tf.tensor[int]),
                    validation set.
                    First dimension, examples.
                    Second dimension, one hot labels.
                epochs -> int, number of epochs for the training of the
                    neural network, default value 10.
                batch_size -> int, number of examples used in a batch 
                    for the neural network, default value 32.
                verbose -> int, amount of printing for the training,
                    default value 0, see Tensorflow for details.
        Output:

        """    
        try:
            assert(epochs > 1)
            assert(np.issubdtype(type(epochs), np.integer))

        except AssertionError:
            print("Error in function simple_train of CustomModel: the number "
                   "of epochs must be an integer greater than 1")
            exit(1)


        # Compile model
        self.model.compile(loss=self.loss_function,
                    optimizer=self.optimizer,
                    metrics=['accuracy']
                    )

        # Define callbacks
        stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=1
                )
        
        hist = self.model.fit(
                train_data, train_labels, 
                validation_data=val_set, callbacks=[],
                batch_size=batch_size, epochs=epochs, verbose=verbose
                )
    
        # Save accuracies and losses
        self.train_acc = np.array(hist.history.get('accuracy'))
        self.train_loss = np.array(hist.history.get('loss'))
        self.val_acc = np.array(hist.history.get('val_accuracy'))
        self.val_loss = np.array(hist.history.get('val_loss'))


    def CL_train(self, train_data, train_labels, val_set, epochs=10, 
            batch_size=32, verbose=0):
        """ Trains the model.

        The training uses the curriculum strategy.
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> tf.tensor[int], list of one hot labels 
                    associated with the train data.
                val_set -> tuple(tf.tensor[float32], tf.tensor[int]),
                    validation set.
                    First dimension, examples.
                    Second dimension, one hot labels.
                epochs -> int, number of epochs for the training of the
                    neural network, default value 10.
                batch_size -> int, number of examples used in a batch 
                    for the neural network, default value 32.
                verbose -> int, amount of printing for training
                    default value 0, see Tensorflow for details.
        Output:

        """    
        try:
            assert(epochs > 1)
            assert(np.issubdtype(type(epochs), np.integer))

        except AssertionError:
            print("Error in function CL_train of CustomModel: the number "
                    "of epochs must be an integer greater than 1")
            exit(1)

        # Compile model
        self.model.compile(loss=self.loss_function,
                    optimizer=self.optimizer,
                    metrics=['accuracy'])
        
        # Define callbacks
        stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=1)

        curriculum_indices = two_step_curriculum(train_data, train_labels)
        curriculum_len = len(curriculum_indices)
        half_epoch = epochs//curriculum_len

        # Train model with easy then hard examples
        for i in range(curriculum_len):
            hist = self.model.fit(
                    tf.gather(train_data, curriculum_indices[i]),
                    tf.gather(train_labels, curriculum_indices[i]),
                    validation_data=val_set, callbacks=[],
                    batch_size=batch_size, epochs=half_epoch, verbose=verbose)

            real_epoch = len(hist.history.get('accuracy'))
            origin = i * half_epoch

            # Save accuracies and losses
            self.train_acc[origin:real_epoch+origin] =  hist.history.get(
                    'accuracy')
            self.train_loss[origin:real_epoch+origin] =  hist.history.get(
                    'loss')
            self.val_acc[origin:real_epoch+origin] =  hist.history.get(
                    'val_accuracy')
            self.val_loss[origin:real_epoch+origin] =  hist.history.get(
                    'val_loss')


    def SPL_train(self, train_data, train_labels, val_set, epochs=10,
            batch_size=32, verbose=0):
        """ Trains the model.

        The training uses the self-paced strategy.
        Input:  train_data -> tf.tensor[float32], list of examples. 
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                train_labels -> tf.tensor[int], list of one hot labels
                    associated with the train data.
                val_set -> tuple(tf.tensor[float32], tf.tensor[int]), 
                    validation set.
                    First dimension, examples.
                    Second dimension, one hot labels.
                epochs -> int, number of epochs for the training of the
                    neural network, default value 10.
                batch_size -> int, number of examples used in a batch 
                    for the neural network, default value 32.
                verbose -> int, amount of printing for the training
                    default value 0, see Tensorflow for details.
        Output:

        """    
        try:
            assert(epochs > 1)
            assert(np.issubdtype(type(epochs), np.integer))

        except AssertionError:
            print("Error in function SPL_train of CustomModel: the number "
                    "of epochs must be an integer greater than 1")
            exit(1)

        # Compile model
        self.model.compile(loss=self.SPL_loss,
                    optimizer=self.optimizer,
                    metrics=['accuracy'])

        # Define callback functions
        batch_callback = tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: self.assign(batch))
        epoch_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.assign(epoch, reset=True))
        stop_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=1)

        hist = self.model.fit(
                train_data, train_labels, 
                validation_data=val_set, callbacks=[batch_callback, 
                    epoch_callback],
                batch_size=batch_size, epochs=epochs, verbose=verbose)
    
        # Save accuracies and losses
        self.train_acc = np.array(hist.history.get('accuracy'))
        self.train_loss = np.array(hist.history.get('loss'))
        self.val_acc = np.array(hist.history.get('val_accuracy'))
        self.val_loss = np.array(hist.history.get('val_loss'))
        

    def SPL_loss(self, labels, predicted_labels):
        """ Calculates the loss for SPL training.

        Input:  labels -> tf.tensor[int], list of one hot labels
                    associated with the data.
                predicted_labels -> tf.tensor[int], list of one hot
                    labels estimated by the model.
        Output: loss_value -> tf.tensor[float32], calculated loss.

        """
        loss_object = tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)

        try:
            assert(len(labels.shape) != 1)
            assert(labels.shape[1] == 10 or labels.shape[1] == 2)

        except AssertionError:
            # If the labels are not one hot
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        # Estimate loss and find examples with smaller loss
        loss_value = loss_object(y_true=labels, y_pred=predicted_labels)
        v = tf.cast(loss_value < self.threshold, dtype=tf.float32) 

        return tf.reduce_mean(v*loss_value) 

    
    def assign(self, batch, reset=False):
        """ Assigns the values to the threshold and the growth rate.

        Function called at the end of each epoch during SPL training.
        Input:  batch -> tf.int32, batch number.
                reset -> bool, resets the threshold and the growth 
                    rate if True. Default value False.
        Output: 

        """
        if reset == True:
            # At the end of each epoch update threshold
            self.threshold.assign(self.threshold*self.growth_rate)

        elif batch == self.warm_up and self.threshold.numpy() >= np.finfo(
                np.float32).max:
            # After warm up change threshold and growth rate values 
            self.threshold.assign(self.threshold_value)
            self.growth_rate.assign(self.growth_rate_value)

       
    def test(self, test_data, test_labels, batch_size=32, verbose=0):
        """ Tests the model.

        Input:  test_data -> tf.tensor[float32], list
                    of examples.
                    First dimension, number of examples.
                    Second and third dimensions, image. 
                    Fourth dimension, color channel. 
                test_labels -> tf.tensor[int], list of one hot 
                    labels associated with the test data.
                batch_size -> int, number of examples used in a 
                    batch for the neural network, default value 32.
                verbose -> int, amount of printing for testing,
                    default value 0, see Tensorflow for details.
        Output:

        """
        score = self.model.evaluate(test_data, test_labels, 
                batch_size=batch_size, verbose=verbose)
        self.test_acc = np.append(self.test_acc, score[1])


def create_teacher_set(train_data, train_labels, exp_rate, target_acc=0.9,
        filename="", archi_type=1, epochs=10, batch_size=32):
    """ Produces the optimal teaching set.

    The search stops if the accuracy of the model is greater than 
    target_acc, if the teaching set contains a tenth of the training
    examples or if there are no missclassified examples anymore. 
    The indices are saved to the file given by filename, if no file
    is given then they are not saved. The indices are also returned
    as well. The architecture used for the model depends on the
    variable archi_type.
    1 for LeNet5, 2 for All-CNN, 3 for CNN.
    Input:  train_data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> tf.tensor[int], list of one hot labels
                associated with the train data.
            exp_rate -> int, coefficiant of the exponential distribution
                for the thresholds.
            target_acc -> float, accuracy threshold. 
            filename -> str, file name to save indices,
                default value "".
            archi_type -> int, selects the architecture,
                default value 1.
            epochs -> int, number of epochs for the training of the 
                neural network, default value 10.
            batch_size -> int, number of examples used in a batch for
                the neural network, default value 32.
    Output: added_indices -> np.array[int], list of indices of selected
                examples.

    """
    rng = default_rng() # Set seed 

    # Get number of classes
    max_class_nb = find_class_nb(train_labels)

    # Variables
    ite = 1
    nb_examples = train_data.shape[0]
    accuracy = 0
    thresholds = rng.exponential(1/exp_rate, size=(nb_examples)) 
    teaching_set_len = np.array([0], dtype=np.intc) 
    added_indices = np.array([], dtype=np.intc) 

    # Initial example indices
    init_indices = nearest_avg_init(train_data, train_labels)

    # Declare model
    model = CustomModel(train_data[0].shape, max_class_nb,
            archi_type=archi_type)

    # Initialize teaching data and labels
    teaching_data = tf.gather(train_data, init_indices)
    teaching_labels = tf.gather(train_labels, init_indices)
    added_indices = np.concatenate([added_indices, init_indices], axis=0)

    # Initialize the model
    model.train(teaching_data, teaching_labels, "Full", epochs=epochs,
            batch_size=2)

    while len(teaching_data) != len(train_data):
        # Exit if all of the examples are in the teaching set or enough 
        # examples in the teaching set

        # Weights for each example
        weights = np.ones(shape=(nb_examples))/nb_examples 

        # Find all the missed examples indices
        missed_indices = np.where(np.argmax(model.model.predict(train_data), 
            axis=1) - np.argmax(train_labels, axis=1) != 0)[0]

        if missed_indices.size == 0 or accuracy >= target_acc or \
                len(teaching_data) >= len(train_data) // 10:
            # All examples are placed correctly or sufficiently precise
            break

        # Find indices of examples that could be added
        new_indices = select_examples(missed_indices, thresholds, weights)

        # Find the indices of the examples already present and remove them from
        # the new ones
        new_indices = np.setdiff1d(new_indices, added_indices)
        added_indices = np.concatenate([added_indices, new_indices], axis=0)

        if len(new_indices) != 0: 
            # Update the data for the model 
            data = tf.gather(train_data, new_indices)
            labels = tf.gather(train_labels, new_indices)

            # Add data and labels to teacher set and set length to list
            teaching_data = tf.concat([teaching_data, data], axis=0)
            teaching_labels = tf.concat([teaching_labels, labels], axis=0)
            teaching_set_len = np.concatenate((teaching_set_len, 
                [len(teaching_data)]), axis=0)

        model.train(data, labels, "Full", batch_size=batch_size, epochs=epochs) 
        accuracy = model.train_acc[-1]

        ite += 1

    print("\nIteration number: {}".format(ite))

    if filename != "":
        np.save(filename, added_indices)

    return added_indices


def two_step_curriculum(data, labels):
    """ Creates a curriculum for training.

    The examples in data are seperated into easy and hard examples
    depending on their proximity to the average of their class.
    Input:  data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of one hot labels 
                associated with the data.
    Output: curriculum_indices -> list[np.array[int]], list of indices 
                sorted from easy to hard.

    """
    max_class_nb = find_class_nb(labels)

    easy_indices = np.array([], dtype=np.intc)
    hard_indices = np.array([], dtype=np.intc)
    
    classes = np.random.choice(range(max_class_nb), max_class_nb, replace=False)
    averages = average_examples(data, labels)  
    indices = find_indices(labels) 
    examples = find_examples(data, labels) 

    for i in classes:
        dist = tf.norm(averages[i]-examples[i], axis=(1, 2))
        score = tf.norm(dist-np.mean(dist, axis=0), axis=1)
        easy_indices = np.concatenate([easy_indices, indices[i][
            np.where(score <= np.median(score))[0]]], axis=0)
        hard_indices = np.concatenate([hard_indices, indices[i][
            np.where(score > np.median(score))[0]]], axis=0)
        
    return [easy_indices, hard_indices]
