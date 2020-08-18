#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program.
Date: 18/8/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import os
import sys
import time
import pickle as pkl
from data_fct import *
from misc_fct import *
from plot_fct import *
from custom_model import * 


if __name__ == "__main__":
    """ Main function.

    Two databases are available, mnist and cifar.
    The results are pickled into the input file file_name.
    Input:  database -> str, name of the database used.
            file_name -> str, file used for saving results.
    Output: 

    """
    if len(sys.argv) != 3:
        print("usage: main.py database file_name")
        exit(1)

    data_name = sys.argv[1]

    try:
        # Check that the database is correct
        assert(data_name == 'cifar' or data_name == 'mnist')

    except AssertionError:
        print("Error: the data name must be cifar or mnist.")
        exit(1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = dir_path+ "/" + sys.argv[2]

    # Variables for machine teaching
    exp_rate = 0.0005

    # Variables for self-paced learning
    warm_up = 150 
    threshold = 1.5
    growth_rate = 1.5

    # Variables for neural networks
    archi_type = 1
    epochs = 5
    batch_size = 128

    # Other variables
    strat_names = ["Full", "MT", "CL", "SPL"]
    iteration_nb = 1
    class_nb = -1  # Class number for one vs all
    sparse = False  # If labels are to be sparse or not
    verbose = 2  # Verbosity for learning

    # Dictionnaries 
    time_dict = dict()
    train_acc_dict = dict()
    test_acc_dict = dict()
    train_loss_dict = dict()
    val_loss_dict = dict()
    model_dict = dict()
    conf_mat_dict = dict()

    # Extract data from files
    train_data, test_data, train_labels, test_labels = extract_data(data_name)

    try:
        assert(sparse)

    except AssertionError:
        # Convert the labels to one hot
        if class_nb != -1:
            train_labels = tf.keras.utils.to_categorical(
                    prep_labels(train_labels, class_nb=class_nb), 2)
            test_labels = tf.keras.utils.to_categorical(
                    prep_labels(test_labels, class_nb=class_nb), 2)

        else:
            train_labels = tf.keras.utils.to_categorical(train_labels, 10)
            test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # Find the optimal set indices or extract indices from file
    try:
        indices_file = "{}_{}_indices.npy".format(data_name, archi_type)
        optimal_indices = np.load(indices_file)

    except FileNotFoundError:
        _, optimal_indices = create_teacher_set(train_data, train_labels,
                exp_rate, target_acc=0.95, file_name=indices_file, 
                archi_type=archi_type, epochs=2, batch_size=batch_size)

    # Create data sets
    train_set, optimal_set, val_set = split_data((train_data, train_labels),
            optimal_indices)
    train_data = train_set[0]
    train_labels = train_set[1]
    optimal_data = optimal_set[0]
    optimal_labels = optimal_set[1]

    # Data information variables
    max_class_nb = find_class_nb(train_labels) 
    data_shape = train_data[0].shape

    for strat in strat_names:
        # Initialize dictionaries
        time_dict[strat] = 0
        train_acc_dict[strat] = np.zeros(epochs)
        test_acc_dict[strat] = np.zeros(iteration_nb)
        model_dict[strat] = CustomModel(data_shape, max_class_nb, archi_type,
                warm_up, threshold, growth_rate)
        train_loss_dict[strat] = np.zeros(epochs)
        val_loss_dict[strat] = np.zeros(epochs)
        conf_mat_dict[strat] = np.zeros(max_class_nb)

    for i in range(iteration_nb):
        print("\nITERATION {}".format(i+1))

        for strat in strat_names:
            # For each strategy 
            model = model_dict.get(strat)
            tic = time.time()

            # Train model
            print("\n{} training".format(strat))

            if strat == "MT":
                model.train(optimal_data, optimal_labels, strat, val_set,
                        epochs, batch_size, verbose)

            else:
                model.train(train_data, train_labels, strat, val_set, epochs,
                        batch_size, verbose)

            toc = time.time()

            # Test model
            model.test(test_data, test_labels)

            conf_mat_dict[strat] = tf.math.confusion_matrix(np.argmax(
                test_labels, axis=1), np.argmax(model.model.predict(test_data),
                    axis=1))

            # Save accuracy and losses
            train_acc_dict.update({strat: model.train_acc})
            train_loss_dict.update({strat: model.train_loss})
            val_loss_dict.update({strat: model.val_loss})
            time_dict.update({strat: time_dict.get(strat) + toc-tic})
            test_acc_dict[strat] = model_dict.get(strat).test_acc

            # Reset model
            model.reset_model(data_shape, archi_type)

    # Average time
    for strat in strat_names:
        time_dict.update({strat: np.round(
            time_dict.get(strat)/iteration_nb, decimals=2)})

    # Save results in file
    with open(file_name, 'wb') as f:
        pkl.dump([time_dict, train_acc_dict, test_acc_dict, train_loss_dict,
            val_loss_dict], f)
        pkl.dump(conf_mat_dict, f)
