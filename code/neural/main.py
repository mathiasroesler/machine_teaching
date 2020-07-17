#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program.
Date: 16/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import time
import sys
import os
from data_fct import *
from misc_fct import *
from plot_fct import *
from custom_model import * 


if __name__ == "__main__":
    """ Main function.

    Two databases are available, mnist and cifar.
    Input:  database -> str, name of the database used.
            filename -> str, file used for saving results.
    Output: 

    """
    if len(sys.argv) != 3:
        print("usage: main.py database filename")
        exit(1)

    data_name = sys.argv[1]

    try:
        # Check that the database is correct
        assert(data_name == 'cifar' or data_name == 'mnist')

    except AssertionError:
        print("Error: the data name must be cifar or mnist.")
        exit(1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path+ "/" + sys.argv[2]

    # Variables for machine teaching
    exp_rate = 150

    # Variables for self-paced learning
    warm_up = 10 
    threshold = 0.6
    growth_rate = 1.3

    # Variables for neural networks
    archi_type = 1
    epochs = 10
    batch_size = 128

    # Variables for plotting
    plot_types = ['ro-', 'bo-', 'go-', 'ko-']

    # Other variables
    strat_names = ["CL"]#["Full", "MT", "CL", "SPL"]
    iteration_nb = 3 
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
    normalizer_dict = dict()

    # Extract data from files
    train_data, test_data, train_labels, test_labels = extract_data(data_name)

    try:
        assert(not sparse)

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
        file_name = data_name + "_indices.npy"
        optimal_indices = np.load(file_name)

    except FileNotFoundError:
        optimal_indices = create_teacher_set(train_data, train_labels,
                exp_rate, target_acc=0.95, batch_size=batch_size, epochs=2)

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
        model_dict[strat] = CustomModel(data_shape, max_class_nb, archi_type,
                warm_up, threshold, growth_rate)
        train_loss_dict[strat] = np.zeros(epochs)
        val_loss_dict[strat] = np.zeros(epochs)
        normalizer_dict[strat] = np.zeros(epochs)

    for i in range(iteration_nb):
        print("\nITERATION", i+1)

        for strat in strat_names:
            # For each strategy 
            model = model_dict.get(strat)
            tic = time.time()

            # Train model
            print("\n"+ strat+" training")

            if strat == "MT":
                model.train(optimal_data, optimal_labels, val_set, strat,
                        epochs, batch_size, verbose)

            else:
                model.train(train_data, train_labels, val_set, strat, epochs,
                        batch_size, verbose)

            toc = time.time()

            # Test model
            model.test(test_data, test_labels)

            # Save accuracy, losses and 
            train_acc_dict.update({strat: update_dict(
                train_acc_dict.get(strat), model.train_acc)})
            train_loss_dict.update({strat: update_dict(
                train_loss_dict.get(strat), model.train_loss)})
            val_loss_dict.update({strat: update_dict(
                val_loss_dict.get(strat), model.val_loss)})
            normalizer_dict.get(strat)[np.argwhere(train_acc_dict.get(
                strat))] += 1
            time_dict.update({strat: time_dict.get(strat) + toc-tic})

            # Reset model
            model.reset_model(data_shape, archi_type)

    # Average time, accuracies and losses
    for strat in strat_names:
        time_dict.update({strat: np.round(
            time_dict.get(strat)/iteration_nb, decimals=2)})
        train_acc_dict.update({strat: np.round(
            train_acc_dict.get(strat)[train_acc_dict.get(strat) != 0] / 
            normalizer_dict.get(strat)[normalizer_dict.get(strat) != 0],
            decimals=2)})
        train_loss_dict.update({strat: np.round(
            train_loss_dict.get(strat)[train_loss_dict.get(strat) != 0] /
            normalizer_dict.get(strat)[normalizer_dict.get(strat) != 0],
            decimals=2)})
        val_loss_dict.update({strat: np.round(
            val_loss_dict.get(strat)[val_loss_dict.get(strat) != 0] /
            normalizer_dict.get(strat)[normalizer_dict.get(strat) != 0],
            decimals=2)})
        test_acc_dict[strat] = model_dict.get(strat).test_acc

    # Save results in file
    with open(filename, 'w') as f:
        f.write(str(time_dict) + '\n')
        f.write(str(train_acc_dict) + '\n')
        f.write(str(test_acc_dict) + '\n')
        f.write(str(train_loss_dict) + '\n')
        f.write(str(val_loss_dict) + '\n')
