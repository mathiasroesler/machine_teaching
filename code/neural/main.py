#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program.
Date: 11/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import time
from data_fct import *
from misc_fct import *
from plot_fct import *
from custom_model import * 


def main(data_name):
    """ Main function. 
    Input:  data_name -> str {'mnist', 'cifar'}, name of the dataset
            to use.
    """

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
    strat_names = ["Full", "MT", "CL", "SPL"]
    class_nb = -1
    iteration_nb = 3

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

    if class_nb != -1:
        train_labels = tf.keras.utils.to_categorical(prep_labels(train_labels, class_nb=class_nb), 2)
        test_labels = tf.keras.utils.to_categorical(prep_labels(test_labels, class_nb=class_nb), 2)

    else:
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # Find the optimal set indices or extract indices from file
    try:
        file_name = data_name + "_indices.npy"
        optimal_indices = np.load(file_name)

    except FileNotFoundError:
        optimal_indices = create_teacher_set(train_data, train_labels, exp_rate, target_acc=0.95, batch_size=batch_size, epochs=2)

    # Create data sets
    train_set, optimal_set, val_set = split_data((train_data, train_labels), optimal_indices)
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
        train_acc_dict[strat] = [0]
        model_dict[strat] = CustomModel(data_shape, max_class_nb, archi_type, warm_up, threshold, growth_rate)
        train_loss_dict[strat] = [0]
        val_loss_dict[strat] = [0]
        normalizer_dict[strat] = [0]

    for i in range(iteration_nb):
        print("\nITERATION", i+1)

        for strat in strat_names:
            # For each strategy 
            model = model_dict.get(strat)
            tic = time.time()

            # Train model
            print("\n"+ strat+" training")

            if strat == "MT":
                model.train(optimal_data, optimal_labels, val_set, strat, epochs, batch_size)

            else:
                model.train(train_data, train_labels, val_set, strat, epochs, batch_size)

            toc = time.time()

            # Test model
            model.test(test_data, test_labels)

            # Save accuracy, losses and time
            train_acc_dict.update({strat: update_dict(train_acc_dict.get(strat), model.train_acc)})
            train_loss_dict.update({strat: update_dict(train_loss_dict.get(strat), model.train_loss)})
            val_loss_dict.update({strat: update_dict(val_loss_dict.get(strat), model.val_loss)})
            normalizer_dict.update({strat: update_dict(normalizer_dict.get(strat), np.ones(len(model.train_acc)))})
            time_dict.update({strat: time_dict.get(strat) + toc-tic})

            # Reset model
            model.reset_model(data_shape, archi_type)

    # Average time, accuracies and losses
    for strat in strat_names:
        time_dict.update({strat: time_dict.get(strat)/normalizer_dict.get(strat)})
        train_acc_dict.update({strat: train_acc_dict.get(strat)/normalizer_dict.get(strat)})
        train_loss_dict.update({strat: train_loss_dict.get(strat)/normalizer_dict.get(strat)})
        val_loss_dict.update({strat: val_loss_dict.get(strat)/normalizer_dict.get(strat)})
        test_acc_dict[strat] = model_dict.get(strat).test_acc

    # Plot results
    plot_train_acc(train_acc_dict, plot_types)
    plot_test_acc(test_acc_dict)
    plot_losses(train_loss_dict, val_loss_dict, plot_types)

    
print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnist:")
    data_name = input().rstrip()

main(data_name)

