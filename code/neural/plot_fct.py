#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Plot functions.
Date: 22/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import sys
import os
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors
from misc_fct import *
from scipy.spatial import distance


def plot_train_acc(acc_dict, plot_types):
    """ Plots the training accuracies for each strategy.

    Input:  acc_dict -> dict(str: np.array[float32]), dictionnary of
                accuracies for each strategy.
            plot_types -> list[str], list of plot color and type
                for each strategy.
    Output:

    """
    try:
        assert(len(acc_dict) == len(plot_types)) 

    except AssertionError:
        print("Error in function plot_train_acc: the inputs must all have the "
                "same dimension.")
        exit(1)

    strategy_nb = len(acc_dict)
    i = 0

    for key, value in acc_dict.items():
        plt.plot(range(1, len(value)+1), value, plot_types[i], label=key)
        i+=1

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    
    plt.show()


def plot_test_acc(acc_dict):
    """ Plots the comparaison of the test accuracies for each strategy.

    Input:  acc_list -> dict(str:np.array[float32]), dictionnary of
                accuracies for each strategy.
    Output:

    """
    plt.boxplot(list(acc_dict.values()), labels=list(acc_dict.keys()))

    plt.show()


def plot_losses(train_loss_dict, val_loss_dict, plot_types):
    """ Plots the train and validation losses.

    Input:  train_loss_dict -> dict(str:np.array[float32]), dictionnary
                of the training loss for each strategy.
            val_loss_dict -> dict(str:np.array[float32]), dictionnary
                of the validation loss for each strategy.
            plot_types -> list[str], list of plot color and type for
                each strategy.
    Output:

    """
    try:
        assert(len(train_loss_dict) == len(val_loss_dict) == len(plot_types)) 
        assert(train_loss_dict.keys() == val_loss_dict.keys())

    except AssertionError:
        print("Error in function plot_losses: the inputs must all have the "
                "same dimension and the dictionnaries must have the same keys.")
        exit(1)

    i = 0

    for key in train_loss_dict.keys():
        plt.figure()
        plt.plot(range(1, len(train_loss_dict.get(key))+1), 
                train_loss_dict.get(key), plot_types[i], label="Train "+key)
        plt.plot(range(1, len(val_loss_dict.get(key))+1), 
                val_loss_dict.get(key), plot_types[i]+'-', 
                label="Validation "+key)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, which="both")

        i+=1

    plt.show() 


def plot_confusion(conf_mat_dict, data_name):
    """ Plots the confusion matrix.

    The data is either cifar or mnist.
    Input:  conf_mat_dict -> dict(str:tf.tensor[tf.int32]), dictionnary
            of the confusion matrix of each strategy.
            data_name -> str, name of the data.
    Output:

    """
    try:
        assert(data_name == 'cifar' or data_name == 'mnist')

    except:
        print("Error in function plot_confusion: the data must be cifar or \
                mnist.")
        exit(1)

    if data_name == 'cifar':
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                'frog', 'horse', 'ship', 'truck']

    else:
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for key in conf_mat_dict.keys():
        plt.figure()
        ax = plt.gca()
        ax.set_title(key)

        im = ax.imshow(conf_mat_dict[key])
        
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Show all ticks and label them
        ax.set_xticks(np.arange(conf_mat_dict[key].shape[1]))
        ax.set_yticks(np.arange(conf_mat_dict[key].shape[0]))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Horizontal labeling on top
        ax.tick_params(top=False, labeltop=False, 
                bottom=True, labelbottom=True)

        # Rotate the tick labels and set their alignement
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', 
                rotation_mode='anchor')

    plt.show()
    
#######################################################################

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("usage: plot.py file_name data_name")
        exit(1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = dir_path+ "/" + sys.argv[1]

    try:
        assert(os.path.isfile(file_name))

    except AssertionError:
        print("Error: the selected file was not found")
        exit(1)

    dict_list = []
    confusion = True
    string = ""
    plot_types = ['r-', 'b-', 'g-', 'k-']

    with open(file_name, "rb") as f:
            dict_list = pkl.load(f)

            try:
                conf_mat_dict = pkl.load(f)

            except EOFError:
                confusion = False

    if len(dict_list[1]) == 1:
        plot_types = ['r-']

    plot_train_acc(dict_list[1], plot_types)
    plot_test_acc(dict_list[2])
    plot_losses(dict_list[3], dict_list[4], plot_types)

    if confusion:
        # If the confusion matrices were added
        plot_confusion(conf_mat_dict, sys.argv[2])
