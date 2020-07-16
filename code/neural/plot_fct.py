#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Plot functions.
Date: 16/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import sys
import os
import numpy as np
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
    
#######################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: plot.py filename")
        exit(1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path+ "/" + sys.argv[1]

    try:
        assert(os.path.isfile(filename))

    except AssertionError:
        print("Error: the selected file was not found")
        exit(1)

    dict_list = []
    string = ""
    plot_types = ['r-', 'b-', 'g-', 'k-']

    with open(filename, "r") as f:
        for line in f:
            new_line = line.replace("array", "np.array")


            if new_line[-2] != '}':
                string = string + new_line.strip('\n')

            elif string != "":
                string = string + new_line.strip('\n')
                dict_list.append(eval(string))
                string = ""
            
            else:
                dict_list.append(eval(new_line))

    if len(dict_list[1]) == 1:
        plot_types = ['r-']

    plot_train_acc(dict_list[1], plot_types)
    plot_test_acc(dict_list[2])
    plot_losses(dict_list[3], dict_list[4], plot_types)
