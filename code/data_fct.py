#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function to read and edit the data.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import os
import numpy as np
from os.path import isfile
from pathlib import Path
from numpy.random import default_rng 


def extract_data():
    """ Finds the data file in the data directory and extracts the data.
    Input:
    Output: data -> np.array[list[str]]
    """

    cwd = Path(os.getcwd())
    data_dir = str(cwd.parent)+"/data"

    for filename in os.listdir(data_dir):
        # Go through all the files in the data directory
        if filename[-5:] == ".data" and isfile(data_dir+"/"+filename): 
            # If the file contains data extract it 
            f_descriptor = open(data_dir+"/"+filename, 'r')
            data = [line.rstrip().split(',') for line in f_descriptor]
            f_descriptor.close()
    
    return np.asarray(data)


def sort_data(data, nb_classes):
    """ Sorts the different classes of data that contain
    a list of numerical features.
    Input:  data -> np.array[np.array[int]], list of features with
                the last element being the label greater than 0.
            nb_classes -> int, number of classes.
    Output: classes -> np.array[np.array[np.array[int]]], each element is a list
                of the examples for a class.
    """

    nb_elements = len(data[0])
    
    # Check for inconsistencies
    if not isinstance(nb_classes, int) and nb_classes <= 0:
        print("Error in function sort_data: the number of classes must be a positive integer")
        return None

    classes = [np.empty(shape=(0, nb_elements), dtype=np.intc) for i in range(nb_classes)]

    for example in data:
        for i in range(nb_classes):
            # Check the label of the example
            if i+1 == example[-1]:
                # Add the example to the correct class
                classes[i] = np.append(classes[i], [example], axis=0)
                break

    return np.asarray(classes)


def sort_zoo_data(zoo_data, nb_classes):
    """ Sorts the different classes of the zoo data.
    Input:  data -> np.array[list[str]], list of features with
                the last element being the label.
            nb_classes -> int, number of classes.
    Output: classes -> np.array[np.array[np.array[int]]], each element is a list
                of the examples for a class.
    """

    nb_elements = len(zoo_data[0])-1

    data = np.empty(shape=(0, nb_elements), dtype=np.intc)

    for example in zoo_data:
        # Remove animal name and convert str to int
        data = np.append(data, [list(map(int, example[1:]))], axis=0)

    return sort_data(data, nb_classes)


def generate_indices(classes, percent, nb_features):
    """ Generates the indices to create a set.
    Input:  classes -> np.array[np.array[np.array[int]]], each element is a list
                of the examples for a class.
            percent -> int, percent of examples to be put in the set.
            nb_features -> int, number of desired features.
    Output: example_set -> np.array[np.array[int]] -> set for the examples, 
                filled with -1. 
            indices -> np.array[np.array[int]], list of indices for the examples
                to be chosen from each class.
    """

    nb_classes = len(classes)
    lengths = [len(current_class) for current_class in classes] # Number of examples in each class 
    nb_examples = [percent*lengths[i]//100 for i in range(nb_classes)] # Number of examples for the set for each class

    example_set = -np.ones(shape=(np.sum(nb_examples), nb_features+1), dtype=np.intc) # Set of examples for the features and the label

    rng = default_rng() # Set seed
    indices = [rng.choice(range(lengths[i]), nb_examples[i], replace=False) for i in range(nb_classes)]

    return example_set, np.asarray(indices)


def extract_features(classes, features, nb_classes):
    """ Extracts the desired features from the examples of all the classes.
    Input:  classes -> np.array[np.array[np.array[int]]], each element is a list
                of the examples for a class.
            features -> list[int], list of desired features.
            nb_classes -> int, number of classes.
    Output: reduced_classes -> np.array[np.array[np.array[int]]], each element 
                is a list of the examples containing only the desired features 
                for a class. The last element of the example is the label.
    """
    
    features.append(-1) # Add label to the desired features
    nb_features = len(features)

    reduced_classes = list() # All classes with only the desired features

    for i in range(nb_classes):
        reduced_class = np.empty(shape=(0, nb_features), dtype=np.intc) # Class with only the desired features

        for example in classes[i]:
            reduced_class = np.append(reduced_class, [example[features]], axis=0) # Add example with desired features only

        reduced_classes.append(reduced_class) # Add to list of classes

    return np.asarray(reduced_classes)


def create_sets(classes, nb_classes, percent=80, features=None):
    """ Divides the input classes into test and train sets using
        the input features. The first column of the set is the features
        The train set is composed of percent% of the examples.
        Returns None if an error occured.
        Input:  classes -> np.array[np.array[np.array[int]]], each element is a list
                    of the examples for a class.
                nb_classes -> int, number of classes.
                percent -> int, percent of example in a class to be put in
                    the train set.
                features -> int or list[int], features selected in each 
                    example.
        Output: test_set -> np.array[np.array[int]], each row is the features for
                    an example with the last element being the label.
                train_set -> np.array[np.array[int]], each row is the features for
                    an example with the last element being the label.
    """

    nb_features = len(classes[0][0])-1 # Number of features in an example

    if features == None:
        # If no features were specified use all of them (remove label)
        features = [i for i in range(nb_features)]

    # Check for inconsistencies
    if (percent < 0 or percent > 100):
        print("Error in function create_sets: percentage is inconsistent")
        return None, None 
    
    if isinstance(features, int):
        if features > nb_features:
            print("Error in function create_sets: selected features exceeds array")
            return None, None 

        if features < 0:
            print("Error in function create_sets: the features must be a positive integer")
            return None, None

        features = [features] # Convert int feature into a list

    elif isinstance(features, list):
        for feature in features:
            if feature > nb_features:
                print("Error in function create_sets: selected features exceeds array")
                return None, None

    else:
        print("Error in function create_sets: the features argument must be an integer or a list of integers")
        return None, None

    # Create sets and extract desired features
    train_set, train_classes_indices = generate_indices(classes, percent, len(features)) 
    test_set, test_classes_indices = generate_indices(classes, 100-percent, len(features))
    reduced_classes = extract_features(classes, features, nb_classes)
    train_offset= 0
    test_offset = 0

    # Fill the created sets
    for i in range(len(classes)): 
        # Get indices for examples of ith class
        train_set_indices = range(train_offset, train_offset+len(train_classes_indices[i]))
        test_set_indices = range(test_offset, test_offset+len(test_classes_indices[i]))

        # Fill the dedicated space with examples
        train_set[train_set_indices] = reduced_classes[i][train_classes_indices[i]]
        test_set[test_set_indices] = reduced_classes[i][test_classes_indices[i]]

        # Update the offsets
        train_offset = len(train_classes_indices[i])
        test_offset = len(test_classes_indices[i])

    # Shuffle sets
    rng = default_rng() # Set seed
    rng.shuffle(train_set)
    rng.shuffle(test_set)

    return train_set, test_set

