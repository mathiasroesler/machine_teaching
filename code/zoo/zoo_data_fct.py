#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function to manipulate the zoo data.
Date: 13/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import os
import numpy as np
from os.path import isfile
from pathlib import Path
from numpy.random import default_rng 


def extract_zoo_data():
    """ Function that extracts the data from the zoo file.

    The zoo.data file is located in a seperate folder named data.
    Input:
    Output: data -> np.array[list[str]], list of examples with the last
                element the label.
                First dimension number of examples.
                Second dimension features.

    """
    data_dir = '../../data'

    for filename in os.listdir(data_dir):
        # Go through all the files in the data directory
        if filename[-5:] == ".data" and isfile(data_dir+"/"+filename): 
            # If the file is the zoo data file extract data 
            f_descriptor = open(data_dir+"/"+filename, 'r')
            data = [line.rstrip().split(',') for line in f_descriptor]
            f_descriptor.close()
    
    return np.asarray(data)


def sort_data(data, nb_classes):
    """ Function that sorts the examples.

    The examples of each class are put in an array.
    Input:  data -> np.array[np.array[int]], list of features with
                the last element the label.
                First dimension number of examples.
                Second dimension features.
            nb_classes -> int, number of classes.
    Output: classes -> np.array[np.array[np.array[int]]], each element
                is a list of the examples for a class.
                First dimension number of classes.
                Second dimension number of examples.
                Third dimension features.

    """
    nb_elements = len(data[0]) # Number of examples
    
    # Check for inconsistencies
    try:
        assert(isinstance(nb_classes, int))
        assert(nb_classes > 0)

    except AssertionError:
        print("Error in function sort_data: the number of classes must be a
        positive interger")
        exit(1)

    classes = [np.empty(shape=(0, nb_elements), dtype=np.intc) for i in 
            range(nb_classes)]

    for example in data:
        for i in range(nb_classes):
            # Check the label of the example
            if i+1 == example[-1]:
                # Add the example to the correct class
                classes[i] = np.append(classes[i], [example], axis=0)
                break

    return np.asarray(classes)


def sort_zoo_data(zoo_data, nb_classes):
    """ Function that sorts the different classes of the zoo data.

    The zoo data is prepared before used to sort it into different
    classes.
    Input:  data -> np.array[list[str]], list of features with
                the last element being the label.
                First dimension number of examples.
                Second dimension features.
            nb_classes -> int, number of classes.
    Output: classes -> np.array[np.array[np.array[int]]], each element is a list
                of the examples for a class.
                First dimension number of classes.
                Second dimension number of examples.
                Third dimension features.

    """
    nb_elements = len(zoo_data[0])-1

    data = np.empty(shape=(0, nb_elements), dtype=np.intc)

    for example in zoo_data:
        # Remove animal name and convert str to int
        data = np.append(data, [list(map(int, example[1:]))], axis=0)

    return sort_data(data, nb_classes)


def generate_indices(classes, percent, nb_features):
    """ Generates the indices to create a set.

    Input:  classes -> np.array[np.array[np.array[int]]], each element 
                is a list of the examples for a class.
                First dimension number of classes.
                Second dimension number of examples.
                Third dimension features.
            percent -> int, percent of examples to be put in the set.
            nb_features -> int, number of desired features.
    Output: example_set -> np.array[np.array[int]] -> set for the 
                examples, filled with -1. 
                First dimension number of examples.
                Second dimension features.
            indices -> np.array[np.array[int]], list of indices for
                the examples to be chosen from each class.
                First dimension number of examples.
                Second dimension features.
    """

    nb_classes = len(classes)
    lengths = [len(current_class) for current_class in classes] 
    nb_examples = [percent*length//100 for length in lengths]

    example_set = -np.ones(shape=(np.sum(nb_examples), nb_features+1),
            dtype=np.intc)

    rng = default_rng() # Set seed
    indices = [rng.choice(range(lengths[i]), nb_examples[i], replace=False)
            for i in range(nb_classes)]

    return example_set, np.asarray(indices)


def extract_features(classes, features, nb_classes):
    """ Extracts the desired features from the examples.

    Input:  classes -> np.array[np.array[np.array[int]]], each element
                is a list of the examples of a class.
                First dimension number of classes.
                Second dimension number of examples.
                Third dimension features.
            features -> np.array[int], list of desired features.
            nb_classes -> int, number of classes.
    Output: reduced_classes -> np.array[np.array[np.array[int]]], each
                element is a list of examples of a class containing only
                the desired features. 
                The last element of the example is the label.
                First dimension number of classes.
                Second dimension number of examples.
                Third dimension features.

    """
    features = np.append(features, [-1], axis=0)
    nb_features = len(features)
    reduced_classes = list()

    # Fill the list with the desired features only
    for i in range(nb_classes):
        reduced_class = np.empty(shape=(0, nb_features), dtype=np.intc)

        for example in classes[i]:
            reduced_class = np.append(reduced_class, [example[features]],
                    axis=0)

        reduced_classes.append(reduced_class)

    return np.asarray(reduced_classes)


def create_sets(classes, nb_classes, percent=80, features=None):
    """ Divides the input classes into test and train sets.
    
    The first column of the set is the features. If the value of 
    features in None then all of the features are selected.
    Input:  classes -> np.array[np.array[np.array[int]]], each element
                    is a list of the examples for a class.
                    First dimension number of classes.
                    Second dimension number of examples.
                    Third dimension features.
                nb_classes -> int, number of classes.
                percent -> int, percent of example in a class to be
                    put in the train set, default value 80.
                features -> int or np.array[int], features selected in
                    each example, default value None.
    Output: test_set -> np.array[np.array[int]], list of examples with
                the last element is the label. Used for testing.
                First dimension number of examples.
                Second dimension features.
            train_set -> np.array[np.array[int]], list of examples with
                the last element is the label. Used for training.
                First dimension number of examples.
                Second dimension features.
    """

    nb_features = len(classes[0][0]) - 1

    # Check for inconsistencies
    try: 
        assert(isinstance(percent, int))
        assert(percent > 0 and percent < 100)

    except AssertionError:
        print("Error in function create_sets: percentage is inconsistent")
        exit(1)
        
    if type(features) is np.ndarray: 
        if features.size == 0:
            # If no features were specified use all of them (remove label)
            features = [i for i in range(nb_features)]
        
        else:
            # Check that all the features are correct
            for feature in features:
                if feature > nb_features:
                    print("Error in function create_sets: selected features \
                            exceeds array")
                    exit(1) 

    elif type(features) is int:
        if features > nb_features:
            print("Error in function create_sets: selected features exceeds \
                    array")
            exit(1)

        if features < 0:
            print("Error in function create_sets: the features must be a \
                    positive integer")
            exit(1)

        features = [features] # Convert int feature into a list
    
    else:
        print("Error in function create_sets: the features argument must be \
                an integer or a list of integers")
        exit(1)

    # Create sets and extract desired features
    train_set, train_classes_indices = generate_indices(classes, percent, 
            len(features)) 
    test_set, test_classes_indices = generate_indices(classes, 100-percent, 
            len(features))
    reduced_classes = extract_features(classes, features, nb_classes)
    train_offset= 0
    test_offset = 0

    # Fill the created sets
    for i in range(len(classes)):
        train_set_indices = range(train_offset, train_offset +
                len(train_classes_indices[i]))
        test_set_indices = range(test_offset, test_offset +
                len(test_classes_indices[i]))

        # Fill the dedicated space with examples
        train_set[train_set_indices] = reduced_classes[i][
                train_classes_indices[i]]
        test_set[test_set_indices] = reduced_classes[i][
                test_classes_indices[i]]

        # Update the offsets
        train_offset = len(train_classes_indices[i])
        test_offset = len(test_classes_indices[i])

    # Shuffle sets
    rng = default_rng() # Set seed
    rng.shuffle(train_set)
    rng.shuffle(test_set)

    return train_set, test_set

