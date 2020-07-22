#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle as pkl


if __name__ == "__main__":
    """ Fuses results into a single file.

    Input:  data_name -> name of the data used, mnist or cifar.
            archi_type -> architecture used during training.
            run_nb -> number of the run.
    Output: file -> file in which the results are fused.

    """
    dict_list = []
    conf_mat_dict = {}
    strat_names = ["Full", "MT", "CL", "SPL"]

    if len(sys.argv) != 4:
        print("usage: fusion.py data_name archi_type run_nb")
        exit(1)

    archi_type = int(sys.argv[2])
    run_nb = int(sys.argv[3])
    data_name = sys.argv[1]

    for strat in strat_names:
        filename = "res/{}_{}_{}_{}.pkl".format(strat, archi_type,
                run_nb, data_name)

        # Extract results from file
        with open(filename, "rb") as f:
            if dict_list == []:
                dict_list = pkl.load(f)

            else:
                tmp_dict_list = pkl.load(f)

                for i in range(len(dict_list)):
                    dict_list[i].update(tmp_dict_list[i])

            conf_mat_dict.update(pkl.load(f))

        os.remove(filename)

    # Save dicts into a new file
    with open("res/{}_{}_{}.pkl".format(data_name, archi_type, run_nb), 
            "wb") as f:
        pkl.dump(dict_list, f)
        pkl.dump(conf_mat_dict, f)
