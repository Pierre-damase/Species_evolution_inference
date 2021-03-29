"""
This module allows you to read or write files.
"""

import ast
import copy
import json
import os
import sys
import numpy as np
import pandas as pd


######################################################################
# SFS shape verification                                             #
######################################################################

def load_sfs(function, generate=False):
    """
    If generate True, create a new set of SFS for various scenario.

    If generate False, load a set of SFS for various scenario.
    """
    if generate:
        all_sfs, params, params_simulation = function()
        with open("./Data/Msprime/sfs_shape_verification", 'w') as filout:
            filout.write("SFS shape verification and simulations parameters - {}\n"
                         .format(params_simulation))

            for model, sfs in all_sfs.items():
                if model in params.keys():
                    filout.write("{} - {} - {}\n".format(model, sfs, params[model]))
                else:
                    filout.write("{} - {}\n".format(model, sfs))

    else:
        all_sfs, params = {}, {}
        with open("./Data/Msprime/sfs_shape_verification", 'r') as filin:
            lines = filin.readlines()
            params_simulation = ast.literal_eval(lines[0].strip().split(' - ')[1])

            for line in lines[1:]:
                tmp = line.strip().split(' - ')
                if tmp[0] not in ['Constant model', 'Theoretical model']:
                    params[tmp[0]] = ast.literal_eval(tmp[2])
                all_sfs[tmp[0]] = json.loads(tmp[1])

    return all_sfs, params, params_simulation


######################################################################
# SFS - Dadi                                                         #
######################################################################

def dadi_data(sfs_observed, fichier, path="./Data/", name="SFS"):
    """
    Create SFS of a scenario in the format compatible with the dadi software.

    (Gutenkunst et al. 2009, see their manual for details)

    A pre-processing of the SFS is needed
      1. Which one
         Adding 0/n & n/n.
      2. Why
         Spectrum arrays are masked, i.e. certain entries can be set to be ignored
         For example, the two corners [0,0] corresponding to variants observed in zero samples
         or in all samples are ignored

    Parameter
    ---------
    sfs: list
        the original SFS - without corners 0/n & n/n
    fichier: str
        file in which the SFS will be written in the format compatible with dadi
    """
    sfs = copy.deepcopy(sfs_observed)

    with open("{}{}.fs".format(path, name), "w") as filout:
        filout.write("{} unfolded \"{}\"\n".format(len(sfs)+2, fichier))

        # Write SFS
        sfs.insert(0, 0)         # Add 0/n to the sfs
        sfs.insert(len(sfs), 0)  # Add n/n to the sfs
        for freq in sfs:
            filout.write("{} ".format(freq))

        filout.write("\n")
        for freq in sfs:
            if freq == 0:
                filout.write("1 ")
            else:
                filout.write("0 ")

    del sfs


######################################################################
# SFS - Stairway plot 2                                              #
######################################################################

def stairway_data(name, data, path):
    """
    Create SFS of a scenario in the format compatible with the stairway plot v2 software.

    (Xiaoming Liu & Yun-Xin Fu 2020, stairway-plot-v2, see readme file for details)
    """
    sfs, nseq, length, mu, year, ninput = \
        data['sfs'], data['sample_size'], data['length'], data['mu'], data['year'], \
        data['ninput']

    with open("{}{}.blueprint".format(path, name), "w") as filout:
        filout.write("# Blueprint {} file\n".format(name))
        filout.write("popid: {} # id of the population (no white space)\n".format(name))
        filout.write("nseq: {} # number of sequences\n".format(nseq))
        filout.write(
            "L: {} # total number of observed nucleic sites, including poly-/mono-morphic\n"
            .format(int(length)))
        filout.write("whether_folded: false # unfolded SFS\n")

        # SFS
        filout.write("SFS: ")
        for snp in sfs:
            filout.write("{} ".format(snp))
        filout.write("# snp frequency spectrum: number of singleton, doubleton, etc.\n")

        filout.write("#smallest_size_of_SFS_bin_used_for_estimation: 1\n")
        filout.write("#largest_size_of_SFS_bin_used_for_estimation: {}\n".format(nseq-1))
        filout.write("pct_training: 0.67 # percentage of sites for training\n")

        # Break points
        break_points = np.ceil([(nseq-2)/4, (nseq-2)/2, (nseq-2)*3/4, (nseq-2)])
        filout.write("nrand: ")
        for ele in break_points:
            filout.write("{} ".format(int(ele)))
        filout.write("# number of random break points for each try\n")

        filout.write("project_dir: {}{} # project directory\n".format(path, name))
        filout.write("stairway_plot_dir: {}stairway_plot_es\n".format(path))
        filout.write("ninput: {} # number of input files to be created for each estimation\n"
                     .format(ninput))
        filout.write("# random_seed: 6  if commented, the program will randomly pick one\n")

        # Output
        filout.write("# Output settings\n")
        filout.write("mu: {} # assumed mutation rate per site per generation\n".format(mu))
        filout.write(
            "year_per_generation: {} # assumed generation time (in year)\n".format(1))

        # Plot
        filout.write("# Plot settings\n")
        filout.write("plot_title: {} # title of the plot\n".format(name))
        filout.write(
            "xrange: 0,0 # Time (1K year) range; format: xmin, xmax; 0,0 for default\n")
        filout.write(
            "yrange: 0,0 # Ne (1k individual) range; format: ymin, ymax; 0,0 for default\n")
        filout.write("xspacing: 2 # x axis spacing\n")
        filout.write("yspacing: 2 # y axis sapcing\n")
        filout.write("fontsize: 12\n")


######################################################################
# Export json files                                                  #
######################################################################

def export_simulation_files(filin, path_data):
    """
    Export each json file generated with msprime into a single DataFrame.
    Then export this DataFrame to a json file.

    Parameters
    ----------
    path_data:
        path of each json file
    """
    if filin not in os.listdir(path_data):
        # Pandas DataFrame
        simulation = pd.DataFrame(columns=['Parameters', 'SNPs', 'SFS observed', 'Time'])

        for fichier in os.listdir(path_data):
            # Export the json file to pandas DataFrame and store it in simulation
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            simulation = simulation.append(res, ignore_index=True)

            # Delete the json file
            os.remove("{}{}".format(path_data, fichier))

        # Export pandas DataFrame simulation to json file
        simulation.to_json("{}{}".format(path_data, filin))

    else:
        simulation = \
            pd.read_json(path_or_buf="{}{}".format(path_data, filin), typ='frame')

    return simulation


def export_inference_files(model, param, value=None):
    """
    Export each json file generated with dadi into a single DataFrame.
    Then export this DataFrame to a json file.

    Parameter
    ---------
    model: either cst, decline or migration
    param: either all, tau, kappa or ne
    value: not None if param is tau or kappa, it's the value of the fixed parameters for the
    inference
    """
    # Data
    col = ['Parameters', 'Positive hit', 'SNPs', 'SFS observed', 'M0', 'M1', 'Time']
    inference = pd.DataFrame(columns=col)

    # Path data and filin
    path_data = "./Data/Dadi/{}/{}/".format(model, param)
    if param == 'all':
        filin = "dadi_{}-all.json".format(model)
    else:
        filin = "dadi_{}={}-all.json".format(model, value)

    # Read file
    if filin not in os.listdir(path_data):

        fichiers = os.listdir(path_data)
        if param != 'all':
            fichiers = [
                fichier for fichier in fichiers
                if not fichier.endswith('.json')
                and round(float(fichier.split('=')[1].split('-')[0]), 2) == value
            ]

        for fichier in fichiers:
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            inference = inference.append(res, ignore_index=True)

            # Delete the json file
            os.remove("{}{}".format(path_data, fichier))

        # Export pandas DataFrame inference to json file
        inference.to_json("{}{}".format(path_data, filin))

    else:
        inference = pd.read_json(path_or_buf="{}{}".format(path_data, filin), typ='frame')

    return inference


if __name__ == "__main__":
    sys.exit()  # No actions desired
