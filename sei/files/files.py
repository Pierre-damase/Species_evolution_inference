"""
This module allows you to read or write files.
"""

import ast
import copy
import csv
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
# SFS - species (real data)                                          #
######################################################################

def load_species_sfs(species):
    """
    Load the SFS of a given species (real data).

    Parameter
    ---------
    species: str
        A given species

    Return
    ------
    SFS: list
        The SFS for a gicen species
    """
    sfs = []
    with open("./Data/Real_data/SFS/{}".format(species.replace(' ', '_')), 'r') as filin:
        lines = filin.readlines()
        for line in lines:
            sfs.append(int(line.strip().split('\t')[1]))
    return sfs


def load_species_data():
    """
    Load for each species (real data) the SFS.

    Return
    ------
    data: dictionary
        Dictionary of species (key) and SFS (value)
    """
    data = {}
    with open("./Data/Real_data/Species.csv", "r") as filin:
        reader = csv.DictReader(filin)
        for row in reader:
            data[row['Species']] = {
                'SFS': load_species_sfs(row['Species']), 'Status': row['Conservation status']
            }
    return data


######################################################################
# SFS - Dadi                                                         #
######################################################################

def dadi_data(sfs_observed, fichier, fold, path="./Data/", name="SFS"):
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
    sfs_observed: list
        the original SFS - without corners 0/n & n/n
    fichier: str
        file in which the SFS will be written in the format compatible with dadi
    fold: bool
        if the SFS must be fold (True) or not (False)
    """
    # Pre-processing of the SFS
    # sfs = copy.deepcopy(sfs_observed)
    sfs = [0] + sfs_observed + [0]  # Add 0/n & n/n to the sfs (lower and upper bound)
    if fold:
        sfs = sfs[:round(len(sfs)/2) + 1] + [0] * int(np.floor(len(sfs)/2))  # SFS folded

    with open("{}{}.fs".format(path, name), "w") as filout:
        if fold:
            filout.write("{} folded \"{}\"\n".format(len(sfs), fichier))
        else:
            filout.write("{} unfolded \"{}\"\n".format(len(sfs), fichier))


        # Write the SFS
        for freq in sfs:
            filout.write("{} ".format(freq))

        # Write 1 to mask value & 0 to unmask value
        filout.write("\n")
        for freq in sfs:
            filout.write("1 ") if freq == 0 else filout.write("0 ")

    del sfs


######################################################################
# SFS - Stairway plot 2                                              #
######################################################################

def stairway_data(name, data, path):
    """
    Create SFS of a scenario in the format compatible with the stairway plot v2 software.

    (Xiaoming Liu & Yun-Xin Fu 2020, stairway-plot-v2, see readme file for details)
    """
    sfs, nseq, mu, year, ninput = \
        data['sfs'], data['sample_size'], data['mu'], data['year'], data['ninput']

    with open("{}{}.blueprint".format(path, name), "w") as filout:
        filout.write("# Blueprint {} file\n".format(name))
        filout.write("popid: {} # id of the population (no white space)\n".format(name))
        filout.write("nseq: {} # number of sequences\n".format(nseq))
        filout.write(
            "L: {} # total number of observed nucleic sites, including poly-/mono-morphic\n"
            .format(sum(sfs)*10))
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


def read_stairway_final(path):
    """
    Read all file from the final folder, folder generated by stairway at the end of the
    inference.

    Parameter
    ---------
    path: path of the final folder

    Return
    ------
    data: dictionary
      - M0: default model, i.e. model of 1 dimension
        with LL: the log-likelihood of the testing data - stairway return a minus log-likelihood
             Theta: theta of of M0
      - M1: the final model with x dimension
        with LL: idem
             Theta min: minimum theta
             Theta max: maximum theta
    """
    data = {'M0': {'LL': [], 'Theta': []}, 'M1': {'LL': [], 'Theta min': [], 'Theta max': []}}

    for fichier in os.listdir(path):
        with open("{}{}".format(path, fichier), 'r') as filin:
            lines = filin.readlines()

            for line in lines[:-2]:
                # Save M0
                if line.startswith('dim:\t1'):
                    _, _, _, ll, _, _, _, theta = line.strip().split('\t')
                    data['M0']['LL'].append(-float(ll))
                    data['M0']['Theta'].append(float(theta))

                # Save M1
                if line.startswith('final'):
                    _, ll, _ = line.strip().split('\t')
                    data['M1']['LL'].append(-float(ll))

            line = [float(ele) for ele in lines[-1].strip().split(' ')]
            data['M1']['Theta min'].append(min(line))
            data['M1']['Theta max'].append(max(line))

    return data


def min_max(data):
    """
    Compute the min & max (value and index) of a list

    Return
    ------
    Pair Ne min (value, [index]) & Ne max (value, [index])
    """
    minimum, maximum = min(data), max(data)
    index_min = [i for i, ele in enumerate(data) if ele == minimum]
    index_max = [i for i, ele in enumerate(data) if ele == maximum]
    return (minimum, index_min), (maximum, index_max)


def read_stairway_summary(fichier):
    """
    Read the final output summary of the inference with stairway.

    Parameter
    ---------
    fichier: file to read

    Return
    ------
    data: dictionary
     - Ne: pair (Ne min, Ne max)
     - Year: pair (Year of Ne min, Year of Ne max)
    """
    with open(fichier, "r") as filin:
        lines = filin.readlines()
        ne_list, year_list = [], []

        for line in lines[1:]:
            _, _, _, _, _, year, ne, _, _, _, _ = line.strip().split('\t')
            ne_list.append(float(ne))
            year_list.append(float(year))

        # Pair Ne min (value, [index]) & Ne max (value, [index])
        minimum, maximum = min_max(ne_list)

    data = {
        'Ne': (minimum[0], maximum[0]), 'Year': (np.mean([year_list[i] for i in minimum[1]]),
                                                 np.mean([year_list[i] for i in maximum[1]]))
    }

    return data


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


def export_inference_files(model, fold, param, value=None):
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
    col = ['Parameters', 'Positive hit', 'SNPs', 'SFS observed', 'M0', 'M1', 'Time',
           'd2 observed inferred', 'd2 models']
    inference = pd.DataFrame(columns=col)
    inference['Positive hit'] = inference['Positive hit'].astype(int)

    # Path data and filin
    path_data = "./Data/Dadi/{}/{}/".format(model, param)
    path_data += "Folded/" if fold else "Unfolded/"
    if param == 'all':
        filin = "dadi_{}-all".format(model)
        if '{}.zip'.format(filin) in os.listdir(path_data):
            os.system('unzip {0}{1}.zip -d {0}'.format(path_data, filin))

    else:
        filin = "dadi_{}={}_all".format(model, value)

    # Read file
    if filin not in os.listdir(path_data):

        fichiers = os.listdir(path_data)

        # Select estimation for the specific value of param that is either tau, kappa or m12
        if param != 'all':
            fichiers = [
                fichier for fichier in fichiers
                if not fichier.endswith('all')
                and float(fichier.rsplit('-', maxsplit=1)[0].split('=')[1]) == value
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

    # File > 100 Mb
    if os.stat("{}{}".format(path_data, filin)).st_size > 1e8:
        # Zip
        if "{}.zip".format(filin.split('.')[0]) not in os.listdir(path_data):
            os.system('zip -j {0}{1}.zip {0}{2}'.format(path_data, filin.split('.')[0], filin))

        # Remove
        os.remove("{}{}".format(path_data, filin))

    return inference


def export_specific_dadi_inference(model, fixed_param, values, fold):
    """
    Export specific dadi inference file for a given fixed parameter.

    A detailed application of this method can be found in the notebook analyse.ipynb.

    Parameter
    ---------
    model: str
        either decline or migration
    fixed_param: str
        fixed parameter to consider - either kappa, tau or m12
    values: list of float
        fixed parameter value to consider - log scale
        - Tau/m12: between -4 included and 2.5 excluded
        - Kappa: between -3.5 included and 3 excluded
    fold: bool
        True : inference is done with folded SFS
        False: inference is done with unfolded SFS

    Return
    ------
    data: list of pandas DataFrame of dadi inference
    labels: list of label for each DataFrame
    """
    data, labels = [], []

    for val in values:
        data.append(export_inference_files(model, fold, fixed_param, val))
        labels.append("{} = {:.1e}".format(
            fixed_param if fixed_param == "m12" else fixed_param.capitalize(),
            np.power(10, val))
        )

    return data, labels


if __name__ == "__main__":
    sys.exit()  # No actions desired
