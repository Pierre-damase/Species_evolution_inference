"""
This module allows you to read or write files.
"""

import os
import sys
import numpy as np
import pandas as pd

######################################################################
# SFS - Dadi                                                         #
######################################################################

def dadi_data(sfs, fichier, path="./Data/", name="SFS"):
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
            "year_per_generation: {} # assumed generation time (in year)\n".format(year))

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

def export_json_files(model, filein, path_data):
    """
    Export each json file generated with msprime into a single DataFrame.

    Then export this DataFrame to a json file.

    Parameters
    ----------
    model:
        kind of model - decline, growth, migration, etc.
    path_data:
        path of each json file
    """
    if filein not in os.listdir(path_data):
        # Pandas DataFrame
        simulation = pd.DataFrame(columns=['Parameters', 'SNPs', 'SFS observed', 'Time'])

        for fichier in os.listdir(path_data):
            # Export the json file to pandas DataFrame and store it in simulation
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            simulation = simulation.append(res, ignore_index=True)

            # Delete the json file
            os.remove("{}{}".format(path_data, fichier))

        # Export pandas DataFrame simulation to json file
        simulation.to_json("{}SFS_{}-all.json".format(path_data, model))

    else:
        simulation = \
            pd.read_json(path_or_buf="{}{}".format(path_data, filein), typ='frame')

    return simulation


######################################################################
# A SUPPRIMER - probablement                                         #
######################################################################

def export_to_dataframe(path_data):
    """
    Export json file to a Pandas DataFrame.
    """
    col = [
        "Parameters", "Positive hit", "SNPs", "SFS obs", "SFS M0", "SFS M1",
        "Model0 ll", "Model1 ll", "Time simulation", "Time inference"
    ]
    data = pd.DataFrame(columns=col)

    for fichier in [ele for ele in os.listdir(path_data)]:
        res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
        data = data.append(res, ignore_index=True)

    return data


def factor_to_file(data):
    """
    Compute the length factor of each (tau, kappa) pairs nad save it to a file.
    """
    theoritical_theta = 32000

    length_factor = [np.mean(ele) / theoritical_theta for ele in data['SNPs']]
    with open("./Data/length_factor-decline", "w") as filout:
        for ele in sorted(length_factor, reverse=True):
            filout.write("{} ".format(ele))


if __name__ == "__main__":
    sys.exit()  # No actions desired
