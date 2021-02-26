"""
This module allows you to read or write files.
"""

import sys


def dadi_data(sfs, fichier, path="./Data/", name="SFS"):
    """
    Create SFS of a scenario in the format compatible with the dadi software.

    (Gutenkunst et al. 2009, see their manual for details)

    A pre-processing of the SFS is needed
      1. Which one: adding 0/n & n/n.
      2. Why: Spectrum arrays are masked, i.e. certain entries can be set to be ignored
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


def export_sfs(path, name):
    """
    Export SFS file in the format compatible with the dadi software into list.
    """
    with open("{}{}.fs".format(path, name), "r") as filin:
        lines = filin.readlines()
    return [int(ele) for ele in lines[1].strip().split(' ') if ele != "0"]


def stairway_data(path="./Data/", name="SFS"):
    with open("./sei/inference/stairway_plot_v2.1.1/two-epoch.blueprint", "r") as filin:
        lines = filin.readlines()
        print(lines[4])
        print([float(ele.strip()) for ele in lines[6].split('#')[0].split(':')[1].split('\t')])


if __name__ == "__main__":
    sys.exit()  # No actions desired
