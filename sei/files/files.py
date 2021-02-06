"""
This module allows you to read or write files.
"""

import sys


def dadi_data(sfs, fichier, path="./Data/"):
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
    with open("{}/{}.fs".format(path, fichier), "w") as filout:
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


if __name__ == "__main__":
    sys.exit()  # No actions desired
