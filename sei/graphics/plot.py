"""
This module allows you to create graphics.
"""

import matplotlib.pyplot as plt
import sys


def plot_sfs(sfs, label, color, title):
    """
    Graphic representation of Site Frequency Spectrum (SFS), save to the folder ./Figures.

    Parameter
    ---------
    sfs: list
        list of sfs to plot
    label: list
        the label of each sfs
    color: list
        the color of each curve
    title: string
        title of the plot
    """
    # Plot
    for i, spectrum in enumerate(sfs):
        somme = sum(spectrum)
        normalized_spectrum = [ele / somme for ele in spectrum]  # normalization of each sfs

        plt.plot(normalized_spectrum, color=color[i], label=label[i])

    # Caption
    plt.legend(loc="upper right", fontsize="large")

    # Label axis
    plt.xlabel("Allele frequency", fontsize="large")
    plt.ylabel("Percent of SNPs", fontsize="large")

    # X axis values
    x_ax, x_values = [], []
    for i in range(len(sfs[0])):
        x_ax.append(i)
        x_values.append("{}/{}".format(i+1, len(sfs[0])+1))
    plt.xticks(x_ax, x_values)

    # Title + save plot to the folder ./Figures
    plt.title(title, fontsize="xx-large")
    plt.savefig("./Figures/sfs")
    plt.clf()


if __name__ == "__main__":
    sys.exit()  # No actions desired
