"""
This module allows you to create graphics.
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns


def normalization(data):
    """
    Data normalization.
    """
    somme = sum(data)
    normalized_data = [ele / somme for ele in data]

    return normalized_data


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
        normalized_spectrum = normalization(spectrum)  # normalization of each sfs
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


def plot_optimisation_grid(ll_list, theta_list, log_scale, theoritical_theta):
    """
    Plot for a given scenario the likelihood and optimal theta's value for various grid size.

    Parameter
    ---------
    ll_list: list
        the likelihood of the model for each grid point
    theta_list: list
        the optimal value of theta of the model for each grid point
    """
    theta = [theoritical_theta] * len(ll_list)

    # Plot
    # plt.plot(normalization(ll_list), color="orange", label="Likelihood")
    plt.plot(theta_list, color="blue", label="Theta")
    plt.plot(theta, color="red", label="Theoritical theta", linestyle="dashed")

    # Caption
    plt.legend(loc="upper right", fontsize="large")

    # Label axis
    plt.xlabel("Smallest grid size factor", fontsize="large")

    # X axis values
    x_ax, x_values = [], []
    for i, value in enumerate(log_scale):
        x_ax.append(i)
        x_values.append(value)
    plt.xticks(x_ax, x_values)

    # Title + save plot to the folder ./Figures
    plt.title("Likelihood & theta's value for various grid point size", fontsize="xx-large")
    plt.savefig("./Figures/optimisation_grid")
    plt.clf()


def plot_error_rate(data):
    """

    Plot the error rate of theta estimated for 100 inference with dadi.
    """
    for sample_size in data.keys():
        # Plot
        sns.set_theme(style="whitegrid")
        ax = sns.boxplot(x="mu", y="Error rate", hue="Execution time", data=data[sample_size])

        # Set yaxis range
        ax.set(ylim=(0.85, 1.15))

        # Legend out of the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize='small',
                  borderaxespad=0., title="Average run time")

        # Title + save plot to folder ./Figures
        plt.title("Error rate for n={} genomes sampled".format(sample_size), fontsize="large")
        plt.savefig("./Figures/Error_rate/error_rate-{}".format(sample_size),
                    bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    sys.exit()  # No actions desired
