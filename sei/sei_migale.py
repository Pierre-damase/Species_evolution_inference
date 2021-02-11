"""
Migale version.
"""

import time
import pandas as pd
import numpy as np
from collections import Counter

from files import files as f
from graphics import plot
from inference import dadi
from simulation import msprime as ms


def computation_theoritical_theta(ne, mu, length):
    """
    Compute the theoritical theta - theta = 4.Ne.mu.L with

      - 4: diploid population
      - Ne: the effective population size
      - mu: the mutation rate
      - L: the size of the simulated genomes
    """
    return (4 * ne * mu * length)


def simulation_parameters(sample, ne, rcb_rate, mu, length):
    """
    Set up the parametres for the simulation
    """
    params = {
        "sample_size": sample, "size_population": ne, "rcb_rate": rcb_rate, "mu": mu,
        "length": length
    }
    return params


def likelihood_ratio(likelihood, control_ll):
    """
    Likelihood-ratio between two model.

    Parameter
    ---------
    likelihood: list
        Likelihood for x inference with dadi
    """
    return max(likelihood) / control_ll


######################################################################
# Optimization of dadi parameters                                    #
######################################################################

def dadi_params_optimisation():
    """
    Determine the error rate of the inference of 100 observed - simulated with msprime.

    Each observed is a constant population model. The goal is to determine the best mutation
    rate mu and the best number of sampled genomes n.

      - mu: the mutation rate
      - n: the number of sampled monoploid genomes
    """
    sample_list, mu_list = [10, 20, 40, 60, 100], [2e-3, 4e-3]  #, 8e-3, 12e-3, 2e-2,  8e-2,  2e-1]
    nb_simu = 25
    dico = {}
    path_data = "/home/pimbert/work/Species_evolution_inference/Data/"
    path_figures = "/home/pimbert/work/Species_evolution_inference/Figures/Error_rate/"

    for sample in sample_list:
        # Grid point for the extrapolation
        pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

        # Set up the Pandas DataFrame
        col = ["Theoritical theta", "Error rate", "mu"]
        dico[sample] = pd.DataFrame(columns = col)

        # List of execution time of each simulation
        execution_time = []

        for mu in mu_list:
            tmp = []

            # Parameters for the simulation
            params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)
            print("Msprime simulation - sample size {} & mutation rate {}".format(sample, mu))

            for i in range(nb_simu):
                start_time = time.time()
                print("Simulation: {}/{}".format(i, nb_simu), end="\r")

                # Simulation for a constant population with msprime
                sfs_cst = \
                    ms.msprime_simulation(model=ms.constant_model, param=params, tau=0.0,
                                          kappa=0.0, debug=False)

                # Generate the SFS file compatible with dadi
                f.dadi_data(sfs_cst, dadi.constant_model.__name__, path=path_data)

                # Dadi inference
                _, estimated_theta = dadi.dadi_inference(pts_list, dadi.constant_model,
                                                         path=path_data)

                theoritical_theta = computation_theoritical_theta(ne=1, mu=mu, length=1e5)
                error_rate = estimated_theta / theoritical_theta

                row = {
                    "Theoritical theta": theoritical_theta, "Error rate": error_rate, "mu": mu
                }
                dico[sample] = dico[sample].append(row, ignore_index=True)

                tmp.append(time.time() - start_time)

            # Mean execution time for the 100 simulation with the same genome length and mu
            mean_time = round(sum(tmp) / nb_simu, 3)
            execution_time.extend([mean_time for _ in range(nb_simu)])

        dico[sample]["Execution time"] = execution_time

    plot.plot_error_rate(dico, path=path_figures)


######################################################################
# Likelihood-ratio test                                              #
######################################################################

def likelihood_ratio_test(tau, kappa, msprime_model, dadi_model, control_model, optimization,
                          nb_simu):
    """
    Likelihood-ratio test to assesses the godness fit of two model.

    Parameter
    ---------
    tau: float
        the lenght of time ago at which the event (decline, growth) occured
    kappa: float
        the growth or decline force

    Return
    ------
    data: list
        List of 0 and 1.
          - 0: negative hit - ratio > 0.05 
          - 1: positive hit - ratio <= 0.05
    """
    mu, sample = 2e-2, 20
    ll_ratio = []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Parameters for the simulation
    params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)

    # Generate x genomic data for the same kappa and tau
    for _ in range(nb_simu):
        # Simulation with msprime
        sfs = ms.msprime_simulation(model=msprime_model, param=params, kappa=kappa, tau=tau)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, dadi_model.__name__)

        # Dadi inference - run x inference with dadi from the same observed data
        likelihood_list = []
        control_ll, _ = dadi.dadi_inference(pts_list, control_model)

        for _ in range(nb_simu):
            likelihood, _ = dadi.dadi_inference(pts_list, dadi_model, opt=optimization,
                                                verbose=0)
            likelihood_list.append(likelihood)

        # Select the best one, i.e. the one with the highest likelihood
        ll_ratio.append(likelihood_ratio(likelihood_list, control_ll))

    # Likelihood-ratio test
    data = [1] * len(ll_ratio)
    for i, ratio in enumerate(ll_ratio):
        if ratio > 0.05:
            data[i] = 0

    return data


def inference(msprime_model, dadi_model, control_model, optimization):
    """

    Parameter
    ---------
    msprime_model: function
        the custom model to generate genomic data
    dadi_model: function
        the custom model to infer demography history
    control_model: function
        control model
    optimization
        parameter to optimize - (kappa), (tau), (kappa, tau), etc.
    """
    col = ["Tau", "Kappa", "Positive hit"]
    data = pd.DataFrame(columns = col)

    if optimization == "tau":
        for tau in np.arange(0.01, 10.0, 1):  # 0.1
            kappa = 10  # Kappa fixe
            tmp = likelihood_ratio_test(tau, kappa, msprime_model, dadi_model, control_model,
                                        optimization, nb_simu=10)  # 1000
            row = {
                "Tau": tau, "Kappa": kappa, "Positive hit": Counter(tmp)[1]
            }
            data = data.append(row, ignore_index=True)

    elif optimization == "kappa":
        for kappa in np.arange(1.0, 30.0, 0.5):
            tau = 1.0
            tmp = likelihood_ratio_test(tau, kappa, msprime_model, dadi_model, control_model,
                                        optimization, nb_simu=3)
            row = {
                "Tau": tau, "Kappa": kappa, "Positive hit": Counter(tmp)[1]
            }
            data = data.append(row, ignore_index=True)

    else:
        for tau in np.arange(0.01, 10.0, 0.1):
            for kappa in np.arange(1.0, 30.0, 0.5):
                tmp = likelihood_ratio_test(tau, kappa, msprime_model, dadi_model,
                                            control_model, optimization, nb_simu=3)
                row = {
                    "Tau": tau, "Kappa": kappa, "Positive hit": Counter(tmp)[1]
                }
                data = data.append(row, ignore_index=True)

    plot.plot_lrt(
        data, path="/home/pimbert/work/Species_evolution_inference/Figures/Error_rate/")


######################################################################
# Main                                                               #
######################################################################

if __name__ == "__main__":
    # dadi_params_optimisation()
    # inference(msprime_model=ms.sudden_decline_model, dadi_model=dadi.sudden_decline_model,
    #           control_model=dadi.constant_model, optimization="tau")
