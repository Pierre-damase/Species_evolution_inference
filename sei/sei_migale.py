"""
Migale version.
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import chi2

from arguments import arguments as arg
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


######################################################################
# Optimization of dadi parameters                                    #
######################################################################

def dadi_params_optimisation(sample):
    """
    Determine the error rate of the inference of 100 observed - simulated with msprime.

    Each observed is a constant population model. The goal is to determine the best mutation
    rate mu and the best number of sampled genomes n.

      - mu: the mutation rate
      - n: the number of sampled monoploid genomes

    Parameter
    ---------
    sample: int
        it's n
    """
    mu_list = [2e-3, 4e-3, 8e-3, 12e-3, 2e-2, 8e-2, 2e-1]
    nb_simu = 100

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Set up the Pandas DataFrame
    col = ["Theoritical theta", "Error rate", "mu"]
    data = pd.DataFrame(columns=col)

    # List of execution time of each simulation
    execution_time = []

    # Path
    path_data = "/home/pimbert/work/Species_evolution_inference/Data/"
    # path_figures = "/home/pimbert/work/Species_evolution_inference/Figures/Error_rate/"

    for mu in mu_list:
        tmp = []

        # Parameters for the simulation
        params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)
        print("Msprime simulation - sample size {} & mutation rate {}".format(sample, mu))

        for i in range(nb_simu):
            start_time = time.time()
            print("Simulation: {}/{}".format(i+1, nb_simu), end="\r")

            # Simulation for a constant population with msprime
            sfs = ms.msprime_simulation(model=ms.constant_model, param=params)

            # Generate the SFS file compatible with dadi
            f.dadi_data(
                sfs, dadi.constant_model.__name__, path=path_data, name="SFS-{}".format(sample)
            )

            # Dadi inference
            _, estimated_theta = dadi.dadi_inference(
                pts_list, dadi.constant_model, path=path_data, name="SFS-{}".format(sample)
            )

            theoritical_theta = computation_theoritical_theta(ne=1, mu=mu, length=1e5)
            error_rate = estimated_theta / theoritical_theta

            row = {
                "Theoritical theta": theoritical_theta, "Error rate": error_rate, "mu": mu
            }
            data = data.append(row, ignore_index=True)

            tmp.append(time.time() - start_time)

        # Mean execution time for the 100 simulation with the same genome length and mu
        mean_time = round(sum(tmp) / nb_simu, 3)
        execution_time.extend([mean_time for _ in range(nb_simu)])

        data["Execution time"] = execution_time

    #plot.plot_error_rate(data, sample, path=path_figures)
    data.to_csv("{}Error_rate/error-rate-{}.csv".format(path_data, sample), sep='\t', index=False)


######################################################################
# Likelihood-ratio test                                              #
######################################################################

def log_likelihood_ratio(likelihood, control_ll):
    """
    Log-likelihood ratio test.

    lrt = 2 * (LL1 - LL0) with

      - LL1: log-likelihood of M1
      - LL0: log-likelihood of M0

    Here, we are juste subtracting the null hypothesis log-likelihood from the alternative
    hypothesis log-likelihood.

    Parameter
    ---------
    likelihood: list
        Likelihood for x inference with dadi, i.e. likelihood for model M1
    control_ll: float
        Likelihood for model M0
    """
    return 2 * (max(likelihood) - control_ll)


def likelihood_ratio_test(tau, kappa, msprime_model, dadi_model, control_model, optimization,
                          save, nb_simu, dof):
    """
    Likelihood-ratio test to assesses the godness fit of two model.

    Allows you to test whether adding parameters to models significantly increases the
    likelihood of the model.

    Model:
      - M0 a n0-parameter model, the model with less parameters
      - M1 a n1-parameter model, the model with more parameters

      with n0 < n1 (number of parameters)

      The degrees of freedom is the difference in the number of parameters between M0 and M1.

    Hypothesis:
      - H0 the null hypothesis:
        Adding the parameter(s) does not significantly increase the likelihood of the model.
      - H1 the alternative hypothesis:
        Adding the parameter(s) significantly increase the likelihood of the model.

    Decision rule - alpha = 0.05
      - If p-value >= alpha then the test is insignificant and do not reject of H0
      - If p-value < alpha then the test is significant and reject of H0

    Parameter
    ---------
    tau: float
        the lenght of time ago at which the event (decline, growth) occured
    kappa: float
        the growth or decline force
    dof: int
        degrees of freedom

    Return
    ------
    data: list
        List of 0 (negative run) & 1 (positive run)
    ll_list: list
        List of log-likelihood to keep track for each simulation
          - Model0: log-likelihood for the model with less parameters
          - Model1: nbest log-likelihood for the model with more parameters
    """
    mu, sample = 8e-5, 20  # 8e-2
    ll_list, ll_ratio = {"Model0": [], "Model1": []}, []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Parameters for the simulation
    params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)

    # Path & name
    path_data = "/home/pimbert/work/Species_evolution_inference/Data/"
    name = "SFS-tau={}_kappa={}".format(tau, kappa)

    # Generate x genomic data for the same kappa and tau
    for _ in range(nb_simu):
        # Simulation with msprime
        sfs = ms.msprime_simulation(model=msprime_model, param=params, kappa=kappa, tau=tau)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, dadi_model.__name__, path=path_data, name=name)

        # Dadi inference - run x inference with dadi from the same observed data
        tmp = {"LL": [], "SFS": []}
        control_ll, *_ = dadi.dadi_inference(pts_list, control_model, path=path_data, name=name)

        for _ in range(nb_simu):
            ll, _, model = dadi.dadi_inference(
                pts_list, dadi_model, opt=optimization, path=path_data, name=name)
            tmp["LL"].append(ll)
            tmp["SFS"].append(model)

        # Select the best one, i.e. the one with the highest likelihood
        # Compute the log-likelihood ratio
        ll_ratio.append(log_likelihood_ratio(tmp["LL"], control_ll))

        # Keep track of log-likelihood for each simulation
        ll_list["Model0"].append(control_ll)
        ll_list["Model1"].append(max(tmp["LL"]))

    # Keep track of some SFS generated from dadi
    if save:
        index = tmp["LL"].index(max(tmp["LL"]))  # Get index of the best inference (higher ll)
        sfs = tmp["SFS"][index]
        sfs.to_file("{}Optimization_{}/{}".format(path_data, optimization, name))
    del tmp

    # Delete sfs file
    os.remove("{}{}.fs".format(path_data, name))

    # Likelihood-ratio test
    lrt = [1] * len(ll_ratio)
    for i, chi_stat in enumerate(ll_ratio):
        p_value = chi2.sf(chi_stat, dof)
        if p_value > 0.05:
            lrt[i] = 0

    return lrt, ll_list


def inference(msprime_model, dadi_model, control_model, optimization, scale, save=False):
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
    col = ["Tau", "Kappa", "Positive hit", "Model0 ll", "Model1 ll"]
    data = pd.DataFrame(columns=col)

    if optimization == "tau":
        kappa, tau = 10, np.float_power(10, scale[0])  # Kappa fixed

        lrt, ll_list = likelihood_ratio_test(
            tau, kappa, msprime_model, dadi_model, control_model, optimization, save,
            nb_simu=1000, dof=1
        )
        row = {
            "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
            "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
        }
        data = data.append(row, ignore_index=True)

    elif optimization == "kappa":
        kappa, tau = np.float_power(10, scale[0]), 1.0  # Tau fixed

        lrt, ll_list = likelihood_ratio_test(
            tau, kappa, msprime_model, dadi_model, control_model, optimization, save,
            nb_simu=1000, dof=1
        )
        row = {
            "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
            "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
        }
        data = data.append(row, ignore_index=True)

    else:
        kappa, tau = np.float_power(10, scale[1]), np.float_power(10, scale[0])

        lrt, ll_list = likelihood_ratio_test(
            tau, kappa, msprime_model, dadi_model, control_model, optimization, save,
            nb_simu=1000, dof=2
        )
        row = {
            "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
            "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
        }
        data = data.append(row, ignore_index=True)

    # Export data to csv file
    path_data = "/home/pimbert/work/Species_evolution_inference/Data/"
    data.to_csv("{}Optimization_{}/opt-tau={}_kappa={}.csv"
                .format(path_data, optimization, tau, kappa), index=False)


######################################################################
# Main                                                               #
######################################################################

if __name__ == "__main__":
    # dadi_params_optimisation()

    # inference(msprime_model=ms.sudden_decline_model, dadi_model=dadi.sudden_decline_model,
    #           control_model=dadi.constant_model, optimization="tau")

    args = arg.arguments()

    if args.analyse == 'opt':
        sample = [10, 20, 40, 60, 100]
        dadi_params_optimisation(sample[args.number-1])
    elif args.analyse == 'lrt':
        if args.param == 'tau':
            scale = [np.arange(-3, 1.1, 0.1)[int(args.value[0])-1]]
        elif args.param == 'kappa':
            scale = [np.arange(-2.5, 1.6, 0.1)[int(args.value[0])-1]]
        else:
            scale = [
                np.arange(-3, 1.1, 0.1)[int(args.value[0])-1],
                np.arange(-2.5, 1.6, 0.1)[int(args.value[1])-1]
            ]

        if np.mod(args.value, 5) == 0:
            save = True
        else:
            save = False

        inference(
            msprime_model=ms.sudden_decline_model, dadi_model=dadi.sudden_decline_model,
            control_model=dadi.constant_model, optimization=args.param, scale=scale, save=save
        )
