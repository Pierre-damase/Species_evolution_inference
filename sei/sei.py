"""
Programme pour inférer l'évolution d'une population à partir de données génomiques.
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import chi2

import sei.arguments.arguments as arg
import sei.files.files as f
import sei.graphics.plot as plot
import sei.inference.dadi as dadi
import sei.simulation.msprime as ms


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
# SFS verification                                                   #
######################################################################
def sfs_verification():
    """
    Method to check the SFS obtained with msprime.

    I.E. check that:
     - The SFS of a constant population fits well to the theoretical SFS of any constant population
     - The SFS of an increasing or decreasing population
    """
    parametres = {
        "sample_size": 6, "size_population": 1, "rcb_rate": 2e-3, "mu": 2e-3, "length": 1e5
    }

    # Constant
    print("Scénario constant")
    sfs_cst = \
        ms.msprime_simulation(model=ms.constant_model, param=parametres, debug=True)

    # Declin
    print("\n\nScénario de déclin")
    sfs_declin = \
        ms.msprime_simulation(model=ms.sudden_decline_model, param=parametres, tau=1.5, kappa=4,
                              debug=True)

    # Growth
    print("\n\nScénario de croissance")
    sfs_croissance = \
        ms.msprime_simulation(model=ms.sudden_growth_model, param=parametres, tau=1.5, kappa=4,
                              debug=True)

    # Theoretical SFS for any constant population
    sfs_theorique = [0] * (parametres["sample_size"] - 1)
    for i in range(len(sfs_theorique)):
        sfs_theorique[i] = 1 / (i+1)

    # Plot
    plot.plot_sfs(
        sfs=[sfs_cst, sfs_theorique, sfs_declin, sfs_croissance],
        label=["Constant", "Theoretical", "Declin", "Growth"],
        color=["blue", "orange","red", "green"],
        title="Unfold SFS for various scenarios"
    )


######################################################################
# Grip point optimization                                            #
######################################################################

def grid_optimisation():
    """
    Define the optimal value of the smallest size of the grid point.
    """
    mu_list, log_scale = [2e-3, 4e-3, 8e-3, 12e-3], [1, 2, 4, 6, 10, 20, 40, 60]

    dico = {}

    for mu in mu_list:
        # Parameters for the simulation
        params = simulation_parameters(sample=20, ne=1, rcb_rate=mu, mu=mu, length=1e5)
        print("Msprime simulation - sample size {} & mutation rate {}".format(20, mu))

        # Msprime simulation
        sfs = ms.msprime_simulation(model=ms.constant_model, param=params)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, dadi.constant_model.__name__)

        # Data
        theoritical_theta = computation_theoritical_theta(ne=1, mu=mu, length=1e5)
        dico[mu] = {
            "Likelihood": [], "Estimated theta": [], "Theoritical theta": [theoritical_theta]*9
        }

        # Optimisation grid point for the extrapolation
        for _, scale in enumerate(log_scale):
            point = (len(sfs) - 1) * scale
            pts_list = [point, point+10, point+20]

            likelihood, estimated_theta = dadi.dadi_inference(pts_list, dadi.constant_model)

            dico[mu]["Likelihood"].append(likelihood)
            dico[mu]["Estimated theta"].append(estimated_theta)

    plot.plot_optimisation_grid(dico, log_scale)


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
    mu_list = [2e-3, 4e-3, 8e-3, 12e-3, 2e-2]  # ,  8e-2,  2e-1]
    nb_simu = 3

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Set up the Pandas DataFrame
    col = ["Theoritical theta", "Error rate", "mu"]
    data = pd.DataFrame(columns=col)

    # List of execution time of each simulation
    execution_time = []

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
            f.dadi_data(sfs, dadi.constant_model.__name__, name="SFS-{}".format(sample))

            # Dadi inference
            _, estimated_theta = dadi.dadi_inference(pts_list, dadi.constant_model,
                                                     name="SFS-{}".format(sample))

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

    # Export data to csv file
    data.to_csv("./Data/Error_rate/error-rate-{}.csv".format(sample), sep='\t', index=False)


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

    # return max(likelihood) / control_ll
    return 2 * (max(likelihood) - control_ll)


def likelihood_ratio_test(tau, kappa, msprime_model, dadi_model, control_model, optimization,
                          nb_simu, dof):
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
    mu, sample = 8e-3, 20  # 8e-2
    ll_list, ll_ratio = {"Model0": [], "Model1": []}, []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Parameters for the simulation
    params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)

    # Path & name
    path_data = "./Data/"
    name = "SFS-tau={}_kappa={}".format(tau, kappa)

    # Generate x genomic data for the same kappa and tau
    for _ in range(nb_simu):
        # Simulation with msprime
        sfs = ms.msprime_simulation(model=msprime_model, param=params, kappa=kappa, tau=tau)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, dadi_model.__name__, path=path_data, name=name)

        # Dadi inference - run x inference with dadi from the same observed data
        tmp = []
        control_ll, _ = dadi.dadi_inference(pts_list, control_model, path=path_data, name=name)

        for _ in range(nb_simu):
            ll, _ = dadi.dadi_inference(pts_list, dadi_model, opt=optimization, path=path_data,
                                        name=name)
            tmp.append(ll)

        # Select the best one, i.e. the one with the highest likelihood
        # Compute the log-likelihood ratio
        ll_ratio.append(log_likelihood_ratio(tmp, control_ll))

        # Keep track of log-likelihood for each simulation
        ll_list["Model0"].append(control_ll)
        ll_list["Model1"].append(max(tmp))
        #print("\nObserved {} & Estimated {}\n".format(control_ll, max(tmp)))

    # Delete sfs file
    os.remove("{}{}.fs".format(path_data, name))

    # Likelihood-ratio test
    lrt = [1] * len(ll_ratio)
    for i, chi_stat in enumerate(ll_ratio):
        p_value = chi2.sf(chi_stat, dof)
        if p_value > 0.05:
            lrt[i] = 0

    return lrt, ll_list


def inference(msprime_model, dadi_model, control_model, optimization, scale):
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
    data = pd.DataFrame(columns=col)

    if optimization == "tau":
        print("Optimization of {} for {}".format(optimization, dadi_model.__name__))

        kappa, tau = 10, np.float_power(10, scale[0])  # Kappa fixed
        print("Simulation: kappa {} & tau {}".format(kappa, tau))

        lrt, ll_list = likelihood_ratio_test(
            tau, kappa, msprime_model, dadi_model, control_model, optimization,
            nb_simu=3, dof=1
        )  # 1000 simulations
        row = {
            "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
            "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
        }
        data = data.append(row, ignore_index=True)

    elif optimization == "kappa":
        print("Optimization of {} for {}".format(optimization, dadi_model.__name__))

        kappa, tau = np.float_power(10, scale[0]), 1.0  # Tau fixed
        print("Simulation: kappa {} & tau {}".format(kappa, tau), end="\r")

        lrt, ll_list = likelihood_ratio_test(
            tau, kappa, msprime_model, dadi_model, control_model, optimization,
            nb_simu=3, dof=1
        )  # 1000 simulations
        row = {
            "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
            "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
        }
        data = data.append(row, ignore_index=True)

    else:
        print("Optimization of {} for {}".format(optimization, dadi_model.__name__))
        # for t_scale in np.arange(-3, 1.1, 0.1):
        #     for k_scale in np.arange(-2, 1.6, 0.1):
        #         pass
        kappa, tau = np.float_power(10, scale[1]), np.float_power(10, scale[0])
        print("Simulation: kappa {} & tau {}".format(kappa, tau), end="\r")

        lrt, ll_list = likelihood_ratio_test(
            tau, kappa, msprime_model, dadi_model, control_model, optimization,
            nb_simu=3, dof=2
        )
        row = {
            "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
            "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
        }
        data = data.append(row, ignore_index=True)

    #plot.plot_lrt(data)
    # Export data to csv file
    data.to_csv("./Data/Optimization_{}/opt-tau={}_kappa={}.csv"
                .format(optimization, tau, kappa), sep='\t', index=False)


######################################################################
# Main                                                               #
######################################################################

def main():
    """
    Le main du programme.
    """
    # sfs_verification()
    # grid_optimisation()

    args = arg.arguments()

    if args.analyse == 'opt':
        dadi_params_optimisation(args.number)
    elif args.analyse == 'lrt':
        inference(
            msprime_model=ms.sudden_decline_model, dadi_model=dadi.sudden_decline_model,
            control_model=dadi.constant_model, optimization=args.param, scale=args.value
        )
    elif args.analyse == 'er':
        for sample in [10, 20, 40, 60, 100]:
            plot.plot_error_rate(sample)
            plot.plot_sfs_from_dadi(sample, name="SFS-{}".format(sample))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
