"""
Programme pour inférer l'évolution d'une population à partir de données génomiques.
"""

import os
import sys
import time
import warnings
from collections import Counter
import pandas as pd
import numpy as np
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
    return 4 * ne * mu * length


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

def sfs_shape_verification():
    """
    Method to check the SFS obtained with msprime.

    I.E. check that:
     - The SFS of a constant population fits well to the theoretical SFS of any constant population
     - The SFS of an increasing or decreasing population
    """
    params = simulation_parameters(sample=10, ne=1, rcb_rate=2e-2, mu=2e-2, length=1e5)

    # Constant
    print("Scénario constant")
    sfs_cst = \
        ms.msprime_simulation(model=ms.constant_model, param=params, debug=True)

    # Declin
    print("\n\nScénario de déclin")
    sfs_declin = \
        ms.msprime_simulation(model=ms.sudden_decline_model, param=params, tau=1.0, kappa=10,
                              debug=True)

    # Growth
    print("\n\nScénario de croissance")
    sfs_croissance = \
        ms.msprime_simulation(model=ms.sudden_growth_model, param=params, tau=1.0, kappa=10,
                              debug=True)

    # Theoretical SFS for any constant population
    sfs_theorique = [0] * (params["sample_size"] - 1)
    for i in range(len(sfs_theorique)):
        sfs_theorique[i] = 1 / (i+1)

    # Plot
    plot.plot_sfs(
        sfs=[sfs_cst, sfs_theorique, sfs_declin, sfs_croissance],
        label=["Constant", "Theoretical", "Declin", "Growth"],
        color=["tab:blue", "tab:orange", "tab:red", "tab:green"],
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

            likelihood, estimated_theta, _ = dadi.dadi_inference(pts_list, dadi.constant_model)

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
            f.dadi_data(sfs, dadi.constant_model.__name__, path="./Data/Error_rate/",
                        name="SFS-{}".format(sample))

            # Dadi inference
            _, estimated_theta, _ = dadi.dadi_inference(
                pts_list, dadi.constant_model, path="./Data/Error_rate/",
                name="SFS-{}".format(sample)
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

    # Delete sfs file
    os.remove("./Data/Error_rate/SFS-{}.fs".format(sample))

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
    mu, sample = 8e-6, 20  # 8e-2
    ll_list, ll_ratio, model_list = {"Model0": [], "Model1": []}, [], {"LL": [], "SFS": []}

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Parameters for the simulation
    params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)
    
    # Path & name
    path_data = "./Data/Optimization_{}/".format(optimization)
    name = "SFS-tau={}_kappa={}".format(tau, kappa)

    # Generate x genomic data for the same kappa and tau
    for _ in range(nb_simu):
        # Simulation with msprime
        sfs = ms.msprime_simulation(model=msprime_model, param=params, kappa=kappa, tau=tau)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, dadi_model.__name__, path=path_data, name=name)

        # Dadi inference
        tmp = {"LL": [], "SFS": []}
        control_ll, *_ = dadi.dadi_inference(pts_list, control_model, path=path_data, name=name)

        for _ in range(nb_simu):  # run x inference with dadi from the same observed data
            ll, _, model = dadi.dadi_inference(
                pts_list, dadi_model, opt=optimization, path=path_data, name=name
            )
            tmp["LL"].append(ll)
            tmp["SFS"].append(model)

        # Select the best one, i.e. the one with the highest likelihood
        # Compute the log-likelihood ratio
        ll_ratio.append(log_likelihood_ratio(tmp["LL"], control_ll))

        # Keep track of log-likelihood for each simulation
        ll_list["Model0"].append(control_ll)
        ll_list["Model1"].append(max(tmp["LL"]))

        # Keep track of the SFS of the best inference, i.e. with the highest log-likelihood
        if save:
            index = tmp["LL"].index(max(tmp["LL"]))
            model_list["LL"].append(tmp["LL"][index])
            model_list["SFS"].append(tmp["SFS"][index])

    # Keep track of some SFS generated from dadi
    if save:
        index = model_list["LL"].index(max(model_list["LL"]))
        sfs = model_list["SFS"][index]
        sfs.to_file("{}{}-inferred.sfs".format(path_data, name))
    else:
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

    # Set up tau & kappa for the simulation and inference
    if optimization == "tau":
        print("Optimization of {} for {}".format(optimization, dadi_model.__name__))
        kappa, tau, dof = 10, np.float_power(10, scale[0]), 1  # Kappa fixed

    elif optimization == "kappa":
        print("Optimization of {} for {}".format(optimization, dadi_model.__name__))
        kappa, tau, dof = np.float_power(10, scale[0]), 1.0, 1  # Tau fixed

    else:
        print("Optimization of {} for {}".format(optimization, dadi_model.__name__))
        kappa, tau, dof = np.float_power(10, scale[1]), np.float_power(10, scale[0]), 1

    lrt, ll_list = likelihood_ratio_test(
        tau, kappa, msprime_model, dadi_model, control_model, optimization, save,
        nb_simu=10, dof=dof
    )  # 1000 solutions
    row = {
        "Tau": tau, "Kappa": kappa, "Positive hit": Counter(lrt)[1],
        "Model0 ll": ll_list["Model0"], "Model1 ll": ll_list["Model1"]
    }
    data = data.append(row, ignore_index=True)

    # Export data to csv file
    path_data = "./Data/Optimization_{}/".format(optimization)
    data.to_json("{}opt-tau={}_kappa={}.json".format(path_data, tau, kappa))


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

    if args.analyse == 'msprime':
        # sfs_shape_verification()

        mu, sample = 2e-2, 20
        # Parameters for the simulation
        params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)
        # Simulation with msprime
        sfs = ms.msprime_migration_simulation(
            model=ms.simple_migration_model, param=params, migration_rate=0.1, kappa=10,
            debug=True
        )
        print(sfs)

    elif args.analyse == 'opt':
        dadi_params_optimisation(args.number)

    elif args.analyse == 'lrt':
        inference(
            ms.sudden_decline_model, dadi.sudden_decline_model, dadi.constant_model,
            optimization=args.param, scale=args.value, save=True
        )

    elif args.analyse == 'er':
        for sample in [10, 20, 40, 60, 100]:
            plot.plot_error_rate(sample)

        for opt in ["tau-kappa"]:
            path_data = "./Data/Optimization_{}/".format(opt)
            files = os.listdir(path_data)

            # Pandas DataFrame
            col = ["Tau", "Kappa", "Positive hit", "Model0 ll", "Model1 ll"]
            data = pd.DataFrame(columns=col)

            # Append value to data
            for fichier in files:
                data = data.append(
                    pd.read_csv("{}{}".format(path_data, fichier), sep="\t"), ignore_index=True
                )

    elif args.analyse == 'ases':
        path_data = "./Data/Optimization_{}/".format(args.param)

        spectrums = sorted([ele for ele in os.listdir(path_data) if ele.startswith("SFS") \
                            and not ele.endswith("inferred.fs")])

        sfs = []
        for spectr in spectrums:
            with open("{}{}".format(path_data, spectr), "r") as filin:
                lines = filin.readlines()
                sfs.append([int(ele) for ele in lines[1].strip().split(" ")[1: -1]])
            with open("{}{}-inferred.fs".format(path_data, spectr.split('.fs')[0]), "r") as filin:
                lines = filin.readlines()
                sfs.append([float(ele) for ele in lines[1].strip().split(" ")[1: -1]])

            # Plot
            # Theoretical SFS for any constant population
            sfs_theorique = [0] * (20 - 1)
            for i in range(len(sfs_theorique)):
                sfs_theorique[i] = 1 / (i+1)
            sfs.append(sfs_theorique)

            plot.plot_sfs(
                sfs=sfs,
                label=["Observed", "Inferred", "Théorique"],
                color=["tab:blue", "tab:orange", "tab:red"],
                title="Unfold SFS for various scenarios",
                name="Test-sfs"
            )
            break
        #sys.exit()

        # Generate plot puissance
        dico = {
            "Tau": np.array([], dtype=float), "Kappa": np.array([], dtype=float),
            "Positive hit": np.array([], dtype=int),
            "Model0 ll": np.array([], dtype=list), "Model1 ll": np.array([], dtype=list)
        }
        data = pd.DataFrame(dico)

        for fichier in [ele for ele in os.listdir(path_data) if ele.startswith("opt")]:
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            data = data.append(res, ignore_index=True)

        plot.plot_lrt(data)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
