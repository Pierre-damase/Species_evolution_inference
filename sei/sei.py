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
    Set up the parametres for the simulation with msprime.
    """
    parameters = {
        "sample_size": sample, "size_population": ne, "rcb_rate": rcb_rate, "mu": mu,
        "length": length
    }
    return parameters


######################################################################
# Generate a set of SFS with msprime                                 #
######################################################################

def length_from_file(fichier, name, mu, snp):
    with open(fichier, "r") as filin:
        length_factor = filin.readlines()[0].split(" ")
    index = int(name.split('-')[1]) - 1
    return (snp / float(length_factor[index])) / (4 * 1 * mu)


def generate_sfs(params, model, nb_simu, path_data, name):
    """
    Generate a set of unfolded sfs of fixed SNPs size with msprime.
    """
    # Data
    data = pd.DataFrame(columns=['Parameters', 'SFS', 'SNPs'])

    # Convert params from log scale
    params = {k: np.power(10, v) for k, v in params.items()}

    # Parameters for the simulation
    params.update(simulation_parameters(sample=20, ne=1, rcb_rate=8e-5, mu=8e-5, length=1e5))

    sfs, snp = [], []
    for _ in range(nb_simu):
        sfs_observed = ms.msprime_simulation(model=model, params=params)
        sfs.append(sfs_observed)
        snp.append(sum(sfs_observed))

    # Export DataFrame to json file
    row = {'Parameters': params, 'SFS': sfs, 'SNPs': snp}
    data = data.append(row, ignore_index=True)

    data.to_json("{}{}".format(path_data, name))


######################################################################
# SFS verification                                                   #
######################################################################

def sfs_shape_verification():
    """
    Method to check the SFS obtained with msprime.

    I.E. check that:
     - The SFS of a constant population fits well to the theoretical SFS of any constant
       population
     - The SFS of an increasing or decreasing population
    """
    # Fixed parameters for the simulation
    params = simulation_parameters(sample=10, ne=1, rcb_rate=2e-2, mu=2e-2, length=1e5)

    # Constant scenario
    print("Constant scenario !")
    sfs_cst = ms.msprime_simulation(model=ms.constant_model, params=params, debug=True)

    # Define tau & kappa for decline/growth scenario
    params.update({"Tau": 1.0, "Kappa": 10.0})

    print("\n\nDeclin scenario !")
    sfs_declin = \
        ms.msprime_simulation(model=ms.sudden_decline_model, params=params, debug=True)

    print("\n\nGrowth scenario !")
    sfs_croissance = \
        ms.msprime_simulation(model=ms.sudden_growth_model, params=params, debug=True)

    # Migration scenario
    print("\n\nMigration scenario !")
    params.update({"Kappa": 10.0, "m12": 1.0, "m21": 0})
    sfs_migration = \
        ms.msprime_simulation(model=ms.two_pops_migration_model, params=params, debug=True)

    # Theoretical SFS for any constant population
    sfs_theorique = [0] * (params["sample_size"] - 1)
    for i in range(len(sfs_theorique)):
        sfs_theorique[i] = 1 / (i+1)

    # Plot
    plot.plot_sfs(
        sfs=[sfs_cst, sfs_theorique, sfs_declin, sfs_croissance, sfs_migration],
        label=["Constant", "Theoretical", "Declin", "Growth", "Migration"],
        color=["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:gray"],
        style=["solid", "solid", "solid", "solid", "dashed"],
        title="Unfold SFS for various scenarios", axis=True
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
        fixed_params = simulation_parameters(sample=20, ne=1, rcb_rate=mu, mu=mu, length=1e5)
        print("Msprime simulation - sample size {} & mutation rate {}".format(20, mu))

        # Msprime simulation
        sfs = ms.msprime_simulation(model=ms.constant_model, fixed_params=fixed_params)

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


def likelihood_ratio_test(params, models, optimization, nb_simu, dof, name):
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
    params: dictionary

    models: dictionary

    dof: int
        degrees of freedom

    Return
    ------
    data: dictionary
      - LL: list of the log-likelihood
        M0: log-likelihood for the model with less parameters
        M1: best log-likelihood for the model with more parameters
      - SNP: list of the SNPs
        SNPs for the observed sfs generated with msprime
      - LRT: list of log likelihood ratio test
        List of 0 (negative run) & 1 (positive run)
      - Time simulation: float
        The mean time of execution of the x simulations with msprime
      - Time inference: float
        The mean time of execution of the x inferences with dadi
      - SFS: list of SFS
        Obs: the observed SFS generated with msprime
        M0: the inferred SFS for M0 the model with less parameters
        M1: the inferred SFS for M1 the model with more parameters
    """
    mu, sample, ll_ratio = 8e-2, 20, []  # 8e-2

    data = {
        "LL": {"M0": [], "M1": []}, "SNPs": [], "LRT": [], "Time simulation": 0,
        "Time inference": 0, "SFS": {"Obs": [], "M0": [], "M1": []}
    }

    # Time of execution
    simulation_time, inference_time = [], []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Parameters for the simulation
    if optimization == "tau":
        fichier = "./Data/length_factor-kappa={}".format(params['Kappa'])
        length = length_from_file(fichier, name, mu, 500000)

    elif optimization == "kappa":
        fichier = "./Data/length_factor-tau={}".format(params['Tau'])
        length = length_from_file(fichier, name, mu, 500000)

    else:
        length = 1e5

    fixed_params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=length)

    # Path & name
    path_data = "./Data/Optimization_{}/".format(optimization)
    name = "SFS-{}".format(name)

    # Generate x genomic data for the same kappa and tau
    for _ in range(nb_simu):
        # Simulation with msprime
        start_simulation = time.time()

        sfs_observed = ms.msprime_simulation(
            model=models["Simulation"], fixed_params=fixed_params, params=params
        )

        data['SNPs'].append(sum(sfs_observed))
        simulation_time.append(time.time() - start_simulation)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs_observed, models["Inference"].__name__, path=path_data, name=name)

        # Dadi inference
        ll_list, tmp, sfs_m1 = [], [], []
        control_ll, _, sfs_m0 = dadi.dadi_inference(
            pts_list, models["Control"], path=path_data, name=name
        )

        for _ in range(nb_simu):  # run x inference with dadi from the same observed data
            start_inference = time.time()

            ll, _, sfs_inferred = dadi.dadi_inference(
                pts_list, models["Inference"], opt=optimization, path=path_data, name=name
            )

            ll_list.append(ll)
            tmp.append(time.time() - start_inference)
            sfs_m1.append(list(sfs_inferred[1:-1]))

        # Select the best one, i.e. the one with the highest likelihood
        # Compute the log-likelihood ratio
        ll_ratio.append(log_likelihood_ratio(ll_list, control_ll))

        # Mean execution time of x inference with dadi
        inference_time.append(sum(tmp) / nb_simu)

        # Keep track of log-likelihood for each simulation
        data['LL']['M0'].append(control_ll)
        data['LL']['M1'].append(max(ll_list))

        # Keep track of the observed SFS and inferred SFS for M0 & M1 (best one)
        data['SFS']['Obs'].append(sfs_observed[1:-1])
        data['SFS']['M0'].append(list(sfs_m0[1:-1]))
        data['SFS']['M1'].append(sfs_m1[ll_list.index(max(ll_list))])

    # Delete sfs file
    os.remove("{}{}.fs".format(path_data, name))

    # Mean time execution for simulation and inference
    data['Time simulation'] = round(sum(simulation_time) / nb_simu, 3)
    data['Time inference'] = round(sum(inference_time) / nb_simu, 3)

    # Likelihood-ratio test
    data['LRT'] = [1] * len(ll_ratio)
    for i, chi_stat in enumerate(ll_ratio):
        p_value = chi2.sf(chi_stat, dof)
        if p_value > 0.05:
            data['LRT'][i] = 0

    return data


def inference(models, optimization, scale, name):
    """

    Parameter
    ---------
    models: dictionary
      - Simulation: the custom model to generate genomic data
      - Inference: the custom model to infer demography history
      - Control: the control model - model with less parameters
    optimization
        parameter to optimize - (kappa), (tau), (kappa, tau), etc.
    """
    col = [
        "Parameters", "Positive hit", "SNPs", "SFS obs", "SFS M0", "SFS M1",
        "Model0 ll", "Model1 ll", "Time simulation", "Time inference"
    ]
    data = pd.DataFrame(columns=col)

    # Set up tau & kappa for the simulation and inference
    if optimization == "tau":
        params = {"Kappa": 10, "Tau": np.float_power(10, scale[0])}  # Kappa fixed
        dof = 1
    elif optimization == "kappa":
        params = {"Kappa": np.float_power(10, scale[0]), "Tau": 1}  # Tau fixed
        dof = 1
    elif optimization == "tau-kappa":
        params = {"Kappa": np.float_power(10, scale[1]), "Tau": np.float_power(10, scale[0])}
        dof = 2

    print("Likelihood ratio test - optimization of ({}) with x = 100 simulations"
          .format(optimization))

    values = likelihood_ratio_test(params, models, optimization, nb_simu=2, dof=dof, name=name)

    print("Likelihood ratio test done !!!\n")

    row = {
        "Parameters": params, "Positive hit": Counter(values['LRT'])[1], "SNPs": values['SNPs'],
        "SFS obs": values['SFS']['Obs'], "SFS M0": values['SFS']['M0'],
        "SFS M1": values['SFS']['M1'],
        "Model0 ll": values['LL']['M0'], "Model1 ll": values['LL']['M1'],
        "Time simulation": values['Time simulation'], "Time inference": values['Time inference']
    }
    data = data.append(row, ignore_index=True)

    # Export data to json file
    path_data = "./Data/Optimization_{}/".format(optimization)
    data.to_json("{}opt_{}.json".format(path_data, name))


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

    if args.analyse == 'data':

        if args.model == "decline":
            # Range of value for tau & kappa
            tau_list, kappa_list = np.arange(-4, 4, 0.1), np.arange(-3.3, 3.1, 0.08)
            params = []
            for tau in tau_list:
                for kappa in kappa_list:
                    params.append({'Tau': round(tau, 2), 'Kappa': round(kappa, 2)})

            params, model = params[args.value-1], ms.sudden_decline_model

            # Path of data & file name
            path_data = "./Data/Msprime/sfs_{}/".format(args.model)
            name = "SFS_{}-tau={}_kappa={}".format(args.model, params['Tau'], params['Kappa'])

        generate_sfs(params, model, nb_simu=2, path_data=path_data, name=name)

    elif args.analyse == 'msprime':
        sfs_shape_verification()

    elif args.analyse == 'opt':
        dadi_params_optimisation(args.number)

    elif args.analyse == 'lrt':
        if args.param == 'tau':
            scale = [np.arange(-4, 4, 0.1)[int(args.value[0])-1]]
        elif args.param == 'kappa':
            scale = [np.arange(-3.3, 3.1, 0.08)[int(args.value[0])-1]]
        else:
            scale = [
                np.arange(-4, 4.1, 0.1)[int(args.value[0])-1],
                np.arange(0.05, 4.1, 0.05)[int(args.value[1])-1]
            ]

        models = {
            "Simulation": ms.sudden_decline_model, "Inference": dadi.sudden_decline_model,
            "Control": dadi.constant_model
        }
        name = "{}-{}".format(args.param, int(args.value[0]))

        inference(models=models, optimization=args.param, scale=scale, name=name)

    elif args.analyse == 'er':
        for sample in [10, 20, 40, 60, 100]:
            plot.plot_error_rate(sample)

    elif args.analyse == 'ases':
        path_data = "./Data/Optimization_{}_d2/".format(args.param)
        data = f.export_to_dataframe(path_data)

        plot.plot_weighted_square_distance(data)

        # Generate plot puissance
        path_data = "./Data/Optimization_{}/".format(args.param)
        col = [
            "Parameters", "Positive hit", "SNPs",
            "Model0 ll", "Model1 ll", "Time simulation", "Time inference"
        ]
        data = pd.DataFrame(columns=col)

        for fichier in [ele for ele in os.listdir(path_data) if ele.startswith("opt")]:
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            #data = data.append(res, ignore_index=True)

        #plot.plot_lrt(data)

    elif args.analyse == 'snp':
        plot.snp_distribution()

    elif args.analyse == 'stairway':
        # Fixed parameters for the simulation
        fixed_params = \
            simulation_parameters(sample=20, ne=1, rcb_rate=8e-4, mu=8e-4, length=1e5)

        params = {"Tau": 1.0, "Kappa": 10.0}

        sfs = ms.msprime_simulation(
            model=ms.sudden_decline_model, fixed_params=fixed_params, params=params, debug=True)

        path_data = "/home/damase/All/Cours/M2BI-Diderot/Species_evolution_inference/sei/" \
            "inference/stairway_plot_v2.1.1/"
        name = "test"

        # Generate the SFS file compatible with stairway plot v2
        data = {k: v for k, v in fixed_params.items() if k in ['sample_size', 'length', 'mu']}
        data['sfs'], data['year'], data['ninput'] = sfs, 1, 200

        f.stairway_data(name, data, path_data)

        # Create the batch file
        os.system("java -cp {0}stairway_plot_es Stairbuilder {0}{1}.blueprint"
                  .format(path_data, name))

        # Run teh batch file
        os.system("bash {}{}.blueprint.sh".format(path_data, name))

        # Remove all blueprint file
        os.system("rm -rf {}{}.blueprint*".format(path_data, name))


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
