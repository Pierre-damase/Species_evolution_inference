"""
Programme pour inférer l'évolution d'une population à partir de données génomiques.
"""

import os
import random
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

def length_from_file(fichier, params, mu, snp):
    """
    Extract length factor from file and return the length of the sequence.
    """
    res = pd.read_json(path_or_buf="{}".format(fichier), typ='frame')

    print(res[res['Parameters'] == params])
    print(params)
    sys.exit()

    return (snp / float(1)) / (4 * 1 * mu)


def generate_sfs(params, model, nb_simu, path_data, path_length):
    """
    Generate a set of unfolded sfs of fixed SNPs size with msprime.
    """
    mu = 8e-2

    # Data
    data = pd.DataFrame(columns=['Parameters', 'SFS observed', 'SNPs', 'Time'])

    # Define length
    #length = length_from_file(path_length, params, mu, snp=10000)
    #print(length)
    #sys.exit()
    length = 1e4
    nb_simu = 1

    # Convert params from log scale
    params = {k: np.power(10, v) for k, v in params.items()}

    # Parameters for the simulation
    params.update(simulation_parameters(sample=20, ne=1, rcb_rate=mu, mu=mu, length=length))

    sfs, snp, execution = [], [], []
    for _ in range(nb_simu):
        start_time = time.time()

        sfs_observed = ms.msprime_simulation(model=model, params=params)
        sfs.append(sfs_observed)
        snp.append(sum(sfs_observed))

        execution.append(time.time() - start_time)

    # Export DataFrame to json file
    row = {
        'Parameters': params, 'SFS observed': sfs, 'SNPs': snp,
        'Time': round(np.mean(execution), 4)
    }
    data = data.append(row, ignore_index=True)

    data.to_json("{}".format(path_data))


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


def likelihood_ratio_test(params, sfs_observed, models, sample, optimization):
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
    """
    print(params)
    print(sfs_observed)
    print(models)
    data = {
        'LRT': [], 'M0': {'LL': [], 'SFS': 0}, 'M1': {'LL': [], 'SFS': []}, 'Time': 0
    }
    execution, ll_ratio = [], []

    # Path data & name
    path_data = "./Data/Dadi/"

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    for sfs in sfs_observed:

        # Generate the SFS ile compatible with dadi
        f.dadi_data(sfs, models['Inference'].__name__, path=path_data)

        # Dadi inference
        m1_inferrence, m1_execution = [], []

        # Pairs (Log-likelihood, Inferred SFS)
        m0_inferrence = dadi.inference(pts_list, models['Control'], path=path_data)

        for _ in range(10):  # Run 100 inferences with dadi from the observed sfs
            start_inference = time.time()

            # Pairs (Log-likelihood, Inferred SFS, Params)
            tmp = \
                dadi.inference(pts_list, models['Inference'], opt=optimization, path=path_data)

            m1_inferrence.append(tmp)
            m1_execution.append(time.time() - start_inference)

        # Select the best one, i.e the one with the highest log-likelihood
        # Compute the log-likelihood for each simulation
        ll_ratio.append(log_likelihood_ratio())

        


def inference_dadi(simulation, models, optimization):
    """

    Parameter
    ---------
    simulation: dictionary
      - Parameters
        Parameters for the simulation with msprime - mutation rate mu, recombination rate, Ne,
        length L of the sequence, sample size.
      - SNPs
        List of SNPs for each observed SFS
      - SFS observed
        List of the observed SFS generated with msprime for the same set of parameters
      - Time
        Mean execution time to generate the observed SFS

    models: dictionary
      - Inference: the custom model to infer demography history, i.e. the model m1 with more
        parameters
      - Control: the control model, i.e. the model m0 with less parameters
    optimization
        parameter to optimize - (kappa), (tau), (kappa, tau), etc.
    """
    col = ['Parameters', 'Positive hit', 'SNPs', 'SFS observed', 'M0', 'M1', 'Time']
    data = pd.DataFrame(columns=col)

    if models['Inference'].__name__ == 'sudden_decline_model':

        for _, row in simulation.iterrows():
            sfs_observed, sample = row['SFS observed'], row['Parameters']['sample_size']
            params = {k: v for k, v in row['Parameters'].items() if k in ['Tau', 'Kappa']}

            inf = likelihood_ratio_test(params, sfs_observed, models, sample)
            break

    sys.exit()


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
        simulation = pd.DataFrame(columns=['Parameters', 'SNPs', 'SFS', 'Time'])

        for fichier in os.listdir(path_data):
            # Export the json file to pandas DataFrame and store it in simulation
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            simulation = simulation.append(res, ignore_index=True)

            # Delete the json file
            os.remove("{}{}".format(path_data, fichier))

        # Export pandas DataFrame simulation to json file
        # simulation = simulation.rename(columns={'SFS': 'SFS observed'})
        simulation.to_json("{}SFS_{}-all.json".format(path_data, model))

    else:
        simulation = \
            pd.read_json(path_or_buf="{}{}".format(path_data, filein), typ='frame')

    return simulation


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

        if args.snp:
            path_data = "./Data/Msprime/snp_distribution/sfs_{}/".format(args.model)
            filein = "SFS_{}-all.json".format(args.model)

            # Export the observed SFS to DataFrame
            simulation = export_json_files(args.model, filein, path_data)

            # Plot SNPs distribution
            plot.snp_distribution_3d(simulation)
            os.system("jupyter lab sei/graphics/plot.ipynb")

            sys.exit()

        elif args.file:
            path_data = "./Data/Msprime/sfs_{}/".format(args.model)
            filein = "SFS_{}-all.json".format(args.model)

            # Export the observed SFS to DataFrame
            simulation = export_json_files(args.model, filein, path_data)

            factor, theta = pd.DataFrame(columns=['Parameters', 'Factor']), 32000
            if args.model == 'decline':
                # Convert Tau & kappa to log10
                factor['Parameters'] = simulation['Parameters'].apply(lambda ele: {
                    'Tau': round(np.log10(ele['Tau']), 2),
                    'Kappa': round(np.log10(ele['Kappa']), 2)
                })

            # Compute mean SNPs
            factor['Factor'] = simulation['SNPs'].apply(lambda snp: np.mean(snp) / theta)

            # Export pandas DataFrame factor to json file
            factor.to_json('./Data/Msprime/length_factor-{}'.format(args.model))

            sys.exit()

        if args.model == 'decline':
            # Range of value for tau & kappa
            tau_list, kappa_list = np.arange(-4, 4, 0.1), np.arange(-3.3, 3.1, 0.08)

            params = []
            for tau in tau_list:
                for kappa in kappa_list:
                    params.append({'Tau': round(tau, 2), 'Kappa': round(kappa, 2)})
            params, model = params[args.value-1], ms.sudden_decline_model

            path_data = "./Data/Msprime/sfs_{0}/SFS_{0}-tau={1}_kappa={2}"\
                .format(args.model, params['Tau'], params['Kappa'])

        path_length = "./Data/Msprime/length_factor-{}".format(args.model)

        generate_sfs(params, model, nb_simu=2, path_data=path_data, path_length=path_length)

    elif args.analyse == 'msprime':
        sfs_shape_verification()

    elif args.analyse == 'opt':
        dadi_params_optimisation(args.number)

    elif args.analyse == 'inf':
        path_data = "./Data/Msprime/sfs_{}/".format(args.model)
        filein = "SFS_{}-all.json".format(args.model)

        # Export the observed SFS to DataFrame
        simulation = export_json_files(args.model, filein, path_data)

        if args.dadi:  # Dadi

            if args.model == "decline":
                models = \
                    {'Inference': dadi.sudden_decline_model, 'Control': dadi.constant_model}

            inference_dadi(simulation, models, optimization=args.opt)

        elif args.stairway:  # Stairway plot 2
            sys.exit()

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
