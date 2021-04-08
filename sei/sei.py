"""
Decline estimation from genomic data.
"""

import copy
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
        "sample_size": sample, "Ne": ne, "rcb_rate": rcb_rate, "mu": mu, "length": length
    }
    return parameters


def define_parameters(model):
    """
    Define pairs of (Tau, Kappa) - sudden decline/growth model - and (m12, Kappa) - migration
    model.
    """
    if model == 'decline':
        # Range of value for tau & kappa
        params = []
        for tau in np.arange(-4, 2.5, 0.1):
            for kappa in np.arange(-3.5, 3, 0.1):
                params.append({'Tau': round(tau, 2), 'Kappa': round(kappa, 2)})

    else:
        # Range of value for m12 & kappa
        params = []
        for m12 in np.arange(-4, 2.5, 0.1):
            for kappa in np.arange(-3.5, 3, 0.1):
                params.append({'m12': round(m12, 2), 'm21': 0.0, 'Kappa': round(kappa, 2)})

    return params


######################################################################
# SFS shape verification                                             #
######################################################################

def  compute_theoritical_sfs(length):
    """
    Compute the theoritical SFS of any constant population.
    """
    theoritical_sfs = [0] * (length)
    for i in range(length):
        theoritical_sfs[i] = 1 / (i+1)
    return theoritical_sfs


def generate_set_sfs():
    """
    Generate a set of sfs for various scenario with msprime:

      - Constant population
      - Theoritical SFS for any constant population
      - Sudden growth model - growth of force kappa at a time tau in the past
      - Sudden decline model - decline of force kappa at a time tau in the past
      - Migration model - migration into population 1 from 2 and no migration into 2 from 1

    Return
    ------
    sfs: dictionary
        sfs for various scenario
    parameters: dictionary
        specific value of tau & kappa for the growth and decline model
        specifiv value of m12, m21 & kappa for the migration model
    params_simulation:
        Ne, sample size, length, mutation & recombination rate
    """
    sfs, parameters = {}, {}

    # Parameters for the simulation
    params = simulation_parameters(sample=10, ne=1, rcb_rate=2e-2, mu=2e-2, length=1e5)

    # Constant scenario
    sfs['Constant model'] = \
        ms.msprime_simulation(model=ms.constant_model, params=params, debug=True)

    # Theoretical SFS for any constant population
    sfs['Theoretical model'] = \
        compute_theoritical_sfs(length=params["sample_size"] - 1)

    # Define tau & kappa for decline/growth scenario
    params.update({"Tau": 1.0, "Kappa": 10.0})

    sfs['Decline model'] = \
        ms.msprime_simulation(model=ms.sudden_decline_model, params=params, debug=True)
    parameters['Decline model'] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}

    sfs['Growth model'] = \
        ms.msprime_simulation(model=ms.sudden_growth_model, params=params, debug=True)
    parameters['Growth model'] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}

    # Migration scenario
    params.update({"Kappa": 10.0, "m12": 1.0, "m21": 0})

    sfs['Migration model'] = \
        ms.msprime_simulation(model=ms.twopops_migration_model, params=params, debug=True)
    parameters['Migration model'] = {
        k: v for k, v in params.items() if k in ['m12', 'm21', 'Kappa']
    }

    params_simulation = \
        {k: v for k, v in params.items() if k not in ['m12', 'm21', 'Kappa', 'Tau']}

    return sfs, parameters, params_simulation


######################################################################
# Generate a set of SFS with msprime                                 #
######################################################################

def length_from_file(fichier, params, mu, snp):
    """
    Extract length factor from file and return the length of the sequence.
    """
    res = pd.read_json(path_or_buf="{}".format(fichier), typ='frame')

    # Bug of to_json or from_json method of pandas
    # Some value of tau, kappa or m12 with many decimal points
    for i, row in res.iterrows():
        res.at[i, 'Parameters'] = {k: round(v, 2) for k, v in row['Parameters'].items()}

    factor = res[res['Parameters'] == params]['Factor'].values[0]

    return (snp / factor) / (4 * 1 * mu)


def generate_sfs(params, model, nb_simu, path_data, path_length):
    """
    Generate a set of unfolded sfs of fixed SNPs size with msprime.
    """
    # Define length
    length = length_from_file(path_length, params, mu=8e-2, snp=100000)

    # Convert params from log scale
    params.update({k: (np.power(10, v) if k != 'm21' else v) for k, v in params.items()})

    # Parameters for the simulation
    params.update(
        simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=length))

    sfs, snp, execution = [], [], []
    for _ in range(nb_simu):
        start_time = time.time()

        sfs_observed = ms.msprime_simulation(model=model, params=params)
        sfs.append(sfs_observed)
        snp.append(sum(sfs_observed))

        execution.append(time.time() - start_time)

    # Create DataFrame form dictionary
    dico = {
        'Parameters': [params], 'SFS observed': [sfs], 'SNPs': [snp],
        'Time': [round(np.mean(execution), 4)]
    }
    data = pd.DataFrame(dico)

    # Export DataFrame to json file
    data.to_json("{}".format(path_data))


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
# Inference with Dadi                                                #
######################################################################

def likelihood_ratio_test(ll_m0, ll_m1, dof):
    """
    Likelihood-ratio test to assesses the godness fit of two model.

    c.f. jupyter notebook "analyse.ipynb" for more information

    Parameters
    ----------
    ll_m0: float
        log-likelihood of model m0
    ll_m1: float
        log-likelihood of model m1
    dof: int
        degree of freedom

    Return
    ------
    Either 1 - test significant and reject of H0
        Or 0 - test insignificant and no reject of H0
    """
    lrt = 2 * (ll_m1 - ll_m0)  # LL ratio test
    p_value = chi2.sf(lrt, dof)  # Chi2 test

    if p_value > 0.05:
        return 0  # test insignificant and no reject of h0
    return 1  # test significant and reject of h0


def weighted_square_distance(sfs):
    """
    Compute the weighted square distance d2.

    c.f. jupyter notebook "analyse.ipynb" for more information

    Parameter
    ---------
    sfs: dictionary
        Either observed SFS and inferred SFS with M1
            Or inferred SFS of two models - M0 & M1

    Return
    ------
    d2: float
        the weighted square distance
    """
    # Normalization of the SFS
    normalized_sfs = {}
    for key, spectrum in sfs.items():
        normalized_sfs[key] = [ele / sum(spectrum) for ele in spectrum]

    # Weighted square distance
    if "Observed" in sfs.keys():
        d2 = [
            np.power(eta_model - eta_obs, 2) / eta_model for eta_obs, eta_model in
            zip(normalized_sfs['Observed'], normalized_sfs['Model'])
        ]
    else:
        d2 = [
            np.power(eta_m0 - eta_m1, 2) / (np.mean([eta_m0, eta_m1])) for eta_m0, eta_m1 in
            zip(normalized_sfs['M0'], normalized_sfs['M1'])
        ]

    del normalized_sfs

    return sum(d2)


def compute_dadi_inference(sfs_observed, models, sample, path_data, job, dof, fixed, value):
    """
    Parameter
    ---------
    sfs_observed: list
        the observed SFS generated with msprime
    models: dictionary
      - Inference
        The model with more parameters, i.e. M1
      - Control
        The model with less parameters, i.e. M0
    sample: int
        The number of sampled monoploid genomes
    fixed: str
        fixed parameter for the inference, either (tau), (kappa) or (migr)
    dof: int
        degrees of freedom

    Return
    ------
    data: dictionary
      - LRT
        Likelihood ratio test
      - M0
        List of log likelihood and sfs for the inference with M0
      - M1
        List of log likelihood, sfs and estimated parameters for the inference with M1.
        In this case from the same observed SFS, 1000 inferences with M1 are made and only the
        best one is kept. I.E. the one with the highest log-likelihood.
      - Time
        Mean execution time for the inference
      - d2 observed inferred
        Weighted square distance between observed SFS & inferred SFS with M1
      - d2 models
        Weighted square distance between inferred SFS with M0 & M1
    """
    data = {
        'LRT': [], 'M0': {'LL': [], 'SFS': []}, 'M1': {'LL': [], 'SFS': [], 'Estimated': []},
        'Time': 0, 'd2 observed inferred': [], 'd2 models': []
    }
    execution = []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    for i, sfs in enumerate(sfs_observed):
        # Generate the SFS file compatible with dadi
        if not value:
            dadi_file = "SFS-{}".format(job)
        else:
            dadi_file = "SFS_{}-{}".format(value, job)

        f.dadi_data(sfs, models['Inference'].__name__, path=path_data,
                    name=dadi_file)

        # Dadi inference for M0
        # Pairs (Log-likelihood, Inferred SFS)
        m0_inference = dadi.inference(pts_list, models['Control'], path=path_data,
                                      name=dadi_file)
        data['M0']['LL'].append(m0_inference[0])
        data['M0']['SFS'].append(m0_inference[1])

        # Dadi inference for M1
        m1_inferences, m1_execution = [], []

        for _ in range(1):  # Run 1000 inferences with dadi from the observed sfs
            start_inference = time.time()

            # Pairs (Log-likelihood, Inferred SFS, Params)
            tmp = dadi.inference(pts_list, models['Inference'], fixed=fixed, value=value,
                                 path=path_data, name=dadi_file)

            m1_inferences.append(tmp)
            m1_execution.append(time.time() - start_inference)

        execution.append(np.mean(m1_execution))

        # m1_inferences is a list of pairs (Log-likelihood, Inferred SFS, Params)
        # Compare each item of this list by the value at index 0, i.e. the log-likelihood and
        # select the one with this highest value.
        index_best_ll = m1_inferences.index((max(m1_inferences, key=lambda ele: ele[0])))

        data['M1']['LL'].append(m1_inferences[index_best_ll][0])
        data['M1']['SFS'].append(m1_inferences[index_best_ll][1])
        data['M1']['Estimated'].append(m1_inferences[index_best_ll][2])

        # Compute the log-likelihood ratio test between M0 and M1
        data['LRT'].append(
            likelihood_ratio_test(data['M0']['LL'][i], data['M1']['LL'][i], dof)
        )

        # Compute weighted square distance
        data['d2 observed inferred'].append(
            weighted_square_distance({'Observed': sfs, 'Model': data['M1']['SFS'][i]})
        )  # d2 between the observed SFS & inferred SFS with M1

        data['d2 models'].append(
            weighted_square_distance({'M0': data['M0']['SFS'][i], 'M1': data['M1']['SFS'][i]})
        )  # d2 between the inferred SFS of two models - M0 & M1

    # Mean execution time for the inference
    data['Time'] = round(sum(execution) / len(sfs_observed), 4)

    return data


def save_dadi_inference(simulation, models, path_data, job, fixed, value):
    """
    Inference with dadi.

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

    fixed
        fixed parameter for the inference, either (tau), (kappa) or (migration)
    value
        Value of the fixed parameters for the inference - log scale
    """
    # Inference
    sfs_observed, sample = simulation['SFS observed'], simulation['Parameters']['sample_size']

    if not value:
        inf = compute_dadi_inference(sfs_observed, models, sample, path_data, job, dof=2,
                                     fixed=fixed, value=value)
    else:
        inf = compute_dadi_inference(sfs_observed, models, sample, path_data, job, dof=2,
                                     fixed=fixed, value=np.power(10, value))

    # Save data
    params = {
        k: v for k, v in simulation['Parameters'].items() if k in ['Tau', 'Kappa', 'm12',
                                                                   'm21']
    }
    params['Theta'] = 4 * 1 * 8e-2 * simulation['Parameters']['length']  # 4 * Ne * mu * L

    # Create DataFrame from dictionary
    dico = {
        'Parameters': [params], 'Positive hit': [sum(inf['LRT'])],
        'SNPs': [simulation['SNPs']], 'SFS observed': [sfs_observed], 'M0': [inf['M0']],
        'M1': [inf['M1']], 'Time': [inf['Time']],
        'd2 observed inferred': [np.mean(inf['d2 observed inferred'])],
        'd2 models': [np.mean(inf['d2 models'])]
    }
    data = pd.DataFrame(dico)

    # Export dataframe to json files
    if fixed is None:
        name = "dadi_{}-{}".format(models['Inference'].__name__.split('_')[1], job)
    else:
        name = "dadi_{}_{}={}-{}" \
            .format(models['Inference'].__name__.split('_')[1], fixed, value, job)
    data.to_json("{}{}".format(path_data, name))

    # Remove SFS file
    if not value:
        os.remove("{}SFS-{}.fs".format(path_data, job))
    else:
        os.remove("{}SFS_{}-{}.fs".format(path_data, np.power(10, value), job))


######################################################################
# Inference with stairway plot 2                                     #
######################################################################

def compute_stairway_inference(simulation, path_stairway, path_data):
    # Inference
    for i, sfs in enumerate(simulation['SFS observed']):  # Iterate through each observed SFS
        name = "stairway_inference-{}".format(i)

        # Generate the SFS file compatible with stairway plot v2
        data = {
            k: v for k, v in simulation['Parameters'].items() if k in ['sample_size', 'length',
                                                                       'mu']
        }
        data['sfs'], data['year'], data['ninput'] = sfs, 1, 200

        f.stairway_data(name, data, path_data)

        # Create the batch file
        os.system("java -cp {0}stairway_plot_es Stairbuilder {1}{2}.blueprint"
                  .format(path_stairway, path_data, name))

        # Run the batch file
        os.system("bash {}{}.blueprint.sh".format(path_data, name))

        # Remove all blueprint file
        os.system("rm -rf {}{}.blueprint*".format(path_data, name))

        if i == 1:
            break


def save_stairway_inference(simulation, model):
    """
    Inference with stairway plot 2.

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

    model: str
        either decline, migration or cst
    """
    # Set up path data
    path_stairway = "/home/damase/All/Cours/M2BI-Diderot/Species_evolution_inference/sei/" \
        "inference/stairway_plot_v2.1.1/"

    if model == 'decline':
        path_data = path_stairway + "stairway_{}-tau={}_kappa={}/" \
            .format(model, simulation['Parameters']['Tau'], simulation['Parameters']['Kappa'])
    elif model == 'migration':
        path_data = path_stairway + "stairway_{}-m12={}_kappa={}/" \
            .format(model, simulation['Parameters']['m12'], simulation['Parameters']['Kappa'])
    else:
        path_data = path_stairway + "stairway_{}-ne={}/" \
            .format(model, simulation['Parameters']['Ne'])

    if not os.path.isdir(path_data):
        os.mkdir(path_data)
        os.system("cp -r {} {}".format(path_stairway + "stairway_plot_es", path_data))

    # Compute the inference with stairway plot 2
    compute_stairway_inference(simulation, path_stairway, path_data)

    # os.system("rm -rf {}stairway_plot_es".format(path_data))


######################################################################
# Main                                                               #
######################################################################

def main():
    """
    Le main du programme.
    """
    # grid_optimisation()

    args = arg.arguments()

    if args.analyse == 'data':

        # Simulation of constant population - inference of decline population
        if args.model == 'cst':
            # Parameters for the simulation
            params = simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=1e5)

            dico = {'Parameters': params, 'SFS observed': [], 'SNPs': [], 'Time': []}
            for i in range(100):
                print("Simulation {}/100".format(i+1), end="\r")
                start_time = time.time()

                sfs_cst = ms.msprime_simulation(model=ms.constant_model, params=params)
                dico['SFS observed'].append(sfs_cst)
                dico['SNPs'].append(sum(sfs_cst))

                dico['Time'].append(time.time() - start_time)

            # Compute mean execution time
            dico['Time'] = round(np.mean(dico['Time']), 4)

            # Store data in pandas DataFrame
            simulation = pd.DataFrame(columns=['Parameters', 'SFS observed', 'SNPs', 'Time'])
            simulation = simulation.append(dico, ignore_index=True)

            # Export to json
            simulation.to_json("./Data/Msprime/sfs_cst/SFS_{}.json".format(args.model))

            # # Path data
            # path_data = "./Data/Dadi/{}/".format(args.model)

            # # Inference with Dadi
            # print("Inference with dadi")
            # models = {'Inference': dadi.sudden_decline_model, 'Control': dadi.constant_model}
            # save_dadi_inference(simulation, models, path_data, fixed=None, value=None)
            print("Over")
            sys.exit()

        # Generate length factor file
        elif args.file:
            path_data = "./Data/Msprime/{}/".format(args.model)
            filin = "SFS_{}-all".format(args.model)

            # Export the observed SFS to DataFrame
            simulation = f.export_simulation_files(filin, path_data)

            factor, theta = pd.DataFrame(columns=['Parameters', 'Factor']), 32000
            if args.model == 'decline':
                # Convert Tau & kappa to log10
                factor['Parameters'] = simulation['Parameters'].apply(lambda ele: {
                    'Tau': round(np.log10(ele['Tau']), 2),
                    'Kappa': round(np.log10(ele['Kappa']), 2)
                })

            else:
                # Convert m12 & kappa to log10
                factor['Parameters'] = simulation['Parameters'].apply(lambda ele: {
                    'm12': round(np.log10(ele['m12']), 2), 'm21': ele['m21'],
                    'Kappa': round(np.log10(ele['Kappa']), 2)
                })

            # Compute mean SNPs
            factor['Factor'] = simulation['SNPs'].apply(lambda snp: np.mean(snp) / theta)

            # Export pandas DataFrame factor to json file
            factor.to_json('./Data/Msprime/length_factor-{}'.format(args.model))

            sys.exit()

        # Simulation of sudden decline model with msprime for various tau & kappa
        if args.model == 'decline':
            params = define_parameters(args.model)
            params, model = params[args.job-1], ms.sudden_decline_model
            path_data = "./Data/Msprime/{0}/SFS_{0}-tau={1}_kappa={2}"\
                .format(args.model, params['Tau'], params['Kappa'])

        # Simulation of two populations migration models for various migration into 1 from
        # 2 (with m12 the migration rate) and no migration into 2 from 1
        # Population 1 size is pop1 and population 2 size is pop2 = kappa*pop1
        elif args.model == 'migration':
            params = define_parameters(args.model)
            params, model = params[args.job-1], ms.twopops_migration_model

            path_data = "./Data/Msprime/{0}/SFS_{0}-m12={1}_kappa={2}"\
                .format(args.model, params['m12'], params['Kappa'])

        path_length = "./Data/Msprime/length_factor-{}".format(args.model)

        generate_sfs(params, model, nb_simu=10, path_data=path_data, path_length=path_length)

    elif args.analyse == 'opt':
        dadi_params_optimisation(args.number)

    elif args.analyse == 'inf':
        path_data = "./Data/Msprime/{}/".format(args.model)
        filin = "SFS_{}-all".format(args.model)

        # Export the observed SFS to DataFrame
        simulation = f.export_simulation_files(filin, path_data)

        # Inference with dadi
        if args.dadi:

            # Set up M0 & M1 model for the inference with dadi
            if args.model == "decline":
                models = \
                    {'Inference': dadi.sudden_decline_model, 'Control': dadi.constant_model}
            elif args.model == "migration":
                models = \
                    {'Inference': dadi.twopops_migration_model, 'Control': dadi.constant_model}

            # Select observed data for the inference
            if args.param is None:
                simulation = simulation.iloc[args.job - 1]
                path_data = "./Data/Dadi/{}/all/".format(args.model)

            else:
                if args.param != 'm12':
                    param = args.param.capitalize()
                else:
                    param = args.param

                simulation = [
                    ele for _, ele in simulation.iterrows()
                    if round(np.log10(ele['Parameters'][param]), 2) == args.value
                ][args.job-1]
                path_data = "./Data/Dadi/{}/{}/".format(args.model, args.param)

                # args.value = np.power(10, args.value)

            save_dadi_inference(simulation, models, path_data, args.job, fixed=args.param,
                                value=args.value)

        # Inference with stairway plot 2
        elif args.stairway:
            simulation = simulation.iloc[args.job - 1]
            save_stairway_inference(simulation, model=args.model)
            simul

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
