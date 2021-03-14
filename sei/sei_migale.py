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
# Generate a set of SFS with msprime                                 #
######################################################################

def length_from_file(fichier, params, mu, snp):
    """
    Extract length factor from file and return the length of the sequence.
    """
    res = pd.read_json(path_or_buf="{}".format(fichier), typ='frame')
    factor = res[res['Parameters'] == params]['Factor'].values[0]

    return (snp / factor) / (4 * 1 * mu)


def generate_sfs(params, model, nb_simu, path_data, path_length):
    """
    Generate a set of unfolded sfs of fixed SNPs size with msprime.
    """
    mu = 8e-2

    # Data
    data = pd.DataFrame(columns=['Parameters', 'SFS observed', 'SNPs', 'Time'])

    # Define length
    length = length_from_file(path_length, params, mu, snp=100000)

    # Convert params from log scale
    params.update({k: np.power(10, v) for k, v in params.items()})

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
# Main                                                               #
######################################################################

if __name__ == "__main__":
    # inference(msprime_model=ms.sudden_decline_model, dadi_model=dadi.sudden_decline_model,
    #           control_model=dadi.constant_model, optimization="tau")

    args = arg.arguments()

    if args.analyse == 'data' and args.model == "decline":

        if args.model == "decline":

            # Range of value for tau & kappa
            tau_list, kappa_list = np.arange(-4, 2.5, 0.1), np.arange(-3.5, 3, 0.1)
            params = []
            for tau in tau_list:
                for kappa in kappa_list:
                    params.append({'Tau': round(tau, 2), 'Kappa': round(kappa, 2)})

            params, model = params[args.value-1], ms.sudden_decline_model

            # Path of data
            path_data = "/home/pimbert/work/Species_evolution_inference/Data/Msprime/sfs_{0}/"\
                "SFS_{0}-tau={1}_kappa={2}"\
                .format(args.model, params['Tau'], params['Kappa'])

        path_length = \
            "/home/pimbert/work/Species_evolution_inference/Data/Msprime/length_factor-{}"\
            .format(args.model)

        generate_sfs(params, model, nb_simu=1, path_data=path_data, path_length=path_length)

    elif args.analyse == 'opt':
        sample = [10, 20, 40, 60, 100]
        dadi_params_optimisation(sample[args.number-1])
