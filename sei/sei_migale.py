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

    for i, row in res.iterrows():  # bug with some value of tau or kappa - many decimal points
        res.at[i, 'Parameters'] = {k: round(v, 2) for k, v in row['Parameters'].items()}

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
    # length = length_from_file(path_length, params, mu, snp=100000)
    length = 2e2

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
# Inference with Dadi                                                #
######################################################################

def likelihood_ratio_test(sfs_observed, models, sample, fixed, path_data, dof, value=None):
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
        fixed parameter for the inference, either (tau), (kappa) or (migration)
    dof: int
        degrees of freedom

    Return
    ------
    data: dictionary
      - LRT
        Likelihood ratio test, either 1 - test significant and reject of H0
                                   or 0 - test insignificant and no reject of H0
      - M0
        List of log likelihood and sfs for the inference with M0
      - M1
        List of log likelihood, sfs and estimated parameters for the inference with M1.
        In this case from the same observed SFS, 1000 inferences with M1 are made and only the
        best one is kept. I.E. the one with the highest log-likelihood.
      - Time
        Mean execution time for the inference
    """
    data = {
        'LRT': [], 'M0': {'LL': [], 'SFS': 0}, 'M1': {'LL': [], 'SFS': [], 'Estimated': []},
        'Time': 0
    }
    execution = []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    for sfs in sfs_observed:
        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, models['Inference'].__name__, path=path_data)

        # Dadi inference for M0
        # Pairs (Log-likelihood, Inferred SFS)
        m0_inference = dadi.inference(pts_list, models['Control'], path=path_data)
        data['M0']['LL'].append(m0_inference[0])
        data['M0']['SFS'] = m0_inference[1]

        # Dadi inference for M1
        m1_inferences, m1_execution = [], []

        for _ in range(100):  # Run 1000 inferences with dadi from the observed sfs
            start_inference = time.time()

            # Pairs (Log-likelihood, Inferred SFS, Params)
            tmp = dadi.inference(pts_list, models['Inference'], fixed=fixed, value=value,
                                 path=path_data)

            m1_inferences.append(tmp)
            m1_execution.append(time.time() - start_inference)

        execution.append(sum(m1_execution) / 10)

        # m1_inferences is a list of pairs (Log-likelihood, Inferred SFS, Params)
        # Compare each item of this list by the value at index 0, i.e. the log-likelihood and
        # select the one with this highest value.
        index_best_ll = m1_inferences.index((max(m1_inferences, key=lambda ele: ele[0])))

        data['M1']['LL'].append(m1_inferences[index_best_ll][0])
        data['M1']['SFS'].append(m1_inferences[index_best_ll][1])
        data['M1']['Estimated'].append(m1_inferences[index_best_ll][2])

        # Compute the log-likelihood ratio test between M0 and M1
        lrt = 2 * (m1_inferences[index_best_ll][0] - m0_inference[0])  # LL ratio test
        p_value = chi2.sf(lrt, dof)  # Chi2 test

        if p_value > 0.05:
            data['LRT'].append(0)  # Test insignificant and no reject of H0
        else:
            data['LRT'].append(1)  # Test significant and reject of H0

        # Mean execution time for the inference
        data['Time'] = round(sum(execution) / len(sfs_observed), 4)

        return data


def inference_dadi(simulation, models, path_data, fixed, value):
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
        Value of the fixed parameters for the inference
    """
    col = ['Parameters', 'Positive hit', 'SNPs', 'SFS observed', 'M0', 'M1', 'Time']
    data = pd.DataFrame(columns=col)

    for _, row in simulation.iterrows():
        sfs_observed, sample = row['SFS observed'], row['Parameters']['sample_size']

        if fixed == 'tau' and round(row['Parameters']['Tau'], 2) == value:
            inf = likelihood_ratio_test(sfs_observed, models, sample, fixed, path_data, dof=2,
                                        value=value)

        elif fixed == 'kappa' and round(row['Parameters']['Kappa'], 2) == value:
            inf = likelihood_ratio_test(sfs_observed, models, sample, fixed, path_data, dof=2,
                                        value=value)

        elif fixed is None:
            inf = likelihood_ratio_test(sfs_observed, models, sample, fixed, path_data, dof=2)

        else:
            continue

        params = {k: v for k, v in row['Parameters'].items() if k in ['Tau', 'Kappa']}

        dico = {
            'Parameters': params, 'Positive hit': inf['LRT'], 'SNPs': row['SNPs'],
            'SFS observed': sfs_observed, 'M0': inf['M0'], 'M1': inf['M1'], 'Time': inf['Time']
        }
        data = data.append(dico, ignore_index=True)

    # Export dataframe to json files
    if fixed is None:
        name = "dadi-{}".format(models['Inference'].__name__)
    else:
        name = "dadi-{}-{}={}".format(models['Inference'].__name__, fixed, value)
    data.to_json("{}{}".format(path_data, name))

    # Remove SFS file
    os.remove("{}SFS.fs".format(path_data))


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

    elif args.analyse == 'inf':
        path_data = "/home/pimbert/work/Species_evolution_inference/Data/Msprime/sfs_{}/" \
            .format(args.model)
        filein = "SFS_{}-all.json".format(args.model)

        # Export the observed SFS to DataFrame
        simulation = f.export_json_files(args.model, filein, path_data)

        # Inference with dadi
        if args.dadi:

            # Set up M0 & M1 model for the inference with dadi
            if args.model == "decline":
                models = \
                    {'Inference': dadi.sudden_decline_model, 'Control': dadi.constant_model}
            elif args.model == "migration":
                models = \
                    {'Inference': dadi.two_pops_migration_model, 'Control': dadi.constant_model}

            # Path data
            path_data = "/home/pimbert/work/Species_evolution_inference/Data/Dadi/{}/" \
                .format(args.model)

            inference_dadi(simulation, models, path_data, fixed=args.param, value=args.value)


