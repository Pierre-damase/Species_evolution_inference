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


def define_parameters(model):
    if model == 'decline':
        # Range of value for tau & kappa
        tau_list, kappa_list = np.arange(-4, 2.5, 0.1), np.arange(-3.5, 3, 0.1)

        params = []
        for tau in tau_list:
            for kappa in kappa_list:
                params.append({'Tau': round(tau, 2), 'Kappa': round(kappa, 2)})

    else:
        # Range of value for m12 & kappa
        m12_list, kappa_list = np.arange(-4, 2.5, 0.1), np.arange(-3.5, 3, 0.1)

        params = []
        for m12 in m12_list:
            for kappa in kappa_list:
                params.append({'m12': round(m12, 2), 'm21': 0.0, 'Kappa': round(kappa, 2)})

    return params


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
    # length = length_from_file(path_length, params, mu=8e-2, snp=100000)
    length = 1e3

    # Convert params from log scale
    params.update({k: np.power(10, v) if k != 'm21' else v for k, v in params.items()})

    # Parameters for the simulation
    params.update(simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=length))

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

def likelihood_ratio_test(sfs_observed, models, sample, path_data, job, dof, fixed, value):
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
        fixed parameter for the inference, either (tau), (kappa) or (migr)
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
        'LRT': [], 'M0': {'LL': [], 'SFS': []}, 'M1': {'LL': [], 'SFS': [], 'Estimated': []},
        'Time': 0
    }
    execution = []

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    for sfs in sfs_observed:
        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, models['Inference'].__name__, path=path_data,
                    name="SFS-{}".format(job))

        # Dadi inference for M0
        # Pairs (Log-likelihood, Inferred SFS)
        m0_inference = dadi.inference(pts_list, models['Control'], path=path_data,
                                      name="SFS-{}".format(job))
        data['M0']['LL'].append(m0_inference[0])
        data['M0']['SFS'].append(m0_inference[1])

        # Dadi inference for M1
        m1_inferences, m1_execution = [], []

        for _ in range(1):  # Run 1000 inferences with dadi from the observed sfs
            start_inference = time.time()

            # Pairs (Log-likelihood, Inferred SFS, Params)
            tmp = dadi.inference(pts_list, models['Inference'], fixed=fixed, value=value,
                                 path=path_data, name="SFS-{}".format(job))

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


def inference_dadi(simulation, models, path_data, job, fixed, value):
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
    # Inference
    sfs_observed, sample = simulation['SFS observed'], simulation['Parameters']['sample_size']

    inf = likelihood_ratio_test(sfs_observed, models, sample, path_data, job, dof=2,
                                fixed=fixed, value=value)

    # Save data
    params = {
        k: v for k, v in simulation['Parameters'].items() if k in ['Tau', 'Kappa', 'm12', 'm21']
    }

    # Create DataFrame from dictionary
    dico = {
        'Parameters': [params], 'Positive hit': [sum(inf['LRT'])], 'SNPs': [simulation['SNPs']],
        'SFS observed': [sfs_observed], 'M0': [inf['M0']], 'M1': [inf['M1']],
        'Time': [inf['Time']]
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
    os.remove("{}SFS-{}.fs".format(path_data, job))


######################################################################
# Inference with stairway plot 2                                     #
######################################################################

def inference_stairway_plot(simulation, model):
    """
    Inference with stairway plot.
    """
    for _, row in simulation.iterrows():
        path_data = "/home/damase/All/Cours/M2BI-Diderot/Species_evolution_inference/sei/" \
            "inference/stairway_plot_v2.1.1/"

        if model == 'decline':
            name = "stairway_{}-tau={}_kappa={}"\
                .format(model, row['Parameters']['Tau'], row['Parameters']['Kappa'])

        for sfs in row['SFS observed']:
            # Generate the SFS file compatible with stairway plot v2
            data = {k: v for k, v in row['Parameters'].items() if k in['sample_size', 'length', 'mu']}
            data['sfs'], data['year'], data['ninput'] = sfs, 1, 200

            f.stairway_data(name, data, path_data)

            # Create the batch file
            os.system("java -cp {0}stairway_plot_es Stairbuilder {0}{1}.blueprint"
                      .format(path_data, name))

            # Run the batch file
            os.system("bash {}{}.blueprint.sh".format(path_data, name))

            # Remove all blueprint file
            os.system("rm -rf {}{}.blueprint*".format(path_data, name))


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
            # inference_dadi(simulation, models, path_data, fixed=None, value=None)
            print("Over")
            sys.exit()

        # Generate length factor file
        elif args.file:
            path_data = "./Data/Msprime/{}/".format(args.model)
            filin = "SFS_{}-all.json".format(args.model)

            # Export the observed SFS to DataFrame
            simulation = f.export_simulation_files(filin, path_data)

            # present = []
            # for i, param in enumerate(simulation['Parameters']):
            #     tau = round(np.log10(param['Tau']), 2)
            #     kappa = round(np.log10(param['Kappa']), 2)
            #     present.append((tau, kappa))

            # tau_list, kappa_list = np.arange(-4, 2.5, 0.1), np.arange(-3.5, 3, 0.1)

            # absent = []
            # for tau in tau_list:
            #     t = round(tau, 2)
            #     for kappa in kappa_list:
            #         k = round(kappa, 2)
            #         if (t, k) not in present:
            #             absent.append((t, k))

            factor, theta = pd.DataFrame(columns=['Parameters', 'Factor']), 32000
            if args.model == 'decline':
                # Convert Tau & kappa to log10
                factor['Parameters'] = simulation['Parameters'].apply(lambda ele: {
                    'Tau': round(np.log10(ele['Tau']), 2),
                    'Kappa': round(np.log10(ele['Kappa']), 2)
                })

            # Compute mean SNPs
            factor['Factor'] = simulation['SNPs'].apply(lambda snp: np.mean(snp) / theta)

            # max_factor = max(factor['Factor'])
            # for val in absent:
            #     dico = {'Parameters': {'Tau': val[0], 'Kappa': val[1]},
            #             'Factor': max_factor * np.power(val[1], val[1])}
            #     factor = factor.append(dico, ignore_index=True)

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

        generate_sfs(params, model, nb_simu=2, path_data=path_data, path_length=path_length)

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
                simulation = simulation.iloc[args.job-1]
                path_data = "./Data/Dadi/{}/all/".format(args.model)

            else:
                param = args.param.capitalize()
                simulation = [
                    ele for _, ele in simulation.iterrows()
                    if round(np.log10(ele['Parameters'][param]), 2) == args.value
                ][args.job-1]
                path_data = "./Data/Dadi/{}/{}/".format(args.model, args.param)

                args.value = np.power(10, args.value)

            inference_dadi(simulation, models, path_data, args.job, fixed=args.param,
                           value=args.value)

        # Inference with stairway plot 2
        elif args.stairway:

            inference_stairway_plot(simulation, model=args.model)

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
        pass

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
