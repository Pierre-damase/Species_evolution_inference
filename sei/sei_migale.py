"""
Migale version.
"""

import time
import pandas as pd
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


def dadi_params_optimisation():
    """
    Determine the error rate of the inference of 100 observed - simulated with msprime.

    Each observed is a constant population model. The goal is to determine the best mutation
    rate mu and the best number of sampled genomes n.

      - mu: the mutation rate
      - n: the number of sampled monoploid genomes
    """
    sample_list, mu_list = [10, 20, 40, 60, 100], [2e-3, 4e-3, 8e-3, 12e-3, 2e-2,  8e-2,  2e-1]
    nb_simu = 3
    dico = {}

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
                f.dadi_data(sfs_cst, dadi.constant_model.__name__, path="../Data/")

                # Dadi inference
                _, estimated_theta = dadi.dadi_inference(pts_list, dadi.constant_model,
                                                         path="../Data/")

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

    plot.plot_error_rate(dico, path="../Figures/Error_rate/")


if __name__ == "__main__":
    dadi_params_optimisation()
