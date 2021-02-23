"""
This module allows the generation of different demographic scenarios from simulator msprime.

Various scenarios are set up:
  -       Constant model: control scenario
  - Sudden decline model: decline of force kappa at a time tau
  -  Sudden growth model: growth of force kappa at a time tau
"""

import sys
import numpy as np
import msprime


def msprime_debugger(configuration_pop, history, migration_matrix):
    debugger = msprime.DemographyDebugger(
        population_configurations=configuration_pop, demographic_events=history,
        migration_matrix=migration_matrix
    )
    debugger.print_history()


def constant_model(sample, pop, params, debug):
    """
    Constant model, i.e. population size is constant - control scenario.

    Parameter
    ---------
    params: None

    Return
    ------
    configuration_pop: list
        the configuration of the constant population - size, growth.
    history: None
        constant model so no history
    """
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history, migration_matrix = None, None

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def sudden_decline_model(sample, pop, params, debug):
    """
    Sudden decline model of the population (force kappa) at a time tau in the past.

    Parameter
    ---------
    params: dictionary
      - Tau: the lenght of time ago at which the event (decline, growth) occured
      - Kappa: the growth or decline force

    Return
    ------
    configuration_pop: list
        the initial configuration of the population, i.e. at time 0
    history: list
        the observed demographic change in the population at tau time - decline of force kappa
    """
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history = [
        msprime.PopulationParametersChange(time=params["Tau"], initial_size=pop*params["Kappa"],
                                           growth_rate=0)
    ]
    migration_matrix = None

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def sudden_growth_model(sample, pop, params, debug):
    """
    Sudden growth model of the population (force kappa) at a time tau in the past.

    Parameter
    ---------
    params: dictionary
      - Tau: the lenght of time ago at which the event (decline, growth) occured
      - Kappa: the growth or decline force

    Return
    ------
    configuration_pop: list
        the initial configuration of the population, i.e. at time 0
    history: list
        the observed demographic change in the population at tau time - growth of force kappa
    """
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history = [
        msprime.PopulationParametersChange(time=params["Tau"], initial_size=pop/params["Kappa"],
                                           growth_rate=0)
    ]
    migration_matrix = None

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def two_pops_migration_model(sample, pop, params, debug):
    """
    Migration model with msprime.

    Populations:
      - Population 1 of size n1 - we only choose samples from this population
      - Population 2 of size n2 with n2 = kappa * n1.

    Migration
      - There is no migration from population 1 to 2
      - There is some migrations from population 2 to 1

    Parameter
    ---------
    params: dictionary
      - Kappa: ratio of population's 1 size to population's 2 size
      - m12: migration rate from population 2 to 1
      - m21: migration rate from population 1 to 2
    """
    # The list of PopulationConfiguration instances describing the sampling configuration, the
    # relative sizes and growth rates of the population to be simulated.
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0),
        msprime.PopulationConfiguration(sample_size=0, initial_size=params["Kappa"]*pop,
                                        growth_rate=0)
    ]
    history = None

    # The matrix describing the rates of migration between all pairs of populations.
    # It's an N*N matrix with N the number of populations defined in configuration_pop.
    # Each element of the matrix Mj,k defines the fraction of population j that consists of
    # migrants from population k in each generation.
    migration_matrix = [
        [0, params["m21"]],
        [params["m12"], 0]
    ]

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def msprime_simulation(model, fixed_params, params=None, debug=False):
    """
    Population simulation with msprime.

    Parameter
    ---------
    model: function
        (constant, sudden declin, sudden growth, etc.)
    fixed_params: dictionary
        - sample_size: the number of sampled monoploid genomes
        - size_population: the effective (diploid) population size
        - rcb_rate: the rate of recombinaison per base per generation
        - mu: the rate of infinite sites mutations per unit of sequence length per generation
        - length: the length of the simulated region in bases
    params:
        varie selon le modèle
    tau: the lenght of time ago at which the event (decline, growth) occured
    kappa: the growth or decline force
    debug: Boolean
        1: print msprime debugger, 0: nothing

    Return
    ------
    sfs: list
        Site frequency Spectrum (sfs) - allele mutation frequency
    """
    demography = model(
        fixed_params["sample_size"], fixed_params["size_population"], params, debug
    )

    tree_seq = msprime.simulate(
        length=fixed_params["length"], recombination_rate=fixed_params["rcb_rate"],
        mutation_rate=fixed_params["mu"], population_configurations=demography[0],
        demographic_events=demography[1], migration_matrix=demography[2]
    )

    sfs = [0] * (fixed_params["sample_size"] - 1)
    for variant in tree_seq.variants():
        _, counts = np.unique(variant.genotypes, return_counts=True)
        freq_mutation = counts[1]-1
        sfs[freq_mutation] += 1

    return sfs


if __name__ == "__main__":
    sys.exit()  # No actions desired
