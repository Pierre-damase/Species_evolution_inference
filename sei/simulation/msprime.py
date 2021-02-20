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


def msprime_debugger(configuration_pop, history):
    debugger = msprime.DemographyDebugger(
        population_configurations=configuration_pop, demographic_events=history
    )
    debugger.print_history()


def constant_model(sample, pop, tau, kappa, debug):
    """
    Constant model, i.e. population size is constant - control scenario.

    Parameter
    ---------
    size_pop: int
        population size - constant over time (kappa of 0 or tau of 0 or +inf)

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
    history = None

    if debug:
        msprime_debugger(configuration_pop, history)

    return configuration_pop, history


def sudden_decline_model(sample, pop, tau, kappa, debug):
    """
    Sudden decline model of the population (force kappa) at a time tau in the past.

    Parameter
    ---------
    pop: int
        population size at time 0 (nowadays)

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
        msprime.PopulationParametersChange(time=tau, initial_size=pop*kappa, growth_rate=0)
    ]

    if debug:
        msprime_debugger(configuration_pop, history)

    return configuration_pop, history


def sudden_growth_model(sample, pop, tau, kappa, debug):
    """
    Sudden growth model of the population (force kappa) at a time tau in the past.

    Parameter
    ---------
    pop: int
        population size at time 0 (nowadays)

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
        msprime.PopulationParametersChange(time=tau, initial_size=pop/kappa, growth_rate=0)
    ]

    if debug:
        msprime_debugger(configuration_pop, history)

    return configuration_pop, history


def simple_migration_model(sample, pop, migration_rate, kappa, debug):
    """
    Migration model with msprime.

    Populations:
      - Population 1 of size n1 - we only choose samples from this population
      - Population 2 of size n2 with n2 = kappa * n1.

    Migration
      - There is no migration between population 1 & 2
      - There is some migrations between population 2 & 1 with tau the migration rate

    Parameter
    ---------
    tau: float
        in this model it's the migration rate between the population 2 and 1
    """
    # The list of PopulationConfiguration instances describing the sampling configuration, the
    # relative sizes and growth rates of the population to be simulated.
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0),
        msprime.PopulationConfiguration(sample_size=0, initial_size=kappa*pop, growth_rate=0)
    ]

    # The matrix describing the rates of migration between all pairs of populations.
    # It's an N*N matrix with N the number of populations defined in configuration_pop.
    # Each element of the matrix Mj,k defines the fraction of population j that consists of
    # migrants from population k in each generation.
    migration_matrix = [
        [0, 0],
        [migration_rate, 0]
    ]

    if debug:
        debugger = msprime.DemographyDebugger(
            population_configurations=configuration_pop, migration_matrix=migration_matrix
        )
        debugger.print_history()

    return configuration_pop, migration_matrix


def msprime_simulation(model, param, tau=0.0, kappa=0.0, debug=False):
    """
    Population simulation with msprime.

    Parameter
    ---------
    model: function
        (constant, sudden declin, sudden growth, etc.)
    param: dictionary
        - sample_size: the number of sampled monoploid genomes - int
        - size_population: the effective (diploid) population size - int
        - rcb_rate: the rate of recombinaison per base per generation - float
        - mu: the rate of infinite sites mutations per unit of sequence length per generation-float
        - length: the length of the simulated region in bases - float
    tau: float
        the lenght of time ago at which the event (decline, growth) occured
    kappa: float
        the growth or decline force
    debug: Boolean
        1: print msprime debugger, 0: nothing

    Return
    ------
    sfs: list
        Site frequency Spectrum (sfs) - allele mutation frequency
    """
    demography = model(param["sample_size"], param["size_population"], tau, kappa, debug)

    tree_seq = msprime.simulate(
        length=param["length"], recombination_rate=param["rcb_rate"], mutation_rate=param["mu"],
        population_configurations=demography[0], demographic_events=demography[1]
    )

    sfs = [0] * (param["sample_size"] - 1)
    for variant in tree_seq.variants():
        _, counts = np.unique(variant.genotypes, return_counts=True)
        freq_mutation = counts[1]-1
        sfs[freq_mutation] += 1

    return sfs


def msprime_migration_simulation(model, param, migration_rate=0.0, kappa=0.0, debug=False):
    """

    """
    demography = model(
        param["sample_size"], param["size_population"], migration_rate, kappa, debug
    )

    tree_seq = msprime.simulate(
        length=param["length"], recombination_rate=param["rcb_rate"], mutation_rate=param["mu"],
        population_configurations=demography[0], migration_matrix=demography[1]
    )

    sfs = [0] * (param["sample_size"] - 1)
    for variant in tree_seq.variants():
        _, counts = np.unique(variant.genotypes, return_counts=True)
        freq_mutation = counts[1]-1
        sfs[freq_mutation] += 1

    return sfs


if __name__ == "__main__":
    sys.exit()  # No actions desired
