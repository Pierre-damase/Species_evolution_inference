"""
Ce module permet de générer différents scénarios démographiques à partir de simulateur.

Il n'est utilisé que msprime à l'heure actuelle.

Les scénarios mis en place sont:
  -      Population constante: scénario de contrôle
  - Population déclin brutale: scénario où un déclin de force kappa est observé à un temps tau
"""

import sys
import msprime
import numpy as np


def constant_model(sample, pop, tau, kappa):
    """
    Modèle de population constante - scénario de contrôle.

    Parameter
    ---------
    size_pop: int
        la taille de la population - constant au cours du temps

    Return
    ------
    configuration_pop: list
        la configuration de la population constante
    history: None
        population constante donc pas d'history
    """
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history = None

    debugger = msprime.DemographyDebugger(population_configurations=configuration_pop,
                                          demographic_events=history)
    debugger.print_history()

    return configuration_pop, history


def sharp_decline_model(sample, pop, tau, kappa):
    """
    Modèle de déclin brutale (de force kappa) de la population à un temps tau.

    Parameter
    ---------
    size_pop: int
        la taille de la population au temps 0

    Return
    ------
    configuration_pop: list
        la configuration initiale de la population, i.e. au temps 0
    history: list
        changement démographique observé dans la population à un temps tau, ici déclin de force
        kappa
    """
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history = [
        msprime.PopulationParametersChange(time=tau, initial_size=pop*kappa, growth_rate=0)
    ]

    debugger = msprime.DemographyDebugger(population_configurations=configuration_pop,
                                          demographic_events=history)
    debugger.print_history()

    return configuration_pop, history


def sharp_increase_model(sample, pop, tau, kappa):
    """
    Modèle de croissance brutale (de force kappa) de la population à un temps tau.

    Parameter
    ---------
    size_pop: int
        la taille de la population au temps 0

    Return
    ------
    configuration_pop: list
        la configuration initiale de la population, i.e. au temps 0
    history: list
        changement démographique observé dans la population à un temps tau, ici croissance de
        force kappa
    """
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history = [
        msprime.PopulationParametersChange(time=tau, initial_size=pop/kappa, growth_rate=0)
    ]

    debugger = msprime.DemographyDebugger(population_configurations=configuration_pop,
                                          demographic_events=history)
    debugger.print_history()

    return configuration_pop, history


def simulation(model, param, tau, kappa):
    """
    Simulation d'une population.

    Parameter
    ---------
    model: function
        le scénario à appliquer (constant, déclin brutal, etc.)
    param: dictionary
        - sample_size: le nombre de génomes échantillonés - int
        - size_population: la taille de la population - int
        - rcb_rate: le taux de recombinaison - float
        - mu: le taux de mutation
        - length: la taille des séquences simulées
    tau: float
        la date de la modification de la taille de la population
    kappa: float
        la force de la modification

    Return
    ------
    sfs: list
        fréquence de mutations des allèles
    """
    demography = model(param["sample_size"], param["size_population"], tau, kappa)

    tree_seq = msprime.simulate(
        length=param["length"], recombination_rate=param["rcb_rate"], mutation_rate=param["mu"],
        random_seed=2,
        population_configurations=demography[0], demographic_events=demography[1]
    )

    sfs = [0] * (param["sample_size"] - 1)
    for variant in tree_seq.variants():
        _, counts = np.unique(variant.genotypes, return_counts=True)
        freq_mutation = counts[1]-1
        sfs[freq_mutation] += 1

    somme = sum(sfs)

    return [ele / somme for ele in sfs]


if __name__ == "__main__":
    sys.exit()  # aucunes actions souhaitées
