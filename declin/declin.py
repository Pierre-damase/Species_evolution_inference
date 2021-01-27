"""
Programme pour estimer le déclin d'une population à partir de données génomiques.
"""

import msprime


def sharp_decline_model(size_pop, tau, kappa):
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
        msprime.PopulationConfiguration(sample_size=6, initial_size=size_pop, growth_rate=0)
    ]
    history = [
        msprime.PopulationParametersChange(time=tau, initial_size=size_pop*kappa, growth_rate=0)
    ]

    debugger = msprime.DemographyDebugger(population_configurations=configuration_pop,
                                          demographic_events=history)
    debugger.print_history()

    return configuration_pop, history


def simulation(model, tau, kappa):
    """
    Simulation d'une population.

    Parameter
    ---------
    tau: float
        la date de la modification de la taille de la population
    kappa: float
        la force de la modification
    """
    print("Scénario pour un tau={} et kappa={}".format(tau, kappa), end="\n\n")

    size_pop = 100 # 1 + mu à 1 - 4.Ne.mu vaut 4 soit 4 mutations par base

    param = model(size_pop, tau, kappa)
    print(param)

    tree_seq = msprime.simulate(
        length=1e5, recombination_rate=2e-8, mutation_rate=2e-8, random_seed=2,
        population_configurations=param[0], demographic_events=param[1]
    )

    # print(tree_seq.tables.nodes)
    print(tree_seq.first().draw(format="unicode"))

    variants = []
    for variant in tree_seq.variants():
        variants.append(variant.genotypes)

    # Equivalent pour récupérer les variants
    # tree_seq.genotype_matrix()


def main():
    """
    Le main du programme.
    """

    # Add boucle for - tau & kappa
    simulation(model=sharp_decline_model, tau=500, kappa=10)


if __name__ == "__main__":
    main()
