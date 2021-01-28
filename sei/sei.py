"""
Programme pour inférer l'évolution d'une population à partir de données génomiques.
"""

import sei.simulation.scenario as sce

from sei.graphiques import plot

def main():
    """
    Le main du programme.
    """
    parametres = {
        "sample_size": 6, "size_population": 1, "rcb_rate": 2e-3, "mu": 2e-3, "length": 1e5
    }

    # Constant
    print("Scénario constant")
    sfs_cst = \
        sce.simulation(model=sce.constant_model, param=parametres, tau=0.0, kappa=0.0)

    # Declin
    print("\n\nScénario de déclin")
    sfs_declin = \
        sce.simulation(model=sce.sharp_decline_model, param=parametres, tau=1.5, kappa=4)

    # Croissance
    print("\n\nScénario de croissance")
    sfs_croissance = \
        sce.simulation(model=sce.sharp_increase_model, param=parametres, tau=1.5, kappa=4)

    plot.plot_sfs(parametres["sample_size"], sfs_cst, sfs_declin, sfs_croissance)


if __name__ == "__main__":
    main()
