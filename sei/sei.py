"""
Programme pour inférer l'évolution d'une population à partir de données génomiques.
"""

import sei.files.files as f
import sei.graphics.plot as plot
import sei.inference.dadi as dadi
import sei.simulation.msprime as ms


def sfs_verification():
    """
    Method to check the SFS obtained with msprime.

    I.E. check that:
     - The SFS of a constant population fits well to the theoretical SFS of any constant population
     - The SFS of an increasing or decreasing population
    """
    parametres = {
        "sample_size": 6, "size_population": 1, "rcb_rate": 2e-3, "mu": 2e-3, "length": 1e5
    }

    # Constant
    print("Scénario constant")
    sfs_cst = \
        ms.msprime_simulation(model=ms.constant_model, param=parametres, tau=0.0, kappa=0.0)

    # Declin
    print("\n\nScénario de déclin")
    sfs_declin = \
        ms.msprime_simulation(model=ms.sudden_decline_model, param=parametres, tau=1.5, kappa=4)

    # Growth
    print("\n\nScénario de croissance")
    sfs_croissance = \
        ms.msprime_simulation(model=ms.sudden_growth_model, param=parametres, tau=1.5, kappa=4)

    # Theoretical SFS for any constant population
    sfs_theorique = [0] * (parametres["sample_size"] - 1)
    for i in range(len(sfs_theorique)):
        sfs_theorique[i] = 1 / (i+1)

    # Plot
    plot.plot_sfs(
        sfs=[sfs_cst, sfs_declin, sfs_croissance, sfs_theorique],
        label=["Constant", "Declin", "Growth", "Theoretical"],
        color=["blue", "red", "green", "orange"],
        title="Unfold SFS for various scenarios"
    )


def main():
    """
    Le main du programme.
    """
    #sfs_verification()

    parametres = {
        "sample_size": 6, "size_population": 1, "rcb_rate": 2e-3, "mu": 2e-3, "length": 1e5
    }
    sfs_cst = \
        ms.msprime_simulation(model=ms.constant_model, param=parametres, tau=0.0, kappa=0.0)

    f.dadi_data(sfs_cst)
    #dadi.dadi_inference()
    dadi.main()



if __name__ == "__main__":
    main()
