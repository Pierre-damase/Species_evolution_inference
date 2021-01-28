"""
Ce module permet de réaliser des graphiques.
"""

import matplotlib.pyplot as plt
import sys


def plot_sfs(sample_size, *args):
    """
    Plot un Site Frequency Spectrum - SFS.
    """
    # Générer modèle théorique attendu pour une population constante
    theorique = [0] * (sample_size - 1)
    for i in range(len(theorique)):
        theorique[i] = 1 / (i+1)
    somme = sum(theorique)
    theorique = [ele / somme for ele in theorique]

    # Graphe
    label = ["Constant", "Déclin", "Croissance"]
    color = ["blue", "red", "green"]

    for i in range(3):
        plt.plot(args[i], color=color[i], label=label[i])
    plt.plot(theorique, color="orange", label="Théorique")

    # Ajouter la légende au dessus du plot sans changer sa taille
    plt.legend(loc="upper right", fontsize="large")

    title = "Site Frequency Spectrum pour différents scénarios"
    plt.title(title, fontsize="xx-large")

    plt.savefig("./Figures/sfs")
    plt.clf()



if __name__ == "__main__":
    sys.exit()  # AUcunes actions souhaitées
