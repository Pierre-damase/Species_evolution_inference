#!/bin/bash

# Nom du job
#$ -N DADI_OPTIMISATION

# Short pour un job < 12h
#$ -q short.q

# Envoie mail - (e)nd & (a)bort
# - m ea

# Adresse à envoyer
# -M pierre.imbert@college-de-france.fr

# Sortie standard
#$ -o $HOME/work/Out

# Sortie d'erreur
#$ -e $HOME/work/Err

# Lance la commande depuis le répertoire où est lancé le script
#$ -cwd

