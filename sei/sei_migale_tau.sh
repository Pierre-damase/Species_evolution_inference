#!/bin/bash

# Nom du job
#$ -N dadi_opt_tau

# Number of separate submissions to the cluster
#$ -t 1-61

# Short pour un job < 12h
#$ -q short.q

# Adresse à envoyer
# -M pierre.imbert@college-de-france.fr

# Envoie mail - (e)nd & (a)bort
# -m ea

# Sortie standard
#$ -o $HOME/work/Out

# Sortie d'erreur
#$ -e $HOME/work/Err

conda activate sei-3.8.5
python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py lrt --param tau --value $SGE_TASK_ID
conda deactivate
