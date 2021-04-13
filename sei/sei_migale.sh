#!/bin/bash

# Nom du job
#$ -N stairway

# Number of separate submissions to the cluster
#$ -t 1-2

# Short pour un job < 12h
#$ -q long.q

# Adresse à envoyer
# -M pierre.imbert@college-de-france.fr

# Envoie mail - (b)egin, (e)nd, (a)bort & (s)uspend
# -m as

# Sortie standard
#$ -o $HOME/work/Out

# Sortie d'erreur
#$ -e $HOME/work/Err

conda activate sei-3.8.5
python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -stairway --model decline --job $SGE_TASK_ID
conda deactivate
