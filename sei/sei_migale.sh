#!/bin/bash

# Nom du job
#$ -N dadi_migr_all
# -N dadi_migr_tau
# -N dadi_migr_kappa
# -N smc_decline
# -N smc_growth
# -N smc_cst

# Number of separate submissions to the cluster
#$ -t 1-2

# Short pour un job < 12h
#$ -q short.q

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

#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model migration --job $SGE_TASK_ID
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model migration --job $SGE_TASK_ID --param tau --value 0.0
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model migration --job $SGE_TASK_ID --param kappa --value 0.0

#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py optsmc --model decline --data
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py optsmc --model growth --data
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py optsmc --model cst --data

conda deactivate
