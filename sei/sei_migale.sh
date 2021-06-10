#!/bin/bash

# Nom du job
#$ -N opt_decline
# -N opt_growth
# -N opt_cst
# -N smc_data

# Number of separate submissions to the cluster
#$ -t 1-11
# -t 1-961

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
python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py optsmc --model decline --job $SGE_TASK_ID
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py optsmc --model growth --job $SGE_TASK_ID
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py optsmc --model cst --job $SGE_TASK_ID

#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py data --model decline --job $SGE_TASK_ID --typ vcf

conda deactivate
