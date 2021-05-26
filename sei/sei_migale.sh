#!/bin/bash

# Nom du job
#$ -N stair_decline
# -N stair_migration
# -N stair_decline
# -N dadi_decline_all
# -N dadi_decline_tau
# -N dadi_decline_kappa
# -N dadi_migration_all
# -N dadi_migration_tau
# -N dadi_migration_kappa

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

#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -stairway --model migration --job $SGE_TASK_ID

#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model decline --job $SGE_TASK_ID
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model decline --job $SGE_TASK_ID --param tau --value 2.4
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model decline --job $SGE_TASK_ID --param kappa --value 2.4

#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model migration --job $SGE_TASK_ID
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model migration --job $SGE_TASK_ID --param m12 --value 2.4
#python /home/pimbert/work/Species_evolution_inference/sei/sei_migale.py inf -dadi --model migration --job $SGE_TASK_ID --param kappa --value 2.4

conda deactivate
