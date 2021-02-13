#!/bin/bash

# /home/pimbert/work/Species_evolution_inference/sei/

conda activate sei-3.8.5 && python sei_migale.py opt --nb 10 && conda deactivate
conda activate sei-3.8.5 && python sei_migale.py opt --nb 20 && conda deactivate
conda activate sei-3.8.5 && python sei_migale.py opt --nb 40 && conda deactivate
conda activate sei-3.8.5 && python sei_migale.py opt --nb 60 && conda deactivate
conda activate sei-3.8.5 && python sei_migale.py opt --nb 100 && conda deactivate
