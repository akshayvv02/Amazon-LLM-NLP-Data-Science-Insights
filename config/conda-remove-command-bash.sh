#!/bin/zsh
# Usage:
# 1. Make executable: chmod +x conda-remove-command-bash.sh
# 2. Run: ./conda-remove-command-bash.sh

# Activate base env
conda init bash
source ~/.bashrc # or ~./bash_profile
conda deactivate

# Remove the conda environment named 'nlp-env'
conda remove -n nlp-env --all -y

echo "Environment 'nlp-env' removed successfully."