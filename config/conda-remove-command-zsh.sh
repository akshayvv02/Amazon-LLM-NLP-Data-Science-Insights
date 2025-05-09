#!/bin/zsh
# Usage:
# 1. Make executable: chmod +x conda-remove-command.sh
# 2. Run: ./conda-remove-command-zsh.sh

# Activate base env
conda init zsh
source ~/.zshrc
conda deactivate

# Remove the conda environment named 'nlp-env'
conda remove -n nlp-env --all -y

echo "Environment 'nlp-env' removed successfully."