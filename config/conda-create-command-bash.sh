#!/bin/bash
# Update base conda
conda update -n base conda -y
conda install -y -n base conda-libmamba-solver
conda update -y -n base conda-libmamba-solver
conda config --set solver libmamba

# Create a new conda environment named 'nlp_project' with Python 3.8
conda create -n nlp-env -y python=3.11  pandas numpy matplotlib seaborn nltk gensim pyldavis  scikit-learn imbalanced-learn missingno seaborn jupyter notebook  #tabulate pydantic faiss-cpu plotly 

# Activate the new environment
conda init bash
source activate nasa-env


# Pip install custom package
pip install dojo-ds

# Install the required packages
# conda install -y pandas numpy matplotlib seaborn nltk gensim pyldavis jupyter scikit-learn

# Install the kenrel in jupyter
python -m ipykernel install --user --name=nasa-env 

# Additional installations if required
# conda install -y <other-packages>
# pip install <other-pip-packages>

# Deactivate the environment
conda deactivate

echo "Environment 'nasa-env' created and packages installed successfully."