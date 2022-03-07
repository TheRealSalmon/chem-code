# chem-scripts

Hi, this is where I keep my Python scripts related to cheminformatics and molecular machine learning.

In my **ml-scripts** directory you'll find scripts that evaluate the accuracy of a particular model on ChEMBL data. The typical workflow is illustrated in Jupyter notebooks but is outlined below:
* Load ChEMBL data into a Pandas dataframe
* Clean the data and prepare it for featurization
* Split the data 
* Train models with hyperparameter optimization for an evaluation.

One note on my implementation is that for every split, I create 5 distinct 80/20 train/test splits. Then for each distinct split, I independently perform quick hyperparameter optimization and model evaluation. I do this because I worry about the particular scaffold/fingerprint/cluster split being easier than typical and throwing off results. 

In my **cheminf-scripts** directory you'll find scripts, modules, and notebooks related to more traditional cheminformatics topics. 

[Describe some highlights!]