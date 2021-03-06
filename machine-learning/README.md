# machine-learning

These scripts are a work in progress and can definitely be improved. But I have temporarily taken a break on machine learning to strengthen my background in traditional computer science and cheminformatics topics. 

Here I have a couple scripts that help me simplify my machine learning workflows:
1. chembl_to_dataframe.py usage
* ```user@computer:~$ python chembl_to_dataframe.py [-h] [-q] assay_id thresh```
    * ```-h, --help: flag to show help text.```
    * ```-q, --quiet: flag to suppress printing to console.```
    * ```assay_id: argument to take the ChEMBL assay ID, looks something like *CHEMBL1614275*.```
    * ```thresh: argument for cutoff for removing duplicates entries. if standard deviation of duplicates is greater than thresh, all duplicates are thrown out. if standard deviation of duplicates is smaller than thresh, duplicates are averaged. 0.1 is a good value for log-scale labels.```
* takes a ChEMBL assay ID and saves it as a pandas pickle file.

<br>

2. split_to_dir.py usage
* ```user@computer:~$ python split_to_dir.py [-h] path_to_pickle split out_dir```
    * ```-h, --help: flag to show help text.```
    * ```path_to_pickle: argument with filepath to pandas pickle generated by chembl_to_dataframe.py.```
    * ```split: argument to choose desired type of split (random, fingerprint, butina, or scaffold)```
    * ```out_dir: name of directory containing the train/test splits created by the script.```
* creates 5-fold split of the data
    * **DATA** -> [1] [2] [3] [4] [5]

<br>

3. (model).py usage, where (model) = extreme_gradient_boost, feedforward_neural_network, or random_forest
* ```user@computer:~$ python (model).py [-h] path_to_dir feat n_trials```
    * ```-h, --help: flag to show help text.```
    * ```path_to_dir: argument with filepath to train/test splits created by split_to_dir.py.```
    * ```feat: type of featurizer for the data. currently only ECFP is supported.```
    * ```n_trials: the optuna argument used for hyperparameter optimization. 100 is an okay default but higher is better.```
* uses the 5-fold split data to create 5 distinct splits:
    * split 1: train: [1] [2] [3] [4], test: [5]
    * split 2: train: [1] [2] [3] [5], test: [4]
    * split 3: train: [1] [2] [4] [5], test: [3]
    * split 4: train: [1] [3] [4] [5], test: [2]
    * split 5: train: [2] [3] [4] [5], test: [1]
* for each split, hyperparameters are optimized. then 5 models are trained using those hyperparameters. While this takes quite a bit more work, the benefit is two-fold:
    * we can be sure we aren't seeing artificially high performance from a particular split being "easier" than the other splits. this way we can see performances on ALL splits.
    * we can measure the model performance a little more accuracy by training the model 5 independent times, giving us more confidence in the generated accuracy value. 