#!/usr/bin/env python

import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

import pandas as pd
import numpy as np
import argparse

def load_split_dfs(input_dir):
    test = []
    for i in range(5):
        test.append(pd.read_pickle(f'{input_dir}/test{i}'))
    return test

def ds_from_df_split(split_dfs, featurizer):
    split_dss = []
    for i in range(5):
        df = split_dfs[i]
        X = featurizer.featurize(df.smiles)
        ds = dc.data.DiskDataset.from_numpy(X=X, y=np.vstack(df.label.to_numpy()), ids=df.smiles)
        split_dss.append(ds)
    all_dss = dc.data.DiskDataset.merge(split_dss)
    
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=all_dss)
    for i in range(5):
        split_dss[i] = transformer.transform(split_dss[i])
    
    return all_dss, split_dss, transformer

def get_kfold_from_ds_split(split_dss):
    kfold = []
    for i in range(5):
        temp_dss = split_dss.copy()
        temp_test = temp_dss.pop(i)
        kfold.append((dc.data.DiskDataset.merge(temp_dss), temp_test))
    return kfold

def random_forest_model_from_trial(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 500, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 1000, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1000, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 100, log=True),
    }
    sklearn_random_forest = RandomForestRegressor(**param, n_jobs=-1)
    rf_model = dc.models.SklearnModel(sklearn_random_forest)
    return rf_model

def random_forest_model_from_param(param):
    sklearn_random_forest = RandomForestRegressor(**param, n_jobs=-1)
    rf_model = dc.models.SklearnModel(sklearn_random_forest)
    return rf_model

def random_forest_optuna(trial, kfold):
    mse = []
    for k in kfold:
        rf_model = random_forest_model_from_trial(trial)
        rf_model.fit(k[0].complete_shuffle())
        y_pred = rf_model.predict(k[1])
        y_meas = k[1].y
        mse.append(dc.metrics.mean_squared_error(y_meas, y_pred))
        
    return sum(mse)/len(mse)

def main():
    parser = argparse.ArgumentParser(description='takes a directory containing data split with split_to_dir.py and trains a hyperparameter-optimized random forest model')
    parser.add_argument('path_to_dir', help='the path and directory to the split data')
    parser.add_argument('feat', help='the featurizer to use, options are ECFP, ECFP-Mordred, Mol2Vec, or Mol2Vec-Mordred')
    parser.add_argument('n_trials', help='the number of trials used for hyperparameter optimization')
    args = parser.parse_args()
        
    split_dfs = load_split_dfs(args.path_to_dir)
    
    if args.feat == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(radius=2, size=2048, chiral=True)
    else:
        print('not a valid featurizer')
        quit()
        
    all_dss, split_dss, transformer = ds_from_df_split(split_dfs, featurizer)
    train_tests = get_kfold_from_ds_split(split_dss)
    
    output_info = []
    for i,tt in enumerate(train_tests):
        splitter = dc.splits.RandomSplitter()
        kfold = splitter.k_fold_split(dataset=tt[0], k=5)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: random_forest_optuna(trial, kfold), n_trials=int(args.n_trials))

        test_mse = []
        for j in range(5):
            tuned_rf_model = random_forest_model_from_param(study.best_params)
            tuned_rf_model.fit(tt[0].complete_shuffle()) 
            y_pred = tuned_rf_model.predict(tt[1])
            y_meas = tt[1].y
            test_mse.append(dc.metrics.mean_squared_error(y_meas, y_pred))
        
        output_info.append((i, study.best_value, str(study.best_params), sum(test_mse)/len(test_mse), test_mse))
        print(f'completed split {i} out of 5')
        
    out_df = pd.DataFrame(output_info, columns=['split_index', 'avg_valid_mse', 'best_params', 'avg_test_mse', 'test_mses'])
    out_df.to_csv(path_or_buf=f'{args.path_to_dir}/rf.csv')

if __name__ == '__main__':
    main()