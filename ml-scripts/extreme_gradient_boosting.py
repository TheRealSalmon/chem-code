#!/usr/bin/env python

import deepchem as dc
import xgboost as xgb
xgb.set_config(verbosity=0)
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

def extreme_gradient_booster_from_trial(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 9, step=2),
        'learning_rate': trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        # 'max_delta_step': 
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }
    xgboost_regressor = xgb.XGBRegressor(**param, n_estimators=100, objective='reg:squarederror', tree_method='exact', n_jobs=-1)
    xgb_model = dc.models.GBDTModel(model=xgboost_regressor, early_stopping_rounds=20, eval_metric='rmse')
    return xgb_model

def extreme_gradient_booster_optuna(trial, kfold):
    mse = []
    for k in kfold:
        xgb_model = extreme_gradient_booster_from_trial(trial)
        xgb_model.fit_with_eval(k[0].complete_shuffle(), k[1])
        y_pred = xgb_model.predict(k[1])
        y_meas = k[1].y
        mse.append(dc.metrics.mean_squared_error(y_meas, y_pred))
    return sum(mse)/len(mse)

def main():
    parser = argparse.ArgumentParser(description='takes a directory containing data split with split_to_dir.py and trains a hyperparameter-optimized feedforward neural network model')
    parser.add_argument('path_to_dir', help='the path and directory to the split data')
    parser.add_argument('feat', help='the featurizer to use, options are ECFP, ECFP-Mordred, Mol2Vec, or Mol2Vec-Mordred')
    parser.add_argument('n_trials', help='the number of trials used for hyperparameter optimization')
    args = parser.parse_args()
        
    split_dfs = load_split_dfs(args.path_to_dir)
    
    len_feat_vec = 0
    if args.feat == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(radius=2, size=2048, chiral=True)
        len_feat_vec = 2048
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
        study.optimize(lambda trial: extreme_gradient_booster_optuna(trial, kfold), n_trials=int(args.n_trials))

        test_mse = []
        for j in range(5):
            tuned_xgb_model = extreme_gradient_booster_from_trial(study.best_trial)
            tuned_xgb_model.fit_with_eval(tt[0].complete_shuffle(), tt[1])
            y_pred = tuned_xgb_model.predict(tt[1])
            y_meas = tt[1].y
            test_mse.append(dc.metrics.mean_squared_error(y_meas, y_pred))
        
        output_info.append((i, study.best_value, str(study.best_params), sum(test_mse)/len(test_mse), test_mse))
        print(f'completed split {i} out of 5')
        
    out_df = pd.DataFrame(output_info, columns=['split_index', 'avg_valid_mse', 'best_params', 'avg_test_mse', 'test_mses'])
    out_df.to_csv(path_or_buf=f'{args.path_to_dir}/xgb.csv')

if __name__ == '__main__':
    main()