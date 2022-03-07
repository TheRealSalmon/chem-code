#!/usr/bin/env python

import deepchem as dc
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
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

def feedforward_neural_network_from_trial(trial, len_feat_vec):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)
    fnn_model = tf.keras.Sequential()
    fnn_model.add(tf.keras.layers.Input(shape=(len_feat_vec,)))
    # dropout = trial.suggest_float(f'dropout_{0}', 0.0, 1.0)
    # model.add(tf.keras.layers.Dropout(dropout))
    for i in range(n_layers):
        num_nodes = trial.suggest_int(f'n_nodes_{i+1}', 4, 128, log=True)
        dropout = trial.suggest_float(f'dropout_{i+1}', 0.0, 1.0)
        fnn_model.add(
            tf.keras.layers.Dense(
                num_nodes,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            )
        )
        fnn_model.add(tf.keras.layers.Dropout(dropout))
    fnn_model.add(
        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    )
    rate = trial.suggest_float('learning_rate', 1e-8, 1e-1, log=True)
    dc_model = dc.models.KerasModel(model=fnn_model, loss=dc.models.losses.L2Loss(), learning_rate=rate)
    return dc_model

def feedforward_neural_network_optuna(trial, kfold, transformer, len_feat_vec):
    mse = []
    
    for i,k in enumerate(kfold):
        fnn_model = feedforward_neural_network_from_trial(trial, len_feat_vec)
        metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
        callback = dc.models.callbacks.ValidationCallback(dataset=k[1], interval=100, metrics=[metric], save_dir='tmp')
        fnn_model.fit(dataset=k[0].complete_shuffle(), nb_epoch=100, callbacks=callback)
        mse.append(callback._best_score)
        
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
        study.optimize(lambda trial: feedforward_neural_network_optuna(trial, kfold, transformer, len_feat_vec), n_trials=int(args.n_trials))

        test_mse = []
        for j in range(5):
            tuned_fnn_model = feedforward_neural_network_from_trial(study.best_trial, len_feat_vec)
            metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
            callback = dc.models.callbacks.ValidationCallback(dataset=tt[1], interval=100, metrics=[metric], save_dir='tmp')
            tuned_fnn_model.fit(dataset=tt[0].complete_shuffle(), nb_epoch=100, callbacks=callback)
            test_mse.append(callback._best_score)
        
        output_info.append((i, study.best_value, str(study.best_params), sum(test_mse)/len(test_mse), test_mse))
        print(f'completed split {i} out of 5')
        
    out_df = pd.DataFrame(output_info, columns=['split_index', 'avg_valid_mse', 'best_params', 'avg_test_mse', 'test_mses'])
    out_df.to_csv(path_or_buf=f'{args.path_to_dir}/fnn.csv')

if __name__ == '__main__':
    main()