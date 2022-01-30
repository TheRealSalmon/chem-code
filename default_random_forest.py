#!/usr/bin/env python

import pandas as pd
import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestRegressor
import optuna

def disk_dataset_from_pickle(path_to_pickle, featurizer):
    df = pd.read_pickle(path_to_pickle)
    smiles = df.smiles.to_numpy()
    labels = np.vstack(df.label.to_numpy())

    features = featurizer.featurize(smiles)
    dataset = dc.data.DiskDataset.from_numpy(X=features, y=labels, ids=smiles)

    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=dataset)
    dataset = transformer.transform(dataset)

    return dataset, transformer

def split_dataset(dataset, n_kfold, splitter):
    train, test = splitter.train_test_split(dataset, frac_train=0.8)
    kfold = splitter.k_fold_split(train, k=n_kfold)

    return kfold, test, train
()
def objective(trial, kfold, transformer):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 500, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 1000, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1000, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 100, log=True),
    }

    sklearn_random_forest = RandomForestRegressor(**param, n_jobs=-1)
    rf = dc.models.SklearnModel(sklearn_random_forest)

    mse = []
    for k in kfold:
        rf.fit(k[0])
        y_pred = rf.predict(k[1], transformers=[transformer])
        y_meas = transformer.untransform(k[1].y)
        mse.append(dc.metrics.mean_squared_error(y_meas, y_pred))

    return sum(mse)/len(mse)

def main(n_trials):
    ft = dc.feat.CircularFingerprint(radius=2, size=2048)
    ds, tf = disk_dataset_from_pickle('small_dataset1_CHEMBL1963937.pkl', ft)

    sp = dc.splits.RandomSplitter()
    kfold, test, train = split_dataset(ds, 5, sp)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, kfold, tf), n_trials=n_trials)

    sklearn_random_forest = RandomForestRegressor(**study.best_params, n_jobs=-1)
    rf = dc.models.SklearnModel(sklearn_random_forest, model_dir='.')
    rf.fit(train)

    y_pred = rf.predict(test, transformers=[tf])
    y_meas = tf.untransform(test.y)
    rf.save()

    return dc.metrics.mean_squared_error(y_meas, y_pred)

if __name__ == '__main__':
    print(main(1))
