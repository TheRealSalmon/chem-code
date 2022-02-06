#!/usr/bin/env python

import deepchem as dc
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
import os
import argparse

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def try_standardize(smi):
    try:
        return StandardizeSmiles(Chem.MolToSmiles(neutralize_atoms(Chem.MolFromSmiles(smi))))
    except:
        pass
    return 'fail'

def clean_up_smiles(df):
    df['smiles'] = df['smiles'].map(lambda s: try_standardize(s))
    df = df[df.smiles != 'fail']
    return df

def df_to_ds(df):
    X = np.zeros(shape=(len(df),1))
    return dc.data.DiskDataset.from_numpy(X=X, y=np.vstack(df.label.to_numpy()), ids=df.smiles)

def main():
    parser = argparse.ArgumentParser(description='meep')
    parser.add_argument('path_to_pickle', help='the path and filename of the pd.Dataframe pickle')
    parser.add_argument('split', help='the dc.splits splitter used to create the test set, valid options are random, fingerprint, butina, or scaffold')
    parser.add_argument('out_dir', help='the output directory containing the pd.Dataframe pickles for the train/test splits')
    # parser.add_argument('-q', '--quiet', help='this flag suppresses any printing to the console', action='store_true')
    args = parser.parse_args()
    
    # first we read the pd.Dataframe containing the raw data
    # then we standardize and neutralize the SMILES strings discarding any that fail the process
    # finally we convert our dataframe to a dc.data.DiskDataset to apply DeepChem splitters 
    df = pd.read_pickle(args.path_to_pickle)
    df = clean_up_smiles(df)
    ds = df_to_ds(df)
    
    # here we select our splitter based on user input
    splitter = None
    split_type = ''
    if args.split == 'random':
        splitter = dc.splits.RandomSplitter()
        split_type = 'random'
    elif args.split == 'fingerprint':
        splitter = dc.splits.FingerprintSplitter()
        split_type = 'finger'
    elif args.split == 'scaffold':
        splitter = dc.splits.ScaffoldSplitter()
        split_type = 'scafld'
    elif args.split == 'butina':
        splitter = dc.splits.ButinaSplitter()
        split_type = 'butina'
    else:
        print('that is not a valid splitter')
        quit()
    
    # here we split our DiskDataset into 5 train/test splits
    train_test_splits = splitter.k_fold_split(dataset=ds, k=5)
    
    # we create the output directory
    out_dir_name = f'{args.out_dir}_{split_type}_split'
    os.mkdir(path=out_dir_name)
    
    # finally we convert our DiskDataset back into Dataframes and pickle them 
    for i,tt in enumerate(train_test_splits):
        train_df = tt[0].to_dataframe()
        train_df = train_df.rename({'ids': 'smiles', 'y': 'label'}, axis='columns').drop(['X', 'w'], axis='columns')
        train_df.to_pickle(f'{out_dir_name}/train{i}')

        test_df = tt[1].to_dataframe()
        test_df = test_df.rename({'ids': 'smiles', 'y': 'label'}, axis='columns').drop(['X', 'w'], axis='columns')
        test_df.to_pickle(f'{out_dir_name}/test{i}')
    
if __name__ == '__main__':
    main()
    
def get_df_from_splits(input_dir):
    test = []
    for i in range(5):
        test.append((pd.read_pickle(f'{input_dir}/train{i}'), pd.read_pickle(f'{input_dir}/test{i}')))
    return test