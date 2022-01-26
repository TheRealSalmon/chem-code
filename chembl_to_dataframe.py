#!/usr/bin/env python

from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
import time
import argparse

def chembl_to_dataframe(assay_chembl_id, thresh, quiet):
    """
    Fetches assay data from ChEMBL and returns a pandas Dataframe containing the canonical
    SMILES and the ChEMBL value.

    Parameters
    ----------
    assay_chembl_id : str
        The assay ID of interest. Looks like CHEMBL#######.

    Returns
    -------
    pandas.Dataframe
        The canonical SMILES string and log potency (pKi, pIC50, etc.) for each entry.
    """
    start_time = time.time()
    if not quiet:
        print("""|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
|||                                                             |||
|||                                                             |||
|||            Welcome to Sam's ChEMBL Data Fetcher             |||
|||                                                             |||
|||                                                             |||
|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        """)
    # canonical_smiles contains the SMILES string that specifies the structure of the molecule
    # standard_type contains the type of data ie Ki, IC50, LogD, etc
    # standard_value contains the measured value of the assay
    # standard_units contains the units of measure, typically something like nM for potencies
    columns_of_interest = ['canonical_smiles', 'standard_type', 'standard_value', 'standard_units']
    # these next two lines store a QuerySet object in the variable 'res'. this is not the same as
    #   the Django QuerySet object.
    activity = new_client.activity
    res = activity.filter(assay_chembl_id = assay_chembl_id).only(columns_of_interest)

    n_entries = len(res)

    if not quiet:
        print('Now fetching data from the ChEMBL database. This may take a while.')
    # using the built-in pandas method to load data is much faster than iterating manually
    df = pd.DataFrame.from_records(res)
    if not quiet:
        print(f'Time to fetch data from ChEMBL: {time.time()-start_time}')
        print(f'There are {n_entries} entries in assay {assay_chembl_id}.\n')
        print(f'The data type is {df.type[0]}')

    # we take some preliminary clean-up steps. first we drop any entries missing a SMILES string,
    #   an assay value, or unit for the value
    df = df.dropna(subset=['canonical_smiles', 'standard_value', 'standard_units'])
    # then we remove any entries that are perfect duplicates
    df = df.drop_duplicates()

    # finally we remove columns we don't want and rename things to be convenient for us
    df = df.drop(columns=['standard_type', 'standard_units', 'standard_value', 'type', 'units'])
    df = df.rename(columns={'canonical_smiles': 'smiles', 'value': 'label'})
    # we have to retype the label column from str to float for this next part
    df = df.astype({'label': 'float64'})

    # first we find all entries with duplicated SMILES strings then we drop those entries
    #   from the original dataframe, we'll bring them back later
    dup_smi = df[df.duplicated('smiles', keep=False)]
    df = df.drop(dup_smi.index, axis=0)

    # here, for each SMILES string among the duplicates, we find the standard deviation.
    #   if it is below the threshold, we save the mean of the duplicated values
    clean_dup_list = []
    smi_list = list(set(dup_smi.smiles))
    for smi in smi_list:
        p = dup_smi.loc[dup_smi['smiles'] == smi].label
        if np.std(p) < thresh:
            clean_dup_list.append([smi, np.mean(p)])

    # then we take those values and put them back into the dataframe
    uni_smi = pd.DataFrame(clean_dup_list, columns=['smiles', 'label'])
    df = df.append(uni_smi, ignore_index=True)

    if not quiet:
        print(f'The shape of the final dataset is {df.shape}')
        print('The last five entries are:')
        print(df.tail())
        print('')

    if not quiet:
        print(f'Wall time: {time.time()-start_time}')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='takes a ChEMBL assay ID and returns a pandas Dataframe object containing the canonical SMILES and the ChEMBL value')
    parser.add_argument('assay_id', help='the ChEMBL assay ID you want to fetch data for')
    parser.add_argument('thresh', help='this argument sets the threshold for removing duplicates', type=float)
    parser.add_argument('-q', '--quiet', help='this flag suppresses any printing to the console', action='store_true')
    args = parser.parse_args()

    df = chembl_to_dataframe(args.assay_id, args.thresh, args.quiet)
    if not args.quiet:
        print(f'\nNow saving a pandas Dataframe to {args.assay_id}.pkl')
    df.to_pickle(f'./{args.assay_id}.pkl')
