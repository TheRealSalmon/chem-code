from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
import time
import argparse

def chembl_to_dataframe(assay_chembl_id, quiet):
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
    
    # # next we convert the assay value into a log value. we create a dict that converts the unit type
    # #   into the corresponding power of 10
    # units = {'nM': '1000000000',
    #          'uM': '1000000',
    #          'mM': '1000'}
    # # then we use replace() to feed the data in the column 'standard_units' through the dict called 'units'
    # #   to repopulate the column 'standard_units' with the right power of 10 for the next step. note that we
    # #   have to retype this data as floats for the next step
    # df = df.replace({'standard_units': units})
    # df = df.astype({'standard_value': 'float64', 'standard_units': 'float64'})
    # # here we create a new column called 'p_potency' and fill it with the log potency values
    # df['p_potency'] = -np.log10(df.standard_value / df.standard_units)
    # # finally we get rid of all the columns we don't need 
    # df = df.drop(columns=['standard_type', 'standard_units', 'standard_value', 'type', 'units', 'value'])
    
    df = df.drop(columns=['standard_type', 'standard_units', 'standard_value', 'type', 'units'])
    
    if not quiet:
        print('The last five entries are:')
        print(df.tail())
        print('')
    
    # if we run into a case where a ChEMBL assay has repeated data for the same compound, we'll have to do some
    #   more coding. until then, this will suffice to alert me if it happens
    dup_smi = df[df.duplicated('canonical_smiles')]
    if not dup_smi.empty:
        print('Some SMILES strings are duplicated. Time to do some more coding!\n')
    
    if not quiet:
        print(f'Wall time: {time.time()-start_time}')
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='takes a ChEMBL assay ID and returns a pandas Dataframe object containing the canonical SMILES and the ChEMBL value')
    parser.add_argument('assay_id', help='the ChEMBL assay ID you want to fetch data for')
    parser.add_argument('-q', '--quiet', help='this flag suppresses any printing to the console', action='store_true')
    args = parser.parse_args()
    
    df = chembl_to_dataframe(args.assay_id, args.quiet)
    if not args.quiet:
        print(f'\nNow saving a pandas Dataframe to {args.assay_id}.pkl')
    df.to_pickle(f'./{args.assay_id}.pkl')
