from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolDescriptors, AllChem

def get_low_energy_conformer(input_mol: Chem.rdchem.Mol, 
                             max_iters: int = 200) -> Chem.rdchem.Mol:
    """Obtain the lowest energy conformer.
    
    Finds the MMFF low-energy conformer. It generates n conformers, where n 
    depends on the number of rotatable bonds. Then the conformers are optimized
    with the MMFF forcefield. Finally, the lowest energy conformer is returned.
    Will raise error if the number of rotatable bonds is greater than 10.

    Examples
    --------
    mol = Chem.MolFromSmiles('OCCCO')
    low_e_mol = get_low_energy_conformer(mol)

    Parameters
    ----------
    input_mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object. 
    max_iters: `int`, default = 200
        The number of iterations allowed for the MMFF optimization.

    Returns
    -------
    `rdkit.Chem.rdchem.Mol`
        An RDKit Mol object embedded with the (hopefully) lowest energy
        conformer"""
    # Make a copy of input mol. Second argument indicates a quickCopy, where 
    #  properties and conformers are not copied.
    mol = Chem.rdchem.Mol(input_mol, True)
    mol = Chem.AddHs(mol)
    low_e_mol = Chem.rdchem.Mol(mol, True)

    # Use the number of rotatable bonds to determine the number of conformers to
    #  generate. Raise ValueError if number of rotatable bonds is too high. 
    #  See https://doi.org/10.1021/ci400020a for further details.
    n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot_bonds <= 6:
        rdDistGeom.EmbedMultipleConfs(mol, numConfs=50, pruneRmsThresh=0.5)
    elif n_rot_bonds > 6 and n_rot_bonds <=10:
        rdDistGeom.EmbedMultipleConfs(mol, numConfs=200, pruneRmsThresh=0.5)
    else:
        raise ValueError('Too many rotatable bonds.')

    # Optimize all conformers embeded in mol.
    opt = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)

    # Find the index of lowest energy conformer.
    energy = 1000.0
    index = 0
    for i, o in enumerate(opt):
        # o is a tuple where o[0] = 0 if optimization converged and 1 if not. 
        #  o[1] is the energy of the final structure.
        if o[0] != 0:
            print(f'Conformer {i} failed to converge.')
        else:
            if o[1] < energy:
                energy = o[1]
                index = i

    # Add the lowest energy found conformer to low_e_mol and return it.
    low_e_mol.AddConformer(mol.GetConformer(index))
    return low_e_mol

def remove_nonpolar_hs(input_mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """Remove nonpolar hydrogen atoms.
    
    Finds all hydrogens bonded to carbon atoms and returns an RDKit Mol object
    with these hydrogens removed.

    Examples
    --------
    mol = Chem.MolFromSmiles('OCCCO')
    mol_polar_h = remove_nonpolar_hs(mol)

    Parameters
    ----------
    input_mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object. 

    Returns
    -------
    `rdkit.Chem.rdchem.Mol`
        An RDKit Mol object with all nonpolar hydrogens removed."""
    # Make a copy of input mol.
    mol = Chem.rdchem.Mol(input_mol)

    # Find indices of all hydrogens bonded to carbons.
    nonpolar_hs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            for n in atom.GetNeighbors():
                if n.GetAtomicNum() == 1:
                    nonpolar_hs.append(n.GetIdx())
    # The list needs to be ordered from high-to-low to avoid indexing issues.
    nonpolar_hs = sorted(nonpolar_hs, reverse=True)

    # We create a Read/Write Mol and remove the nonpolar hydrogens.
    rwmol = Chem.rdchem.RWMol(mol)
    for h in nonpolar_hs:
        rwmol.RemoveAtom(h)

    return rwmol.GetMol()

def get_atom_ids_in_substruct(input_mol: Chem.rdchem.Mol, 
                              smarts_substruct: str, 
                              unique_and_sorted: bool = True) -> list:
    """Returns all atoms that are part of a SMARTS substructure.

    Parameters
    ----------
    input_mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object.
    smarts_substruct: `str`
        The SMARTS substructure of interest.
    unique_and_sorted: `bool`, default = True
        If True, the returned list will not have duplicates and will be sorted.
        It is useful to set to false when looking at ring systems.

    Returns
    -------
    atom_ids: `list`
        A list of atoms that are part of the specified SMARTS substructure."""

    tuple_of_tuples = input_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_substruct))
    atom_ids = []
    for i in range(len(tuple_of_tuples)):
        for j in range(len(tuple_of_tuples[i])):
            atom_ids.append(tuple_of_tuples[i][j])
    if unique_and_sorted:
        atom_ids = sorted(set(atom_ids))
    return atom_ids

