from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdDistGeom, AllChem
import py3Dmol
import tempfile
import subprocess

class xtbError(Exception):
    pass

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

def xtb_single_point(input_mol: Chem.rdchem.Mol,
                     charge: int = 0,
                     e_state: int = 0,
                     solvent: str = '',
                     mmff_max_iters: int = 200) -> float:
    """Use xtb to calculate the single point energy in Hartrees.
    
    Takes an RDKit Mol object and calculates a single point energy using xtb. 
    Requires prior installation of xtb.

    Examples
    --------
    mol = Chem.MolFromSmiles('OCCCO')
    mol = get_low_energy_conformer(mol)
    e = xtb_single_point(mol)

    Parameters
    ----------
    mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object. 
    charge: `int`, default = 0
        The total charge of the molecule
    e_state: `int`, default = 0
        N_alpha - N_beta. The difference between the number of spin up and
        spin down electrons. Should usually be 0 unless you are running open
        shell or triplet calculations.
    solvent: `str`, default = ''
        The solvent used for xtb calculations. Choices are acetone, 
        acetonitrile, ch2cl2, chcl3, cs2, dmf, dmso, ether, h2o, methanol,
        n-hexane, thf, toluene. The default is no solvent.
    mmff_max_iters: `int`, default = 200
        The number of iterations RDKit's MMFF optimizer will use.

    Returns
    -------
    `float`
        The total energy of the system in Hartrees."""
    # checks for embedded conformer, if not found, adds an MMFF conformer
    mol = Chem.rdchem.Mol(input_mol)
    if len(mol.GetConformers()) == 0:
        mol = get_low_energy_conformer(mol, mmff_max_iters)

    # runs calculations in tmp directory
    with tempfile.TemporaryDirectory() as tmp:
        # create .xyz file in the tmp directory
        Chem.rdmolfiles.MolToXYZFile(mol, f'{tmp}/input.xyz')
        # run xtb on the input file
        xtb_args = ['-c', str(charge), '-u', str(e_state)]
        if solvent != '':
            xtb_args += ['-g', solvent]
        proc = subprocess.run(['xtb', 'input.xyz'] + xtb_args, 
                              cwd=tmp,
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise xtbError('xtb abnormal termination')
        xtb_out = proc.stdout.decode('utf-8').split('\n')
        for line in reversed(xtb_out):
            if 'TOTAL ENERGY' in line:
                return line.split()[3]

def xtb_geom_opt(input_mol: Chem.rdchem.Mol,
                 charge: int = 0,
                 e_state: int = 0,
                 solvent: str = '',
                 mmff_max_iters: int = 200) -> Chem.rdchem.Mol:
    """Use xtb to perform a geometry optimization.

    Examples
    --------
    mol = Chem.MolFromSmiles('OCCCO')
    mol = get_low_energy_conformer(mol)
    mol = xtb_geom_opt(mol)
    energy = xtb_single_point(mol)

    Parameters
    ----------
    mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object. 
    charge: `int`, default = 0
        The total charge of the molecule
    e_state: `int`, default = 0
        N_alpha - N_beta. The difference between the number of spin up and
        spin down electrons. Should usually be 0 unless you are running open
        shell or triplet calculations.
    solvent: `str`, default = ''
        The solvent used for xtb calculations. Choices are acetone, 
        acetonitrile, ch2cl2, chcl3, cs2, dmf, dmso, ether, h2o, methanol,
        n-hexane, thf, toluene. The default is no solvent.
    mmff_max_iters: `int`, default = 200
        The number of iterations RDKit's MMFF optimizer will use.

    Returns
    -------
    `rdkit.Chem.rdchem.Mol`
        An RDKit Mol object embedded with an optimized structure."""
    # makes sure that there is a conformer embedded in mol
    mol = Chem.rdchem.Mol(input_mol)
    if len(mol.GetConformers()) == 0:
        mol = get_low_energy_conformer(mol, mmff_max_iters)

    xtb_xyz = ''
    # runs calculations in tmp directory
    with tempfile.TemporaryDirectory() as tmp:
        # create .xyz file in the tmp directory
        Chem.rdmolfiles.MolToXYZFile(mol, f'{tmp}/input.xyz')
        # run xtb on the input file
        xtb_args = ['-c', str(charge), '-u', str(e_state)]
        if solvent != '':
            xtb_args += ['-g', solvent]
        proc = subprocess.run(['xtb', 'input.xyz', '--opt'] + xtb_args, 
                              cwd=tmp,
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise xtbError('xtb abnormal termination')
        with open(f'{tmp}/xtbopt.xyz') as file:
            # first two lines of xyz are atom count and comments
            # last line is blank
            xtb_xyz = file.read().split('\n')[2:len(xtb_xyz)-1]

    # creates a new RDKit Mol with embedded conformer from the xtb xyz output
    mol.RemoveAllConformers()
    conf = Chem.rdchem.Conformer(mol.GetNumAtoms())
    for i, line in enumerate(xtb_xyz):
        ls = line.split()
        x, y, z = float(ls[1]), float(ls[2]), float(ls[3])
        conf.SetAtomPosition(i, Geometry.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)
    return mol

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

def display_3d_mol(mol: Chem.rdchem.Mol, 
                   nonpolar_h: bool = False) -> None:
    """Use py3Dmol to visualize mol in 3D.

    Examples
    --------
    mol = Chem.MolFromSmiles('OCCCO')
    mol = get_low_energy_conformer(mol)
    display_3d_mol(mol)

    Parameters
    ----------
    mol: `rdkit.Chem.rdchem.Mol`
        The input RDKit mol object with an embedded 3D conformer. 
    nonpolar_h: `bool`, default = False
        Whether or not to show nonpolar (C-H) hydrogens"""
    mol_block = ''
    if nonpolar_h:
        mol_block = Chem.rdmolfiles.MolToMolBlock(mol, includeStereo=True)
    else:
        mol_block = Chem.rdmolfiles.MolToMolBlock(remove_nonpolar_hs(mol), includeStereo=True)
    view = py3Dmol.view(data=mol_block, 
                        style={'stick':{'colorscheme':'grayCarbon'}})
    view.show()

