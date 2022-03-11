# cheminformatics

**pharm4.py**
<br>
pharm4.py is a pharmacophore visualizer I wrote while I was first learning
RDKit. I was planning to add methods to align pharmacophores but gave up once 
I realized how difficult it would be to do so. I hope to rewrite to the code 
to be better documented, follow modern conventions, and add pharmacophore 
alignment.

**crest_confs_py**
CREST and xtb are two very cool command line programs created by the Grimme 
group. xtb is a semiempirical tight-binding DFT package, meaning that it still 
solves an SCF for the Hamiltonian, but the solutions are parametrized. 
Basically it has the flexibility of DFT but the speed of MM. CREST is built off
xtb and has a lot of useful applications, particularly in using sampling to 
find relevant conformers. It uses "meta-dynamics" MD simulations to sample more
of the conformation space. 

I wrote some Python functions to interface with these two softwares. The 
calculations are all run from subprocess.run in a tmp directory. 

**utils.py**
<br>
utils.py contains a hodgepodge of useful Python functions I've written 
throughout my learning. Some neat examples include:
* get_low_e_conformer
  * uses RDKit to generate conformers and find the lowest MMFF energy conformer
* remove_nonpolar_hs
  * removes all nonpolar (C-H) hydrogens from a molecules
* display_3d_mol
  * uses py3Dmol to display an RDKit Mol in jupyter-lab
