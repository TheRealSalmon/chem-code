# cheminformatics

**pharm4.py**
<br>
pharm4.py is a pharmacophore visualizer I wrote while I was first learning
RDKit. I was planning to add methods to align pharmacophores but gave up once 
I realized how difficult it would be to do so. I hope to rewrite to the code 
to be better documented, follow modern conventions, and add pharmacophore 
alignment.

**utils.py**
<br>
utils.py contains a hodgepodge of useful Python functions I've written throughout 
my learning. Some neat examples include:
* get_low_e_conformer
  * uses RDKit to generate conformers and find the lowest MMFF energy conformer
* remove_nonpolar_hs
  * removes all nonpolar (C-H) hydrogens from a molecules
* display_3d_mol
  * uses py3Dmol to display an RDKit Mol in jupyter-lab
