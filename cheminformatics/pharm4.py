from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
import py3Dmol

class Pharm4:
    def __init__(self, _mol):
        self.mol = GetLowestEnergyConformer(_mol)
        self.features = {
            "HBA": [],
            "wHBA": [],
            "HBD": [],
            "cation": [],
            "anion": [],
            "aromatic": []
        }
        self.vectors = {
            "HBA": [],
            "wHBA": [],
            "HBD": [],
            "aromatic": []
        }
        self.GenerateFeaturesAndVectors()
        self.pharmacophore = self.GeneratePharmacophore()
    
    @classmethod
    def Pharm4FromSmiles(cls, _smi):
        _mol = Chem.MolFromSmiles(_smi)
        return cls(_mol)

    def GenerateFeaturesAndVectors(self):
        mol_no_h = Chem.RemoveAllHs(self.mol)
        feat_coords = []

        smart_feat_object = SmartsFeatures()
        smarts_features = smart_feat_object.smarts_features
        smarts_features_keys = smart_feat_object.smarts_features_keys

        smarts_features_ids = {}
        for i in range(len(smarts_features_keys)):
            smarts_feat = smarts_features_keys[i]
            smarts_feat_ids = []
            if "aromatic_rings" in smarts_feat:
                for j in range(len(smarts_features[smarts_feat])):
                    smarts_feat_ids += GetAtomIdsInSubstruct(mol_no_h, smarts_features[smarts_feat][j], _unique_and_sorted=False)
                
                if "six" in smarts_feat:
                    self.FeaturizeAromatic(mol_no_h, smarts_feat_ids, 6)
                if "five" in smarts_feat:
                    self.FeaturizeAromatic(mol_no_h, smarts_feat_ids, 5)

            else:
                for j in range(len(smarts_features[smarts_feat])):
                    smarts_feat_ids += GetAtomIdsInSubstruct(mol_no_h, smarts_features[smarts_feat][j])
            smarts_features_ids.update({smarts_feat: smarts_feat_ids})

        for atom in mol_no_h.GetAtoms():
            self.FeaturizeHBAAndCation(atom, smarts_features_ids)
            self.FeaturizeHBDAndAnion(atom, smarts_features_ids)

    def FeaturizeHBAAndCation(self, _atom, _smarts_features_ids):
        sfi = _smarts_features_ids
        weak_HBA = False
        none_HBA = False
        cation = False
        id = _atom.GetIdx()
        conf = self.mol.GetConformer()

        if _atom.GetAtomicNum() == 7:
            if id in sfi["aromatic_amine_smarts"] or (id in sfi["pyrrole_smarts"] and CountSingleBonds(_atom) == 3):
                weak_HBA = True
            if (id in sfi["nitroso_smarts"] or id in sfi["amide_smarts"] or id in sfi["sulfonamide_smarts"] or 
                (id in sfi["n_oxide_smarts"] and _atom.GetFormalCharge() == 1)):
                weak_HBA = False
                none_HBA = True
            if not weak_HBA and not none_HBA and (CountSingleBonds(_atom) == 3 or id in sfi["pyridine_smarts"] or id in _smarts_features_ids["imine_smarts"]):
                cation = True

            if weak_HBA:
                self.features["wHBA"].append(conf.GetAtomPosition(id))
            elif none_HBA:
                pass
            else:
                self.features["HBA"].append(conf.GetAtomPosition(id))
            if cation and not weak_HBA and not none_HBA:
                self.features["cation"].append(conf.GetAtomPosition(id))

        if _atom.GetAtomicNum() == 8:
            if id in sfi["phenol_smarts"] or id in sfi["furan_smarts"] or id in sfi["nitro_smarts"]:
                weak_HBA = True
            
            if weak_HBA:
                self.features["wHBA"].append(conf.GetAtomPosition(id))
            else:
                self.features["HBA"].append(conf.GetAtomPosition(id))

        if _atom.GetAtomicNum() == 9:
            self.features["wHBA"].append(conf.GetAtomPosition(id))

        if _atom.GetAtomicNum() == 16:
            if not id in sfi["sulfur_oxides_smarts"]:
                if id in sfi["thioether_smarts"]:
                    self.features["wHBA"].append(conf.GetAtomPosition(id))
                if id in sfi["thiocarbonyl_smarts"]:
                    self.features["HBA"].append(conf.GetAtomPosition(id))

    def FeaturizeHBDAndAnion(self, _atom, _smarts_features_ids):
        sfi = _smarts_features_ids
        id = _atom.GetIdx()
        conf = self.mol.GetConformer()
        HBD_atom_pos = conf.GetAtomPosition(id)

        if _atom.GetAtomicNum() == 7 and _atom.GetTotalNumHs() > 0:
            for hyd in self.mol.GetAtoms()[id].GetNeighbors():
                if hyd.GetAtomicNum() == 1:
                    HBD_hyd_pos = conf.GetAtomPosition(hyd.GetIdx())
                    coord = Geometry.rdGeometry.Point3D((HBD_atom_pos.x+HBD_hyd_pos.x)/2, (HBD_atom_pos.y+HBD_hyd_pos.y)/2, (HBD_atom_pos.z+HBD_hyd_pos.z)/2)
                    self.features["HBD"].append(coord)
                    if id in sfi["acyl_sulfonamide_smarts"] or id in sfi["tetrazole_smarts"] or id in sfi["n_acidic_heterocycle_smarts"]:
                        self.features["anion"].append(HBD_atom_pos)

        if _atom.GetAtomicNum() == 8 and _atom.GetTotalNumHs() > 0:
            for hyd in self.mol.GetAtoms()[id].GetNeighbors():
                if hyd.GetAtomicNum() == 1:
                    HBD_hyd_pos = conf.GetAtomPosition(hyd.GetIdx())
                    coord = Geometry.rdGeometry.Point3D((HBD_atom_pos.x+HBD_hyd_pos.x)/2, (HBD_atom_pos.y+HBD_hyd_pos.y)/2, (HBD_atom_pos.z+HBD_hyd_pos.z)/2)
                    self.features["HBD"].append(coord)
                    if id in sfi["carboxylic_acid_smarts"] or id in sfi["phosphorus_acids_smarts"] or id in sfi["sulfur_acids_smarts"] or id in sfi["o_acidic_heterocycle_smarts"]:
                        self.features["anion"].append(HBD_atom_pos)

    def FeaturizeAromatic(self, _mol, _aromatic_ring_ids, _num_mem_ring):
        conf = _mol.GetConformer()
        for i in range(int(len(_aromatic_ring_ids)/_num_mem_ring)):
            new_coords = [0, 0, 0]
            for j in range(_num_mem_ring):
                atom_pos = conf.GetAtomPosition(_aromatic_ring_ids[i*_num_mem_ring + j])
                new_coords[0] += atom_pos[0]
                new_coords[1] += atom_pos[1]
                new_coords[2] += atom_pos[2]
            new_coords = [new_coords[0]/_num_mem_ring, new_coords[1]/_num_mem_ring, new_coords[2]/_num_mem_ring]
            self.features["aromatic"].append(Geometry.rdGeometry.Point3D(new_coords[0], new_coords[1], new_coords[2]))

    def GeneratePharmacophore(self):
        pharmacophore = Chem.rdchem.RWMol()
        pharm_coords = Chem.rdchem.Conformer(self.GetTotalNumFeatures())
        features_keys = list(self.features)

        feat_index = 0
        for i in range(len(self.features)):
            feat_key = features_keys[i]
            for j in range(len(self.features[feat_key])):
                pharmacophore.AddAtom(Chem.Atom(FeatureToAtomNum(feat_key)))
                pharm_coords.SetAtomPosition(feat_index, self.features[feat_key][j])
                feat_index += 1

        if not feat_index == self.GetTotalNumFeatures():
            print("the number of features does not match up")

        pharmacophore = pharmacophore.GetMol()
        pharmacophore.AddConformer(pharm_coords)
        return pharmacophore

    def GetTotalNumFeatures(self):
        count = 0
        features_keys = list(self.features)
        for i in range(len(self.features)):
            feat_key = features_keys[i]
            count += len(self.features[feat_key])
        return count

    def DisplayPharm4(self, _display_mol=True):
        view = py3Dmol.view(width=800)
        if _display_mol:
            view = py3Dmol.view(data=Chem.rdmolfiles.MolToMolBlock(RemoveNonPolarH(self.mol), includeStereo=True), style={'stick':{'colorscheme':'grayCarbon'}}, width=800)

        features_keys = list(self.features)
        for i in range(len(self.features)):
            feat_key = features_keys[i]
            for j in range(len(self.features[feat_key])):
                color, radius = FeatureToColorAndRadius(feat_key)
                coord = self.features[feat_key][j]
                coord = dict(x=coord.x, y=coord.y, z=coord.z)

                view.addSphere({"center": coord,
                                "hidden": False,
                                "radius": radius,
                                "scale": 1,
                                "color": color,
                                "opacity": 0.5,
                              })

        view.show()
        view.clear()

def CountSingleBonds(_atom):
    """
    this function takes an RDKit Atom objected embedded in an RDKit Mol object and returns the number of 
      single bonds including bonds to hydrogens and aromatic bonds
    """
    bonds = _atom.GetBonds()
    singles = 0

    for i in range(len(bonds)):
        if bonds[i].GetBondTypeAsDouble() == 1 or bonds[i].GetBondTypeAsDouble() == 1.5:
            singles = singles + 1
    
    return singles + _atom.GetTotalNumHs()

def GetLowestEnergyConformer(_mol):
    """
    this function takes an RDKit Mol object and returns an RDKit Mol with the lowest energy conformer embedded
    """
    # first we create a copy of the Mol object. we then remove all conformers and add hydrogens
    mol = Chem.Mol(_mol)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    return_mol = Chem.Mol(mol)

    # the number of conformers generated depends on the number of rotatable bonds
    #     pruneRmsThresh ensures that all conformers will be at least different by an RMSD of 0.5
    n_rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot <= 7:
        confs_id = Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=50, pruneRmsThresh=0.5)
    elif n_rot > 7 and n_rot <= 12:
        confs_id = Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=200, pruneRmsThresh=0.5)
    elif n_rot > 12:
        confs_id = Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=300, pruneRmsThresh=0.5)

    # the conformers are optimized using the MMFF force field
    #     the maxIters has to be high enough to ensure convergence of results
    mmff = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters = 1000)

    # this loop finds the index and energy of the lowest energy conformer
    energy = 1000000.0
    index = 0
    for i, mf in enumerate(mmff):
        if mf[0] != 0:
            print("there is an error with conformer. could be failure to initialize or converge. index:  ", i)
        if mf[1] < energy:
            energy = mf[1]
            index = i

    # we add the lowest energy conformer to our return_mol object which does not have any conformers embedded
    return_mol.AddConformer(mol.GetConformer(index))

    return return_mol

def RemoveNonPolarH(_mol_h):
    """
    this function takes an RDKit Mol object and returns an RDKit Mol object with all nonpolar hydrogens removed
    """
    mol_polar_h = Chem.rdchem.RWMol(_mol_h)
    h_to_remove = []
    bond_to_remove = []
    for atom in mol_polar_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            for hyd in atom.GetNeighbors():
                if hyd.GetAtomicNum() == 1:
                    h_to_remove.append(hyd.GetIdx())
                    bond_to_remove.append(atom.GetIdx())
    h_to_remove = sorted(h_to_remove, reverse=True)
    for idx in h_to_remove:
        # mol_polar_h.RemoveBond(idx, bond_to_remove[i])
        mol_polar_h.RemoveAtom(idx)
    
    return mol_polar_h.GetMol()

def ReflectMol(_mol):
    old_conf = _mol.GetConformer()
    return_mol = Chem.Mol(_mol)
    return_mol.RemoveAllConformers()
    atoms_list = return_mol.GetAtoms()
    new_conf = Chem.rdchem.Conformer(len(atoms_list))
    for i, atom in enumerate(atoms_list):
        atom_pos = old_conf.GetAtomPosition(i)
        new_conf.SetAtomPosition(i, Geometry.rdGeometry.Point3D(-atom_pos.x, atom_pos.y, atom_pos.z))
    return_mol.AddConformer(new_conf)
    return return_mol

def Display3DMol(_mol): 
    """
    this function takes an RDKit Mol object and displays it using py3Dmol in the notebook output
    """
    view = py3Dmol.view(data=Chem.rdmolfiles.MolToMolBlock(RemoveNonPolarH(_mol), includeStereo=True), style={'stick':{'colorscheme':'grayCarbon'}}, width=800)
    view.show()
    view.clear()

def TuplesToList(_tuple_of_tuples, _unique_and_sorted=True):
    """
    this function takes a _tuple_of_tuples generated by GetSubstructMatches and returns a list containing all
      the elements of the _tuple_of_tuples.
    if _unique_and_sorted = True, duplicates will be removed and the elements are sorted
    if _unique_and_sorted = False, the duplicates and ordering in the _tuple_of_tuples will be retained
    """
    output = []
    for i in range(len(_tuple_of_tuples)):
        for j in range(len(_tuple_of_tuples[i])):
            output.append(_tuple_of_tuples[i][j])
    if _unique_and_sorted:
        output = sorted(set(output))
    return output

def GetAtomIdsInSubstruct(_mol, _smarts_substruct, _unique_and_sorted=True):
    """
    
    """
    return TuplesToList(_mol.GetSubstructMatches(Chem.MolFromSmarts(_smarts_substruct)), _unique_and_sorted)

def FeatureToAtomNum(_feature_type):
    """
    
    """
    if _feature_type == "HBA":
        return 6
    if _feature_type == "wHBA":
        return 7
    if _feature_type == "HBD":
        return 8
    if _feature_type == "cation":
        return 9
    if _feature_type == "anion":
        return 15
    if _feature_type == "aromatic":
        return 16

def FeatureToColorAndRadius(_feature_type):
    """
    
    """
    if _feature_type == "HBA":
        return "blue", 0.8
    if _feature_type == "wHBA":
        return "lightblue", 0.8
    if _feature_type == "HBD":
        return "darkorange", 0.8
    if _feature_type == "cation":
        return "purple", 0.8
    if _feature_type == "anion":
        return "black", 0.8
    if _feature_type == "aromatic":
        return "green", 1.3

class SmartsFeatures:
    def __init__(self):
        self.smarts_features = {
            # SMARTS features for nitrogen HBAs
            "aromatic_amine_smarts": ["Na"],
            "pyrrole_smarts": ["n1aaaa1"],
            "nitroso_smarts": ["N=O"],
            "n_oxide_smarts": ["[#7]O"],
            "sulfonamide_smarts": ["[#7]S=O"],
            "amide_smarts": ["[#7][#6]=O", "[#7][#6]=S", "[#7][#6]=N"],

            # SMARTS features for oxygen HBAs
            "phenol_smarts": ["Oc"],
            "furan_smarts": ["o1aaaa1"],
            "nitro_smarts": ["ON=O"],

            # SMARTS features for sulfur HBAs
            "sulfur_oxides_smarts": ["S~O"],
            "thioether_smarts": ["[#6]S"],
            "thiocarbonyl_smarts": ["S=C"],

            # SMARTS features for nitrogen bases
            "pyridine_smarts": ["n1ccccc1"],
            "imine_smarts": ["N=C"],

            # SMARTS features for nitrogen anions
            "acyl_sulfonamide_smarts": ["O=CNS=O"],
            "tetrazole_smarts": ["c1nnnn1"],
            "n_acidic_heterocycle_smarts": ["O=C1NC(CS1)=O", "O=C1NC(CO1)=O", "O=C1NC=NO1", "O=C1NC=NS1", "O=S1NC=NO1", "S=C1NC=NO1", "O=C(C1)CNC1=O"],

            # SMARTS features for oxygen anions
            "carboxylic_acid_smarts": ["O=CO"],
            "phosphorus_acids_smarts": ["O=PO"],
            "sulfur_acids_smarts": ["O=SO"],
            "o_acidic_heterocycle_smarts": ["OC1=NOC=C1", "O=C1CCC(O)=C1", "O=C(C=C1O)C1=O"],

            # key features for aromatic rings
            "six_mem_aromatic_rings_smarts": ["a1aaaaa1"],
            "five_mem_aromatic_rings_smarts": ["a1aaaa1"]
        }
        self.smarts_features_keys = list(self.smarts_features)