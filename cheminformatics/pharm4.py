from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from utils import (get_low_energy_conformer, remove_nonpolar_hs, 
                  get_atom_ids_in_substruct)
import py3Dmol

class Pharm4:
    def __init__(self, mol):
        self.mol = get_low_energy_conformer(mol)
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
    def Pharm4FromSmiles(cls, smi):
        mol = Chem.MolFromSmiles(smi)
        return cls(mol)

    def GenerateFeaturesAndVectors(self):
        mol_no_h = Chem.RemoveAllHs(self.mol)
        smarts_features = {
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
        smarts_features_keys = list(smarts_features)
        smarts_features_ids = {}

        # this way of iterating through my SMARTS features is admittedly hacky
        for i in range(len(smarts_features_keys)):
            smarts_feat = smarts_features_keys[i]
            smarts_feat_ids = []
            if "aromatic_rings" in smarts_feat:
                for j in range(len(smarts_features[smarts_feat])):
                    smarts_feat_ids += get_atom_ids_in_substruct(mol_no_h, smarts_features[smarts_feat][j], unique_and_sorted=False)
                
                if "six" in smarts_feat:
                    self.FeaturizeAromatic(mol_no_h, smarts_feat_ids, 6)
                if "five" in smarts_feat:
                    self.FeaturizeAromatic(mol_no_h, smarts_feat_ids, 5)

            else:
                for j in range(len(smarts_features[smarts_feat])):
                    smarts_feat_ids += get_atom_ids_in_substruct(mol_no_h, smarts_features[smarts_feat][j])
            smarts_features_ids.update({smarts_feat: smarts_feat_ids})

        for atom in mol_no_h.GetAtoms():
            self.FeaturizeHBAAndCation(atom, smarts_features_ids)
            self.FeaturizeHBDAndAnion(atom, smarts_features_ids)

    def FeaturizeHBAAndCation(self, atom, smarts_features_ids):
        sfi = smarts_features_ids
        weak_HBA = False
        none_HBA = False
        cation = False
        id = atom.GetIdx()
        conf = self.mol.GetConformer()

        if atom.GetAtomicNum() == 7:
            if id in sfi["aromatic_amine_smarts"] or (id in sfi["pyrrole_smarts"] and atom.GetTotalDegree() == 3):
                weak_HBA = True
            if (id in sfi["nitroso_smarts"] or id in sfi["amide_smarts"] or id in sfi["sulfonamide_smarts"] or 
                (id in sfi["n_oxide_smarts"] and atom.GetFormalCharge() == 1)):
                weak_HBA = False
                none_HBA = True
            if not weak_HBA and not none_HBA and (atom.GetTotalDegree() == 3 or id in sfi["pyridine_smarts"] or id in sfi["imine_smarts"]):
                cation = True

            if weak_HBA:
                self.features["wHBA"].append(conf.GetAtomPosition(id))
            elif none_HBA:
                pass
            else:
                self.features["HBA"].append(conf.GetAtomPosition(id))
            if cation and not weak_HBA and not none_HBA:
                self.features["cation"].append(conf.GetAtomPosition(id))

        if atom.GetAtomicNum() == 8:
            if id in sfi["phenol_smarts"] or id in sfi["furan_smarts"] or id in sfi["nitro_smarts"]:
                weak_HBA = True
            
            if weak_HBA:
                self.features["wHBA"].append(conf.GetAtomPosition(id))
            else:
                self.features["HBA"].append(conf.GetAtomPosition(id))

        if atom.GetAtomicNum() == 9:
            self.features["wHBA"].append(conf.GetAtomPosition(id))

        if atom.GetAtomicNum() == 16:
            if not id in sfi["sulfur_oxides_smarts"]:
                if id in sfi["thioether_smarts"]:
                    self.features["wHBA"].append(conf.GetAtomPosition(id))
                if id in sfi["thiocarbonyl_smarts"]:
                    self.features["HBA"].append(conf.GetAtomPosition(id))

    def FeaturizeHBDAndAnion(self, atom, smarts_features_ids):
        sfi = smarts_features_ids
        id = atom.GetIdx()
        conf = self.mol.GetConformer()
        HBD_atom_pos = conf.GetAtomPosition(id)

        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0:
            for hyd in self.mol.GetAtoms()[id].GetNeighbors():
                if hyd.GetAtomicNum() == 1:
                    HBD_hyd_pos = conf.GetAtomPosition(hyd.GetIdx())
                    coord = Geometry.rdGeometry.Point3D((HBD_atom_pos.x+HBD_hyd_pos.x)/2, (HBD_atom_pos.y+HBD_hyd_pos.y)/2, (HBD_atom_pos.z+HBD_hyd_pos.z)/2)
                    self.features["HBD"].append(coord)
                    if id in sfi["acyl_sulfonamide_smarts"] or id in sfi["tetrazole_smarts"] or id in sfi["n_acidic_heterocycle_smarts"]:
                        self.features["anion"].append(HBD_atom_pos)

        if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0:
            for hyd in self.mol.GetAtoms()[id].GetNeighbors():
                if hyd.GetAtomicNum() == 1:
                    HBD_hyd_pos = conf.GetAtomPosition(hyd.GetIdx())
                    coord = Geometry.rdGeometry.Point3D((HBD_atom_pos.x+HBD_hyd_pos.x)/2, (HBD_atom_pos.y+HBD_hyd_pos.y)/2, (HBD_atom_pos.z+HBD_hyd_pos.z)/2)
                    self.features["HBD"].append(coord)
                    if id in sfi["carboxylic_acid_smarts"] or id in sfi["phosphorus_acids_smarts"] or id in sfi["sulfur_acids_smarts"] or id in sfi["o_acidic_heterocycle_smarts"]:
                        self.features["anion"].append(HBD_atom_pos)

    def FeaturizeAromatic(self, mol, aromatic_ring_ids, num_mem_ring):
        conf = mol.GetConformer()
        for i in range(int(len(aromatic_ring_ids)/num_mem_ring)):
            new_coords = [0, 0, 0]
            for j in range(num_mem_ring):
                atom_pos = conf.GetAtomPosition(aromatic_ring_ids[i*num_mem_ring + j])
                new_coords[0] += atom_pos[0]
                new_coords[1] += atom_pos[1]
                new_coords[2] += atom_pos[2]
            new_coords = [new_coords[0]/num_mem_ring, new_coords[1]/num_mem_ring, new_coords[2]/num_mem_ring]
            self.features["aromatic"].append(Geometry.rdGeometry.Point3D(new_coords[0], new_coords[1], new_coords[2]))

    def GeneratePharmacophore(self):
        pharmacophore = Chem.rdchem.RWMol()
        pharm_coords = Chem.rdchem.Conformer(self.GetTotalNumFeatures())
        features_keys = list(self.features)
        feature_to_atom_num = {
            'HBA': 6,
            'wHBA': 7,
            'HBD': 8,
            'cation': 9,
            'anion': 15,
            'aromatic': 16,
        }
        feat_index = 0
        for i in range(len(self.features)):
            feat_key = features_keys[i]
            for j in range(len(self.features[feat_key])):
                pharmacophore.AddAtom(Chem.Atom(feature_to_atom_num[feat_key]))
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

    def DisplayPharm4(self, display_mol=True):
        view = py3Dmol.view(width=800)
        if display_mol:
            view = py3Dmol.view(data=Chem.rdmolfiles.MolToMolBlock(remove_nonpolar_hs(self.mol), includeStereo=True), style={'stick':{'colorscheme':'grayCarbon'}}, width=800)
        feature_to_color_and_radius = {
            'HBA': ('blue', 0.8),
            'wHBA': ('lightblue', 0.8),
            'HBD': ('darkorange', 0.8),
            'cation': ('purple', 0.8),
            'anion': ('black', 0.8),
            'aromatic': ('green', 1.3),
        }

        features_keys = list(self.features)
        for i in range(len(self.features)):
            feat_key = features_keys[i]
            for j in range(len(self.features[feat_key])):
                color, radius = feature_to_color_and_radius[feat_key]
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
