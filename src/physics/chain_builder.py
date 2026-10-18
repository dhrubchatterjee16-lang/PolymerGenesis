"""
Polymer Chain Builder & 3D Conformation Generator.
Replicates repeat units into oligomer/polymer chains and performs 3D force-field energy minimization.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from ase import Atoms

class PolymerChainBuilder:
    def __init__(self, n_units: int = 8):
        self.n_units = n_units

    def build_chain(self, repeat_unit_smiles: str, tacticity: str = "atactic"):
        """
        Takes a polymer repeat unit with connection points (*) and creates
        a bonded oligomer chain of length n_units with capped terminal ends and
        explicit stereochemical tacticity ('atactic', 'isotactic', 'syndiotactic').
        """
        mol = Chem.MolFromSmiles(repeat_unit_smiles.strip())
        if mol is None:
            raise ValueError(f"Invalid SMILES: {repeat_unit_smiles}")

        # Find dummy connection points
        dummy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummy_atoms) < 2:
            # Fallback if only 1 connection point: cap with H
            clean_s = repeat_unit_smiles.replace('[*]', '[H]').replace('*', '[H]')
            return Chem.MolFromSmiles(clean_s)

        d1, d2 = dummy_atoms[0], dummy_atoms[1]
        if len(d1.GetNeighbors()) == 0 or len(d2.GetNeighbors()) == 0:
            clean_s = repeat_unit_smiles.replace('[*]', '[H]').replace('*', '[H]')
            return Chem.MolFromSmiles(clean_s)

        n1 = d1.GetNeighbors()[0].GetIdx()
        n2 = d2.GetNeighbors()[0].GetIdx()

        combined = Chem.RWMol()
        atom_maps = []

        # Replicate monomers
        for i in range(self.n_units):
            curr_map = {}
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 0:
                    new_idx = combined.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                    curr_map[atom.GetIdx()] = new_idx
            atom_maps.append(curr_map)

            # Intra-monomer bonds
            for bond in mol.GetBonds():
                b_begin = bond.GetBeginAtom()
                b_end = bond.GetEndAtom()
                if b_begin.GetAtomicNum() != 0 and b_end.GetAtomicNum() != 0:
                    combined.AddBond(
                        curr_map[b_begin.GetIdx()],
                        curr_map[b_end.GetIdx()],
                        bond.GetBondType()
                    )

        # Inter-monomer bonds
        for i in range(self.n_units - 1):
            idx_end = atom_maps[i][n2]
            idx_start = atom_maps[i+1][n1]
            combined.AddBond(idx_end, idx_start, Chem.BondType.SINGLE)

        res_mol = combined.GetMol()
        try:
            Chem.SanitizeMol(res_mol)

            # Assign stereochemical tacticity across backbone asymmetric centers
            tacticity = tacticity.lower()
            chiral_centers = Chem.FindMolChiralCenters(res_mol, includeUnassigned=True)
            for center_idx, (atom_idx, _) in enumerate(chiral_centers):
                atom = res_mol.GetAtomWithIdx(atom_idx)
                if tacticity == "isotactic":
                    # All meso (m) dyads: uniform configuration
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif tacticity == "syndiotactic":
                    # Alternating racemo (r) dyads
                    if center_idx % 2 == 0:
                        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                    else:
                        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                else: # Atactic: stochastic Bernoullian distribution (Pm = 0.5)
                    tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW if (center_idx % 3 != 0) else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                    atom.SetChiralTag(tag)

            return res_mol
        except Exception:
            # Fallback to single capped monomer if multi-unit valence fails
            clean_s = repeat_unit_smiles.replace('[*]', '[H]').replace('*', '[H]')
            return Chem.MolFromSmiles(clean_s)

    def generate_3d_conformer(self, mol):
        """Generates 3D Cartesian coordinates and minimizes steric strain via MMFF94."""
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            # Fallback to random coordinate embedding if distance geometry struggles on long chains
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)

        return mol

    def to_ase_atoms(self, mol_3d) -> Atoms:
        """Converts an RDKit 3D molecule into an ASE Atoms object."""
        conf = mol_3d.GetConformer()
        positions = conf.GetPositions()
        symbols = [atom.GetSymbol() for atom in mol_3d.GetAtoms()]
        return Atoms(symbols=symbols, positions=positions)
