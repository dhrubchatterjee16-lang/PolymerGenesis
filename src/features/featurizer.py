"""
Polymer Featurization Module for PolymerGenesis.
Extracts:
1. Morgan Fingerprints (ECFP-equivalent)
2. Normalized RDKit 2D Physicochemical Descriptors
3. PyTorch Geometric Molecular Graph Representations with Atom and Bond features.
"""

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from torch_geometric.data import Data

# Disable noisy C++ parsing errors during generative exploration
RDLogger.DisableLog('rdApp.*')

# Allowed atomic numbers for one-hot encoding (0 represents wildcard/connection point [*] or *)
ALLOWED_ATOMS = [0, 1, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED
]

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_mol(smiles: str):
    """Parses polymer SMILES containing wildcards (*) or regular SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    # Ensure wildcard formatting is parseable by RDKit
    clean_smiles = smiles.strip()
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        # Fallback: try replacing bare '*' with '[*]'
        clean_smiles = clean_smiles.replace('*', '[*]').replace('[[*]]', '[*]')
        mol = Chem.MolFromSmiles(clean_smiles)
    return mol

def get_morgan_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray:
    """Computes a fixed-size Morgan fingerprint vector."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for idx in fp.GetOnBits():
        if idx < n_bits:
            arr[idx] = 1.0
    return arr

def get_rdkit_descriptors(smiles: str) -> np.ndarray:
    """Computes normalized 2D physicochemical descriptors for macromolecular repeat units."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(12, dtype=np.float32)
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        arom_rings = Descriptors.NumAromaticRings(mol)
        sat_rings = Descriptors.NumSaturatedRings(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        f_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        labute_asa = Descriptors.LabuteASA(mol)
        formal_charge = Chem.GetFormalCharge(mol)

        desc = [
            mw, logp, tpsa, float(hbd), float(hba),
            float(rot_bonds), float(arom_rings), float(sat_rings),
            float(heavy_atoms), float(f_sp3), labute_asa, float(formal_charge)
        ]
        return np.nan_to_num(np.array(desc, dtype=np.float32), nan=0.0)
    except Exception:
        return np.zeros(12, dtype=np.float32)

CHIRAL_TYPES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

def atom_features(atom, include_chirality: bool = False) -> list:
    """Extracts informative atom-level feature vector with optional stereochemical chirality."""
    atomic_num = atom.GetAtomicNum()
    # 1. One-hot atomic number
    atom_type = [1.0 if atomic_num == a else 0.0 for a in ALLOWED_ATOMS]
    if sum(atom_type) == 0:
        # Unknown atom
        atom_type[0] = 1.0

    # 2. Degree (0 to 6)
    deg = atom.GetTotalDegree()
    deg_one_hot = [1.0 if deg == d else 0.0 for d in range(7)]

    # 3. Hybridization
    hyb = atom.GetHybridization()
    hyb_one_hot = [1.0 if hyb == h else 0.0 for h in HYBRIDIZATIONS]

    # 4. Aromaticity, Formal charge, Radical electrons, Implicit H count
    misc = [
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetTotalNumHs())
    ]
    feats = atom_type + deg_one_hot + hyb_one_hot + misc

    # 5. Stereochemical tacticity & chirality (4 one-hot features)
    if include_chirality:
        chiral_tag = atom.GetChiralTag()
        chiral_one_hot = [1.0 if chiral_tag == c else 0.0 for c in CHIRAL_TYPES]
        feats += chiral_one_hot

    return feats

def bond_features(bond) -> list:
    """Extracts bond-level feature vector."""
    b_type = bond.GetBondType()
    b_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    b_one_hot = [1.0 if b_type == b else 0.0 for b in b_types]
    if sum(b_one_hot) == 0:
        b_one_hot[0] = 1.0

    misc = [
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0
    ]
    return b_one_hot + misc

def smiles_to_graph(smiles: str, y_val: float = None, include_chirality: bool = False) -> Data:
    """Converts a polymer repeat-unit SMILES into a PyTorch Geometric Data object."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom, include_chirality=include_chirality))
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edges
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = bond_features(bond)
        # Bidirectional graph edges
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([b_feat, b_feat])

    if len(edge_indices) == 0:
        # Isolated atom
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    if y_val is not None:
        data.y = torch.tensor([float(y_val)], dtype=torch.float)
    return data
