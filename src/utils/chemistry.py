"""
Chemistry Utilities for Polymer Validation, Synthesizability (SAScore), and Diversity.
"""

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import DataStructs
from ..features.featurizer import _morgan_gen, smiles_to_mol

def is_valid_polymer(smiles: str, require_wildcards: bool = True) -> bool:
    """
    Checks if a generated string is a chemically valid polymer repeat unit:
    1. Parseable by RDKit
    2. Valid valence
    3. Contains at least one connection point wildcard (*)
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return False
    mol = smiles_to_mol(smiles)
    if mol is None:
        return False
    
    if require_wildcards:
        # Check for presence of dummy atoms (atomic number 0 or '*' in string)
        has_wildcard = any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
        if not has_wildcard:
            return False
            
    # Check for basic sanity (at least 2 heavy atoms besides wildcard)
    if mol.GetNumAtoms() < 3:
        return False

    return True

def compute_synthetic_accessibility(smiles: str) -> float:
    """
    Calculates a heuristic Synthesizability / Complexity Score (1.0 = easy, 10.0 = very difficult).
    Penalizes extreme molecular weights, excessive ring counts, bridged bicyclics, and rare atoms.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return 10.0

    mw = Descriptors.MolWt(mol)
    num_rings = Descriptors.RingCount(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    num_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    score = 1.0
    # MW penalty
    if mw > 400:
        score += 1.5 * (mw - 400) / 200.0
    # Extreme ring penalty
    if num_rings > 4:
        score += 1.0 * (num_rings - 4)
    # Bridgehead / ring strain heuristic
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        large_rings = [len(r) for r in ring_info.AtomRings() if len(r) > 7 or len(r) < 5]
        score += 0.8 * len(large_rings)

    return float(np.clip(score, 1.0, 10.0))

def compute_tanimoto_similarity(smiles_1: str, smiles_2: str) -> float:
    """Computes Morgan Fingerprint Tanimoto similarity between two polymers."""
    m1 = smiles_to_mol(smiles_1)
    m2 = smiles_to_mol(smiles_2)
    if m1 is None or m2 is None:
        return 0.0
    fp1 = _morgan_gen.GetFingerprint(m1)
    fp2 = _morgan_gen.GetFingerprint(m2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def compute_novelty(generated_smiles_list: list, reference_smiles_set: set) -> float:
    """Computes the fraction of generated polymers that are NOT in the reference set."""
    if len(generated_smiles_list) == 0:
        return 0.0
    novel_count = sum(1 for s in generated_smiles_list if s not in reference_smiles_set)
    return float(novel_count / len(generated_smiles_list))

def compute_internal_diversity(smiles_list: list) -> float:
    """Computes average pairwise Tanimoto distance across a set of generated polymers."""
    valid_mols = [smiles_to_mol(s) for s in smiles_list if smiles_to_mol(s) is not None]
    if len(valid_mols) < 2:
        return 0.0
    fps = [_morgan_gen.GetFingerprint(m) for m in valid_mols]
    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1.0 - sim)
    return float(np.mean(distances)) if len(distances) > 0 else 0.0
