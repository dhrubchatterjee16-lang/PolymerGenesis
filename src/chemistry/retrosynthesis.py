"""
Retrosynthetic Reaction Planner for PolymerGenesis (Upgraded & Scientifically Rigorous).
Accurately deconstructs:
1. Vinyl / Addition polymers -> Exact vinyl monomer (e.g. Styrene, Methyl Methacrylate) via double-bond restoration.
2. Polyesters -> Dicarboxylic acid + Diol precursors via ester bond hydrolysis.
3. Polyamides -> Dicarboxylic acid + Diamine precursors via amide bond hydrolysis.
4. Polycarbonates -> Diol/Bisphenol + Carbonyl source (Diphenyl Carbonate / Phosgene).
"""

from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from ..features.featurizer import smiles_to_mol


def deconstruct_addition_vinyl(mol: Chem.Mol) -> Optional[str]:
    """
    Restores the double bond of addition/vinyl monomers (*-C-C-* -> C=C).
    Example: polystyrene repeat unit [*]CC([*])c1ccccc1 -> C=Cc1ccccc1 (Styrene).
    """
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) < 2:
        return None

    # Find the neighbor atoms attached to the two dummy connection points
    n1 = dummies[0].GetNeighbors()
    n2 = dummies[1].GetNeighbors()
    if not n1 or not n2:
        return None

    a1 = n1[0]
    a2 = n2[0]

    # Verify if a1 and a2 are directly bonded along the polymer backbone
    bond = mol.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
    if bond is None:
        return None

    try:
        rw = Chem.RWMol(mol)
        # Disconnect dummy atoms
        rw.RemoveBond(dummies[0].GetIdx(), a1.GetIdx())
        rw.RemoveBond(dummies[1].GetIdx(), a2.GetIdx())
        # Set double bond between backbone carbons
        b = rw.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
        b.SetBondType(Chem.rdchem.BondType.DOUBLE)
        # Remove dummy atoms in reverse order
        for idx in sorted([dummies[0].GetIdx(), dummies[1].GetIdx()], reverse=True):
            rw.RemoveAtom(idx)

        res = rw.GetMol()
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(res)
    except Exception:
        return None


def cap_and_cleave_condensation(mol: Chem.Mol, rxn_smarts: str) -> List[str]:
    """
    Intelligently caps polymer chain ends (* attached to C(=O) becomes -OH, * attached to O/N becomes -H)
    and cleaves condensation linkages (esters, amides) into monomer precursors.
    """
    rw = Chem.RWMol(mol)
    dummies = [a for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

    for d in dummies:
        neighbors = d.GetNeighbors()
        if not neighbors:
            continue
        n = neighbors[0]
        is_carbonyl = False
        if n.GetAtomicNum() == 6:
            for nb in n.GetNeighbors():
                b = rw.GetBondBetweenAtoms(n.GetIdx(), nb.GetIdx())
                if nb.GetAtomicNum() == 8 and b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    is_carbonyl = True
                    break
        if is_carbonyl:
            d.SetAtomicNum(8) # Carbonyl terminal becomes carboxylic acid -OH
        else:
            d.SetAtomicNum(1) # Heteroatom/aliphatic terminal becomes -H

    try:
        capped = rw.GetMol()
        Chem.SanitizeMol(capped)

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        products = rxn.RunReactants((capped,))
        if products and len(products[0]) >= 2:
            monomers = []
            for p in products[0]:
                p_clean = Chem.RemoveHs(p)
                monomers.append(Chem.MolToSmiles(p_clean))
            return sorted(list(set(monomers)))
        else:
            capped_clean = Chem.RemoveHs(capped)
            return [Chem.MolToSmiles(capped_clean)]
    except Exception:
        # Fallback to simple wildcard removal
        clean_s = Chem.MolToSmiles(mol).replace('[*]', '').replace('*', '')
        return [clean_s]


def plan_retrosynthesis(smiles: str) -> dict:
    """
    Deconstructs a polymer repeat unit into precursor monomers and laboratory synthesis protocols.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}

    # Chemical class detection
    ester_smarts = Chem.MolFromSmarts("[#6](=[OX1])[OX2]")
    amide_smarts = Chem.MolFromSmarts("[#6](=[OX1])[NX3]")
    carbonate_smarts = Chem.MolFromSmarts("[OX2][#6](=[OX1])[OX2]")
    vinyl_smarts = Chem.MolFromSmarts("[*][#6][#6][*]")

    has_carbonate = mol.HasSubstructMatch(carbonate_smarts)
    has_ester = mol.HasSubstructMatch(ester_smarts) and not has_carbonate
    has_amide = mol.HasSubstructMatch(amide_smarts)
    has_vinyl = mol.HasSubstructMatch(vinyl_smarts) and not (has_ester or has_amide or has_carbonate)

    if has_carbonate:
        diol_monomers = cap_and_cleave_condensation(
            mol,
            "[OX2:1][#6:2](=[OX1:3])[OX2:4]>>[OX2H:1].[OX2H:4]"
        )
        precursors = [{"name": f"Diol Precursor ({m})", "structure": m, "role": "Nucleophilic Diol"} for m in diol_monomers]
        precursors.append({"name": "Diphenyl Carbonate (DPC)", "structure": "c1ccccc1OC(=O)Oc2ccccc2", "role": "Carbonyl Source"})
        return {
            "polymerization_type": "Step-Growth Melt Transesterification / Polycondensation",
            "mechanism": "Nucleophilic Acyl Substitution",
            "precursor_monomers": precursors,
            "laboratory_protocol": {
                "initiator_catalyst": "Sodium hydroxide (NaOH) / Zinc acetate / DMAP",
                "solvent": "Melt phase (solventless) or Dichloromethane / Pyridine",
                "temperature": "180 - 240 °C under progressive vacuum (< 1 mbar)",
                "byproduct": "Phenol / H2O (condensed and collected under vacuum)"
            }
        }

    elif has_ester:
        # Hydrolyze ester: C(=O)-O -> C(=O)OH + HO-
        precursor_smiles = cap_and_cleave_condensation(
            mol,
            "[#6:1](=[OX1:2])[OX2:3][#6:4]>>[#6:1](=[OX1:2])[OH].[OH][#6:4]"
        )
        precursors = []
        for p in precursor_smiles:
            role = "Dicarboxylic Acid / Core" if "(=O)O" in p else "Diol / Chain Extender"
            precursors.append({"name": f"Monomer Precursor ({p})", "structure": p, "role": role})

        return {
            "polymerization_type": "Step-Growth Polycondensation / Transesterification",
            "mechanism": "Melt Polycondensation",
            "precursor_monomers": precursors,
            "laboratory_protocol": {
                "initiator_catalyst": "Titanium(IV) butoxide (Ti(OBu)4) or Zinc acetate",
                "solvent": "Bulk melt polymerization (inert N2 atmosphere)",
                "temperature": "190 - 250 °C with continuous vacuum removal of byproducts",
                "byproduct": "Water (H2O) or Methanol (MeOH)"
            }
        }

    elif has_amide:
        # Hydrolyze amide: C(=O)-N -> C(=O)OH + H2N-
        precursor_smiles = cap_and_cleave_condensation(
            mol,
            "[#6:1](=[OX1:2])[NX3:3][#6:4]>>[#6:1](=[OX1:2])[OH].[NH2][#6:4]"
        )
        precursors = []
        for p in precursor_smiles:
            role = "Dicarboxylic Acid Precursor" if "(=O)O" in p else "Diamine Precursor"
            precursors.append({"name": f"Polyamide Monomer ({p})", "structure": p, "role": role})

        return {
            "polymerization_type": "Step-Growth Thermal Polycondensation",
            "mechanism": "Nucleophilic Amidation",
            "precursor_monomers": precursors,
            "laboratory_protocol": {
                "initiator_catalyst": "Thermal self-catalyzed or Triphenylphosphite / Pyridine",
                "solvent": "NMP (N-Methyl-2-pyrrolidone) with LiCl or Melt salt",
                "temperature": "200 - 280 °C under inert Argon atmosphere",
                "byproduct": "Water (H2O)"
            }
        }

    elif has_vinyl:
        vinyl_monomer = deconstruct_addition_vinyl(mol)
        if not vinyl_monomer:
            clean_s = Chem.MolToSmiles(mol).replace('[*]', '').replace('*', '')
            vinyl_monomer = f"C={clean_s}"

        return {
            "polymerization_type": "Chain-Growth Addition Polymerization",
            "mechanism": "Reversible Addition-Fragmentation Chain Transfer (RAFT) / Free Radical",
            "precursor_monomers": [
                {"name": f"Vinyl Monomer ({vinyl_monomer})", "structure": vinyl_monomer, "role": "Addition Monomer"}
            ],
            "laboratory_protocol": {
                "initiator_catalyst": "AIBN (Azobisisobutyronitrile) or Benzoyl peroxide (BPO)",
                "solvent": "Toluene / Anisole or Bulk",
                "temperature": "65 - 80 °C for 8-16 hours under degassed N2",
                "byproduct": "None (100% Atom Economy Addition)"
            }
        }

    else:
        # Polyether or other condensation backbone
        clean_s = Chem.MolToSmiles(mol).replace('[*]', '').replace('*', '')
        return {
            "polymerization_type": "Step-Growth Condensation / Polyetherification",
            "mechanism": "Nucleophilic Substitution",
            "precursor_monomers": [
                {"name": f"Functional Monomer ({clean_s})", "structure": clean_s, "role": "Monomer"}
            ],
            "laboratory_protocol": {
                "initiator_catalyst": "Potassium carbonate / Palladium catalyst",
                "solvent": "DMF / DMSO / Toluene",
                "temperature": "110 - 150 °C",
                "byproduct": "Halide salts / Water"
            }
        }
