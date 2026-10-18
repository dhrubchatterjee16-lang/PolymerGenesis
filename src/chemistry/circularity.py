"""
Advanced Circular Economy, Dynamic Vitrimer Kinetics & Bio-Based LCA Engine for PolymerGenesis.
Implements:
1. Dynamic Covalent Adaptable Networks (CANs) & Vitrimer Arrhenius Rheology (Tv, Ea, Reprocessing Window).
2. Dainton-Ivin Quantitative Depolymerization Thermodynamics (Tc, [M]_eq, CRM feasibility).
3. ASTM D6866 Bio-Based Carbon Fraction & Cradle-to-Gate Net GWP-100 LCA Footprint.
4. Green Bio-Monomer Drop-in Substitution Engine (FDCA, Isosorbide, Sebacic Acid).
"""

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from ..features.featurizer import smiles_to_mol

# Gas constant R in J / (mol * K)
R_GAS = 8.314462618

# Dynamic Covalent Network & Chemical Recycling Exchange Motifs
VITRIMER_MOTIFS = {
    "Polyester Solvolysis / Transesterification": {
        "smarts": "[#6](=[OX1])[OX2][#6]", # Ester linkage
        "exchange_type": "Solvolysis / Transesterification (Thermoplastic)",
        "catalyst": "Zinc(II) acetate / Glycol transesterification at 190 °C",
        "Ea_kJ_mol": 85.0, # Activation energy in kJ/mol
        "tau_0_s": 1.0e-11, # Characteristic bond exchange attempt frequency
        "degradability_score": 88.0,
        "is_covalent_network": False,
        "mechanism": "Chemical Glycolysis / Hydrolysis to Monomers",
        "eol_class": "Closed-Loop Chemically Recyclable Polyester"
    },
    "Transimination (Schiff Base Vitrimer)": {
        "smarts": "[#6]=[NX2]",
        "exchange_type": "Associative CAN (Vitrimer)",
        "catalyst": "Catalyst-free or mild Brønsted acid buffer",
        "Ea_kJ_mol": 45.0,
        "tau_0_s": 5.0e-11,
        "degradability_score": 96.0,
        "is_covalent_network": True,
        "mechanism": "Dynamic Amine-Imine Metathesis",
        "eol_class": "Room-Temperature Dynamic Vitrimer Network"
    },
    "Boronic Ester Vitrimer": {
        "smarts": "[#5][OX2][#6]",
        "exchange_type": "Associative CAN (Vitrimer)",
        "catalyst": "Water-triggered / Lewis acid-free",
        "Ea_kJ_mol": 32.0,
        "tau_0_s": 1.0e-10,
        "degradability_score": 98.0,
        "is_covalent_network": True,
        "mechanism": "Diol-Dioxaborolane Dynamic Exchange",
        "eol_class": "Self-Healing Rapid Associative Vitrimer"
    },
    "Dynamic Disulfide Network": {
        "smarts": "[#16X2][#16X2]",
        "exchange_type": "Associative CAN (Vitrimer)",
        "catalyst": "Thiolate catalyst or UV / mild heat",
        "Ea_kJ_mol": 60.0,
        "tau_0_s": 2.0e-11,
        "degradability_score": 95.0,
        "is_covalent_network": True,
        "mechanism": "Thiol-Disulfide Dynamic Metathesis",
        "eol_class": "Dynamic Vitrimer Elastomer"
    },
    "Diels-Alder / Retro-DA": {
        "smarts": "[#6]1-[#6]=[#6]-[#6]-[#8,#7]-[#6]1", # Furan-Maleimide adduct
        "exchange_type": "Dissociative CAN (Reversible Gel-Sol)",
        "catalyst": "Thermal activation (110-140 °C)",
        "Ea_kJ_mol": 110.0,
        "tau_0_s": 1.0e-12,
        "degradability_score": 90.0,
        "is_covalent_network": True,
        "mechanism": "Reversible Retro-Diels-Alder Cycloaddition",
        "eol_class": "Thermally Reversible Crosslinked Network"
    },
    "Carbonate Solvolysis": {
        "smarts": "[OX2][#6](=[OX1])[OX2]",
        "exchange_type": "Solvolytic Alcoholysis (Thermoplastic)",
        "catalyst": "DMAP / Zinc acetate at 120 °C",
        "Ea_kJ_mol": 78.0,
        "tau_0_s": 1.0e-11,
        "degradability_score": 94.0,
        "is_covalent_network": False,
        "mechanism": "Catalytic Alcoholysis to Diols and Dialkyl Carbonates",
        "eol_class": "Chemically Depolymerizable Polycarbonate"
    },
    "Urethane Transcarbamoylation": {
        "smarts": "[#6][OX2][#6](=[OX1])[NX3]",
        "exchange_type": "Associative Carbamate Exchange",
        "catalyst": "Tin(II) octoate or DBU at 140 °C",
        "Ea_kJ_mol": 95.0,
        "tau_0_s": 1.0e-11,
        "degradability_score": 84.0,
        "is_covalent_network": True,
        "mechanism": "Transcarbamoylation / Glycolytic Solvolysis",
        "eol_class": "Recyclable Polyurethane Vitrimer"
    },
    "Acetal / Ketal Hydrolysis": {
        "smarts": "[OX2][#6][OX2]",
        "exchange_type": "Acid-Triggered Dissociative",
        "catalyst": "Dilute aqueous acid (pH 2-4)",
        "Ea_kJ_mol": 40.0,
        "tau_0_s": 1.0e-10,
        "degradability_score": 97.0,
        "is_covalent_network": False,
        "mechanism": "Mild Acidic Acetal Hydrolysis to Aldehydes and Diols",
        "eol_class": "Acid-Triggered Closed-Loop Depolymerizable"
    }
}

# Thermodynamic group-contribution parameters for Ceiling Temperature Tc
# Delta_Hp (kJ/mol), Delta_Sp (J / (mol * K))
THERMO_CLASSES = [
    {
        "name": "5-Membered Cyclic Lactone / Reversible CRM",
        "smarts": "[#6]1-[#6]-[#6]-[#6](=[OX1])[OX2]1", # gamma-butyrolactone
        "delta_Hp": -5.4, # kJ/mol
        "delta_Sp": -28.4, # J/(mol*K)
        "mechanism": "Complete Catalytic Unzipping to Monomer (100% Atom Economy)",
        "crm_class": "Ultra-High CRM Feasibility (Depolymerizable at 120-160 °C)"
    },
    {
        "name": "Aliphatic Polyester (PLA / Polycaprolactone)",
        "smarts": "[#6][#6](=[OX1])[OX2][#6]",
        "delta_Hp": -22.0,
        "delta_Sp": -65.0,
        "mechanism": "Catalytic Transesterification / Ring-Closing Unzipping",
        "crm_class": "Solvolytic & Thermal CRM (T_c ~ 350-400 K)"
    },
    {
        "name": "Polycarbonate",
        "smarts": "[OX2][#6](=[OX1])[OX2]",
        "delta_Hp": -30.0,
        "delta_Sp": -80.0,
        "mechanism": "Methanolysis / Phenol Exchange",
        "crm_class": "High-Yield Chemical Depolymerization"
    },
    {
        "name": "Vinyl / Addition Polymer",
        "smarts": "[*][#6][#6][*]",
        "delta_Hp": -75.0,
        "delta_Sp": -110.0,
        "mechanism": "Thermal Radical Pyrolysis / Monomer Distillation",
        "crm_class": "Energy-Intensive Thermal Cracking (T_c > 650 K)"
    }
]

# Green Bio-Monomer Drop-in Substitutions
BIO_SUBSTITUTIONS = [
    {
        "target": "Aromatic Phthalate / Terephthalate (PTA)",
        "smarts": "c1ccc(C(=O)O)cc1C(=O)O",
        "green_alternative": "2,5-Furandicarboxylic Acid (FDCA)",
        "green_smiles": "O=C(O)c1ccc(C(=O)O)o1",
        "origin": "Lignocellulosic biomass (HMF oxidation)",
        "benefits": "Higher Tg (+10 K), 10x lower O2 permeability, -45% cradle-to-gate CO2-eq"
    },
    {
        "target": "Bisphenol A (BPA)",
        "smarts": "Oc1ccc(C(C)(C)c2ccc(O)cc2)cc1",
        "green_alternative": "Isosorbide (1,4:3,6-Dianhydro-D-glucitol)",
        "green_smiles": "OC1C2OCC(O)C2OC1",
        "origin": "Corn starch / D-glucose catalytic dehydration",
        "benefits": "Zero endocrine disruption, non-toxic, high rigidity (Tg > 130 °C)"
    },
    {
        "target": "Petrochemical Adipic Acid",
        "smarts": "OC(=O)CCCCC(=O)O",
        "green_alternative": "Bio-Sebacic Acid",
        "green_smiles": "OC(=O)CCCCCCCCC(=O)O",
        "origin": "Castor oil cleavage",
        "benefits": "100% Biogenic carbon fraction (ASTM D6866), enhanced hydrophobicity"
    }
]


def calculate_vitrimer_kinetics(Ea_kJ_mol: float, tau_0_s: float = 1e-11) -> dict:
    """
    Computes Arrhenius network relaxation and Topology Freezing Transition Temperature (Tv):
    Tv is defined at tau*(Tv) = 10^5 seconds (Maxwell viscosity eta ~ 10^12 Pa*s).
    Reprocessing temp T_reprocess is defined where tau* <= 100 seconds (eta ~ 10^4 Pa*s).
    """
    # Tv = Ea / (R * ln(10^5 / tau_0))
    ln_ratio_tv = math.log(1.0e5 / tau_0_s)
    Tv_K = (Ea_kJ_mol * 1000.0) / (R_GAS * ln_ratio_tv)

    # Reprocessing temperature (tau* = 100 s)
    ln_ratio_reproc = math.log(100.0 / tau_0_s)
    T_reproc_K = (Ea_kJ_mol * 1000.0) / (R_GAS * ln_ratio_reproc)

    service_window_K = max(0.0, T_reproc_K - Tv_K)
    malleability_score = float(np.clip(100.0 - (Ea_kJ_mol / 1.5), 10.0, 99.0))

    return {
        "Tv_K": float(Tv_K),
        "Tv_C": float(Tv_K - 273.15),
        "T_reproc_K": float(T_reproc_K),
        "T_reproc_C": float(T_reproc_K - 273.15),
        "service_window_K": float(service_window_K),
        "malleability_score": malleability_score
    }


def calculate_dainton_ivin_thermodynamics(delta_Hp: float, delta_Sp: float, m0_conc: float = 5.0) -> dict:
    """
    Computes Dainton-Ivin ceiling temperature Tc and equilibrium monomer concentration [M]_eq:
    Tc = Delta_Hp / (Delta_Sp + R * ln([M]_0))
    [M]_eq(T) = exp( Delta_Hp / (R*T) - Delta_Sp / R )
    """
    # Tc in Kelvin
    denominator = delta_Sp + R_GAS * math.log(max(1e-3, m0_conc))
    if abs(denominator) > 1e-5:
        Tc_K = (delta_Hp * 1000.0) / denominator
    else:
        Tc_K = 350.0

    # Ensure positive physical absolute temperature
    Tc_K = float(np.clip(Tc_K, 150.0, 850.0))

    # [M]_eq at standard recycling temperature T = 200 °C = 473.15 K
    T_recycle = 473.15
    ln_m_eq = (delta_Hp * 1000.0) / (R_GAS * T_recycle) - (delta_Sp / R_GAS)
    m_eq_200C = math.exp(np.clip(ln_m_eq, -10.0, 10.0))

    return {
        "Tc_K": float(Tc_K),
        "Tc_C": float(Tc_K - 273.15),
        "delta_Hp_kJ_mol": float(delta_Hp),
        "delta_Sp_J_mol_K": float(delta_Sp),
        "equilibrium_monomer_conc_200C_mol_L": float(m_eq_200C)
    }


# Verified Biogenic Monomer Cores for ASTM D6866 Radiocarbon Mass Balance
VERIFIED_BIOGENIC_MOTIFS = [
    # 2,5-Furandicarboxylic acid (FDCA) / Furan ring from carbohydrate HMF dehydration
    {"name": "2,5-FDCA Diacid Synthon", "smarts": "[#6](=[#8])c1ccc(o1)[#6](=[#8])"},
    {"name": "FDCA / Furan Ring", "smarts": "o1cccc1"},
    # Isosorbide / 1,4:3,6-dianhydrohexitol core from starch/glucose
    {"name": "Isosorbide / Starch Bicyclic Core", "smarts": "C1C2OCCC2OC1"},
    # Bio-Sebacic acid / Ricinoleic acid derivatives (C10 linear aliphatic core)
    {"name": "Sebacate / Castor Oil C10 Synthon", "smarts": "[#6](=[#8])[#6X4][#6X4][#6X4][#6X4][#6X4][#6X4][#6X4][#6X4][#6](=[#8])"},
    # Polylactic acid (PLA) / Lactide synthon (fermentation-derived alpha-hydroxypropionate)
    {"name": "Lactide / PLA Repeat Unit", "smarts": "[#6X4;H3][#6X4;H1]([#8])[#6](=[#8])"},
    # Bio-Succinic acid / bio-1,4-butanediol core (aliphatic C4 diacid)
    {"name": "Bio-Succinate / BDO Core", "smarts": "[#6](=[#8])[#6X4][#6X4][#6](=[#8])"},
    # Itaconic acid motif (fermentation of Aspergillus terreus)
    {"name": "Itaconate Motif", "smarts": "C(=C)(C(=O))CC(=O)"},
    # Vanillin / Lignin-derived methoxyphenol synthon
    {"name": "Lignin Vanillyl / Guaiacyl", "smarts": "c1cc(O)c(OC)cc1"}
]


def calculate_bio_carbon_and_lca(mol) -> dict:
    """
    Evaluates bio-based carbon fraction f_bio_C pursuant to ASTM D6866 radiocarbon
    mass balance (preventing false-positive petroleum polyester scores) and
    computes cradle-to-gate net Global Warming Potential (GWP-100 in kg CO2-eq / kg polymer).
    """
    if mol is None:
        return {"f_bio_C": 0.0, "f_bio_C_percent": 0.0, "net_GWP_kg_CO2_kg": 3.5, "bio_substitutions": []}

    num_total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    if num_total_carbons == 0:
        return {"f_bio_C": 0.0, "f_bio_C_percent": 0.0, "net_GWP_kg_CO2_kg": 3.0, "bio_substitutions": []}

    # Match verified biogenic synthon atom indices (preventing double-counting)
    bio_carbon_atom_indices = set()
    for b_motif in VERIFIED_BIOGENIC_MOTIFS:
        pat = Chem.MolFromSmarts(b_motif["smarts"])
        if pat and mol.HasSubstructMatch(pat):
            for match in mol.GetSubstructMatches(pat):
                for idx in match:
                    atom = mol.GetAtomWithIdx(idx)
                    if atom.GetAtomicNum() == 6:
                        bio_carbon_atom_indices.add(idx)

    # Check green drop-in substitutions
    substitutions_identified = []
    for sub in BIO_SUBSTITUTIONS:
        pat = Chem.MolFromSmarts(sub["smarts"])
        if pat and mol.HasSubstructMatch(pat):
            substitutions_identified.append(sub)

    num_bio_carbons = len(bio_carbon_atom_indices)
    f_bio_C = float(np.clip(num_bio_carbons / num_total_carbons, 0.0, 1.0))

    # Weight fraction of carbon in polymer
    mol_wt = Descriptors.MolWt(mol)
    carbon_wt_fraction = (num_total_carbons * 12.011) / max(mol_wt, 1e-4)

    # Net GWP calculation (kg CO2-eq / kg polymer):
    # Fossil cradle-to-gate benchmark ~ 3.80 kg CO2-eq/kg (feedstock + polymerization)
    # Biogenic carbon uptake factor = 44.01 / 12.011 = 3.664 kg CO2 captured per kg bio-carbon
    gwp_fossil_baseline = 3.80
    biogenic_co2_offset = 3.664 * f_bio_C * carbon_wt_fraction
    # Net GWP subtracts the biogenic carbon sequestration from the fossil baseline
    net_gwp = float(max(0.2, gwp_fossil_baseline - biogenic_co2_offset))

    return {
        "f_bio_C": f_bio_C,
        "f_bio_C_percent": float(f_bio_C * 100.0),
        "net_GWP_kg_CO2_kg": net_gwp,
        "biogenic_CO2_sequestered_kg": float(biogenic_co2_offset),
        "bio_substitutions": substitutions_identified
    }


def analyze_circularity(smiles: str) -> dict:
    """
    Evaluates the full circular lifecycle:
    1. Dynamic Covalent / Vitrimer Kinetics (Tv, Ea, Reprocessing Window)
    2. Dainton-Ivin Depolymerization Thermodynamics (Tc, [M]_eq)
    3. ASTM D6866 Bio-Based Carbon & Net Cradle-to-Gate GWP-100 LCA
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "circularity_score": 0.0,
            "detected_linkages": [],
            "depolymerization_mechanism": "Unknown",
            "recommended_conditions": "N/A",
            "eol_classification": "Non-Recyclable",
            "vitrimer_kinetics": None,
            "thermodynamics": None,
            "lca": None
        }

    detected_linkages = []
    max_score = 25.0 # Baseline for recalcitrant polyolefin backbone
    best_mechanism = "Thermal Pyrolysis / Mechanical Downcycling"
    best_conditions = "High-temperature cracking (>450 °C) with high carbon penalty"
    best_class = "Mechanically Downcyclable Thermoplastic (Low Circularity)"
    active_vitrimer = None

    # 1. Match Dynamic & Vitrimer Linkages
    for name, info in VITRIMER_MOTIFS.items():
        pattern = Chem.MolFromSmarts(info["smarts"])
        if pattern and mol.HasSubstructMatch(pattern):
            num_matches = len(mol.GetSubstructMatches(pattern))
            detected_linkages.append({
                "type": name,
                "count": num_matches,
                "score": info["degradability_score"],
                "exchange_type": info["exchange_type"],
                "catalyst": info["catalyst"]
            })
            if info["degradability_score"] > max_score:
                max_score = info["degradability_score"]
                best_mechanism = info["mechanism"]
                best_conditions = info["catalyst"]
                best_class = info["eol_class"]
                active_vitrimer = info

    # 2. Compute Vitrimer Kinetics if dynamic linkage present
    if active_vitrimer:
        vitrimer_data = calculate_vitrimer_kinetics(active_vitrimer["Ea_kJ_mol"], active_vitrimer["tau_0_s"])
        vitrimer_data["is_covalent_network"] = active_vitrimer.get("is_covalent_network", True)
        if not vitrimer_data["is_covalent_network"]:
            vitrimer_data["reprocessing_mechanism"] = "Thermoplastic melt flow / Solvolytic CRM"
        else:
            vitrimer_data["reprocessing_mechanism"] = "Associative / Dissociative Network Bond Exchange"
    else:
        # Static baseline
        vitrimer_data = calculate_vitrimer_kinetics(Ea_kJ_mol=140.0, tau_0_s=1e-12)
        vitrimer_data["is_covalent_network"] = False
        vitrimer_data["reprocessing_mechanism"] = "Non-reversible covalent network or recalcitrant polyolefin"

    # 3. Match Dainton-Ivin Thermodynamics
    thermo_match = THERMO_CLASSES[-1] # default addition polymer
    for t_class in THERMO_CLASSES:
        p = Chem.MolFromSmarts(t_class["smarts"])
        if p and mol.HasSubstructMatch(p):
            thermo_match = t_class
            break

    thermo_data = calculate_dainton_ivin_thermodynamics(
        delta_Hp=thermo_match["delta_Hp"],
        delta_Sp=thermo_match["delta_Sp"]
    )
    thermo_data["crm_class"] = thermo_match["crm_class"]
    thermo_data["thermo_mechanism"] = thermo_match["mechanism"]

    # 4. Compute ASTM D6866 Bio-Carbon & GWP-100 LCA
    lca_data = calculate_bio_carbon_and_lca(mol)

    return {
        "circularity_score": float(max_score),
        "detected_linkages": detected_linkages,
        "depolymerization_mechanism": best_mechanism,
        "recommended_conditions": best_conditions,
        "eol_classification": best_class,
        "vitrimer_kinetics": vitrimer_data,
        "thermodynamics": thermo_data,
        "lca": lca_data
    }

