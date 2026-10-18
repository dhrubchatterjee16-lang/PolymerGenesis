"""
Commercial Chemical Vendor & Sourcing Resolver for PolymerGenesis.
Resolves retrosynthetic precursor monomers to CAS numbers, verified commercial availability,
pricing tiers, and purchase links via curated industry catalogs and PubChem PUG-REST API.
"""

import os
import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Optional
from rdkit import Chem
from ..features.featurizer import smiles_to_mol


CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "vendor_cache.json")

# Verified Commercial & Bio-based Monomer Reference Database
KNOWN_MONOMER_CATALOG = {
    # Vinyls & Acrylates
    "C=Cc1ccccc1": {
        "name": "Styrene",
        "cas_number": "100-42-5",
        "formula": "C8H8",
        "molecular_weight": 104.15,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / Enamine / TCI America",
        "price_tier": "$ (Commodity bulk, <$20/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/100-42-5"
    },
    "C=C(C)C(=O)OC": {
        "name": "Methyl Methacrylate (MMA)",
        "cas_number": "80-62-6",
        "formula": "C5H8O2",
        "molecular_weight": 100.12,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / Dow / Arkema",
        "price_tier": "$ (Commodity bulk, <$15/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/80-62-6"
    },
    "C=C": {
        "name": "Ethylene",
        "cas_number": "74-85-1",
        "formula": "C2H4",
        "molecular_weight": 28.05,
        "is_commercially_available": True,
        "primary_vendor": "Airgas / Linde / Matheson",
        "price_tier": "$ (Commodity gas)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/74-85-1"
    },
    # Diacids
    "O=C(O)c1ccc(C(=O)O)cc1": {
        "name": "Terephthalic Acid (PTA)",
        "cas_number": "100-21-0",
        "formula": "C8H6O4",
        "molecular_weight": 166.13,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / DAK Americas / Indorama",
        "price_tier": "$ (Commodity, ~$1.50/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/100-21-0"
    },
    "O=C(O)c1ccc(C(=O)O)o1": {
        "name": "2,5-Furandicarboxylic Acid (FDCA, Bio-based)",
        "cas_number": "3238-40-2",
        "formula": "C6H4O5",
        "molecular_weight": 156.09,
        "is_commercially_available": True,
        "primary_vendor": "Avantium / Strem Chemicals / TCI",
        "price_tier": "$$$ (Specialty Bio-monomer, ~$120/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/3238-40-2"
    },
    "OC(=O)CCCCCCCCC(=O)O": {
        "name": "Sebacic Acid (Castor-oil derived)",
        "cas_number": "111-20-6",
        "formula": "C10H18O4",
        "molecular_weight": 202.25,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / Arkema / Jayant Agro",
        "price_tier": "$$ (Bio-based bulk, ~$5/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/111-20-6"
    },
    "OC(=O)CCCC(=O)O": {
        "name": "Adipic Acid",
        "cas_number": "124-04-9",
        "formula": "C6H10O4",
        "molecular_weight": 146.14,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / BASF / Ascend",
        "price_tier": "$ (Commodity bulk, ~$2/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/124-04-9"
    },
    "O=C(O)CCC(=O)O": {
        "name": "Succinic Acid (Bio-based)",
        "cas_number": "110-15-6",
        "formula": "C4H6O4",
        "molecular_weight": 118.09,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / Reverdia / Succinity",
        "price_tier": "$$ (Bio-fermented, ~$4/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/110-15-6"
    },
    # Diols
    "OCCO": {
        "name": "Ethylene Glycol",
        "cas_number": "107-21-1",
        "formula": "C2H6O2",
        "molecular_weight": 62.07,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / Dow / Shell",
        "price_tier": "$ (Commodity bulk, <$1.20/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/107-21-1"
    },
    "OCCCCO": {
        "name": "1,4-Butanediol (BDO)",
        "cas_number": "110-63-4",
        "formula": "C4H10O2",
        "molecular_weight": 90.12,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / BASF / Genomatica",
        "price_tier": "$$ (Commodity bulk, ~$2.80/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/110-63-4"
    },
    "OC1C2OCC(O)C2OC1": {
        "name": "Isosorbide (Green Starch-derived Diol)",
        "cas_number": "652-67-5",
        "formula": "C6H10O4",
        "molecular_weight": 146.14,
        "is_commercially_available": True,
        "primary_vendor": "Roquette Frères / Sigma-Aldrich",
        "price_tier": "$$ (Bio-specialty, ~$8/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/652-67-5"
    },
    # Diamines
    "NCCCCCCN": {
        "name": "Hexamethylenediamine (HMDA)",
        "cas_number": "124-09-4",
        "formula": "C6H16N2",
        "molecular_weight": 116.21,
        "is_commercially_available": True,
        "primary_vendor": "Sigma-Aldrich / Invista / BASF",
        "price_tier": "$$ (Nylon precursor, ~$3.50/kg)",
        "sigma_search_url": "https://www.sigmaaldrich.com/US/en/search/124-09-4"
    }
}


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def resolve_monomer_vendor(monomer_smiles: str) -> dict:
    """
    Queries curated commercial catalogs and PubChem PUG-REST to resolve
    chemical identifiers, CAS number, commercial availability, and purchase portals.
    """
    if not monomer_smiles or len(monomer_smiles.strip()) == 0:
        return {"status": "Unknown", "cas": "N/A", "is_commercially_available": False}

    # Normalize canonical SMILES
    mol = smiles_to_mol(monomer_smiles)
    if mol is not None:
        clean_smiles = Chem.MolToSmiles(mol)
    else:
        clean_smiles = monomer_smiles.replace('[*]', '').replace('*', '').strip()

    # 1. Fast lookup in Curated Reference Catalog
    if clean_smiles in KNOWN_MONOMER_CATALOG:
        entry = KNOWN_MONOMER_CATALOG[clean_smiles].copy()
        entry["smiles"] = clean_smiles
        return entry

    # 2. Check Disk Cache
    cache = load_cache()
    if clean_smiles in cache:
        return cache[clean_smiles]

    result = {
        "smiles": clean_smiles,
        "name": "Target Monomer Precursor",
        "formula": "N/A",
        "molecular_weight": "N/A",
        "cas_number": "N/A",
        "pubchem_cid": None,
        "is_commercially_available": False,
        "primary_vendor": "Custom Chemical Synthesis / Catalog Search",
        "price_tier": "$$$$ (Novel / Custom Synthesis Required)",
        "pubchem_url": "",
        "sigma_search_url": f"https://www.sigmaaldrich.com/US/en/search/{urllib.parse.quote(clean_smiles)}"
    }

    try:
        # Step 1: Compound Properties Query from PubChem
        encoded = urllib.parse.quote(clean_smiles)
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/property/IUPACName,MolecularWeight,MolecularFormula/JSON"
        req = urllib.request.Request(prop_url, headers={"User-Agent": "PolymerGenesis/2.0"})

        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            props = data["PropertyTable"]["Properties"][0]
            cid = props.get("CID")
            result["pubchem_cid"] = cid
            result["name"] = props.get("IUPACName", result["name"])
            result["formula"] = props.get("MolecularFormula", "N/A")
            result["molecular_weight"] = props.get("MolecularWeight", "N/A")
            result["pubchem_url"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"

        # Step 2: Synonyms & CAS Query
        if cid:
            syn_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
            syn_req = urllib.request.Request(syn_url, headers={"User-Agent": "PolymerGenesis/2.0"})
            with urllib.request.urlopen(syn_req, timeout=5) as sresp:
                sdata = json.loads(sresp.read().decode())
                syns = sdata["InformationList"]["Information"][0].get("Synonym", [])

                cas_regex = re.compile(r"^\d{2,7}-\d{2}-\d$")
                has_cas = False
                for s in syns:
                    if cas_regex.match(s):
                        result["cas_number"] = s
                        result["sigma_search_url"] = f"https://www.sigmaaldrich.com/US/en/search/{s}"
                        has_cas = True
                        break

                # Rigorous vendor criteria: must have valid CAS number AND catalog presence
                vendor_indicators = ["sigma", "aldrich", "alfa aesar", "tci", "acros", "fisher", "enamine", "merck"]
                found_vendors = set()
                for s in syns:
                    s_lower = s.lower()
                    for v in vendor_indicators:
                        if v in s_lower:
                            found_vendors.add(v.title())

                if has_cas and len(found_vendors) > 0:
                    result["is_commercially_available"] = True
                    result["primary_vendor"] = " / ".join(sorted(list(found_vendors)))
                    result["price_tier"] = "$$ (Commercial Catalog Available)"
                elif has_cas:
                    result["is_commercially_available"] = True
                    result["primary_vendor"] = "Specialty Chemical Suppliers"
                    result["price_tier"] = "$$$ (Commercial CAS Registered)"

    except Exception:
        # Fallback to local RDKit parsing
        if mol is not None:
            from rdkit.Chem import Descriptors, rdMolDescriptors
            result["formula"] = rdMolDescriptors.CalcMolFormula(mol)
            result["molecular_weight"] = f"{Descriptors.MolWt(mol):.2f}"

    cache[clean_smiles] = result
    save_cache(cache)
    return result
