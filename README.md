# PolymerGenesis: Materials Informatics & GNNs for Polymer Property Prediction

PolymerGenesis is a graph neural network and materials informatics pipeline designed to predict macroscopic thermal and mechanical properties of synthetic macromolecules directly from monomer topologies and molecular weight distributions.

## Architecture & Methodology
- **Hierarchical Representation:** BigSMILES and multi-scale molecular graphs capturing monomer repeat units, end-groups, and topological branching.
- **Graph Neural Network Backbone:** Directed Message Passing Neural Networks (D-MPNN) combined with persistent homology topological feature extraction.
- **Target Physical Properties:** Glass transition temperature ($T_g$), tensile strength, dielectric permittivity, and polymer-solvent Flory-Huggins interaction parameters ($\\chi$).
- **Benchmark Baselines:** Validated on the Polymer Genome dataset and PolyInfo experimental records.

## Repository Structure
```
PolymerGenesis/
├── data/              # Polymer benchmarks & BigSMILES datasets
├── models/            # D-MPNN, GAT, and feature embedding layers
├── featurizers/       # Topological persistent homology & Morgan fingerprints
├── notebooks/         # Property parity plots and training trajectories
├── requirements.txt   # PyTorch, PyG, RDKit dependencies
└── predict.py         # Command-line inference script
```

## Quickstart
```bash
git clone https://github.com/dhrubchatterjee16-lang/PolymerGenesis.git
cd PolymerGenesis
pip install -r requirements.txt
python predict.py --smiles "*CC(*)c1ccccc1" --property Tg
```
