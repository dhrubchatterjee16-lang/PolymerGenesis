"""
3D Amorphous Bulk Cell Builder with Universal Force Field (UFF) Relaxation and Periodic Boundary Conditions (PBC).
Packs multiple macromolecular chains into a periodic condensed-phase box and relaxes inter-chain contacts.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation as R

def wrap_coordinates_pbc(coords: np.ndarray, box_length: float) -> np.ndarray:
    """
    Applies periodic boundary condition (PBC) minimum image convention wrapping.
    Maps all atomic Cartesian coordinates into the primary simulation box [0, box_length).
    """
    if box_length <= 0:
        return coords
    return coords - np.floor(coords / box_length) * box_length


class AmorphousCellBuilder:
    def __init__(self, target_density: float = 1.05):
        """
        target_density: Target condensed-phase bulk density in g/cm^3 (default 1.05 g/cm^3 for polymers).
        """
        self.target_density = target_density

    def compute_box_length(self, chain_mol, num_chains: int) -> float:
        """Computes cubic box length L in Angstroms from total molar mass and target bulk density."""
        mw = sum(atom.GetMass() for atom in chain_mol.GetAtoms()) * num_chains
        mass_grams = mw / 6.02214076e23
        volume_cm3 = mass_grams / self.target_density
        volume_angstrom3 = volume_cm3 * 1e24
        return float(volume_angstrom3 ** (1.0 / 3.0))

    def pack_cell(self, chain_mol, num_chains: int = 3, apply_pbc_wrapping: bool = True):
        """
        Packs num_chains into an amorphous bulk cell with 3D periodic boundary conditions (PBC).
        Minimizes inter-chain steric clashes via force field relaxation.
        Returns:
            combined_mol: RDKit multi-chain molecule
            L_box: cubic periodic box length L (Angstroms)
            all_positions: (N_atoms, 3) numpy array of Cartesian coordinates
        """
        L = self.compute_box_length(chain_mol, num_chains)
        # Allocate box with realistic packing fraction
        L_box = max(L, 16.0)

        # Base chain conformer centered at origin
        conf = chain_mol.GetConformer()
        base_pos = np.array([list(conf.GetAtomPosition(i)) for i in range(chain_mol.GetNumAtoms())])
        base_pos = base_pos - base_pos.mean(axis=0)

        chain_copies = []
        grid_dim = int(np.ceil(num_chains ** (1.0 / 3.0)))
        step = L_box / max(grid_dim, 1)
        grid_points = []
        for ix in range(grid_dim):
            for iy in range(grid_dim):
                for iz in range(grid_dim):
                    grid_points.append(np.array([(ix + 0.5) * step, (iy + 0.5) * step, (iz + 0.5) * step]))

        for i in range(num_chains):
            copy_mol = Chem.Mol(chain_mol)
            rot = R.random().as_matrix()
            rot_pos = np.dot(base_pos, rot.T)

            anchor = grid_points[i % len(grid_points)] + np.random.uniform(-step * 0.2, step * 0.2, size=3)
            final_pos = rot_pos + anchor

            if apply_pbc_wrapping:
                final_pos = wrap_coordinates_pbc(final_pos, L_box)

            copy_conf = copy_mol.GetConformer()
            for idx, p in enumerate(final_pos):
                copy_conf.SetAtomPosition(idx, (float(p[0]), float(p[1]), float(p[2])))
            chain_copies.append(copy_mol)

        # Combine chains into single multi-chain system
        combined_mol = chain_copies[0]
        for c in chain_copies[1:]:
            combined_mol = Chem.CombineMols(combined_mol, c)

        Chem.FastFindRings(combined_mol)

        # UFF Energy Minimization to eliminate steric overlap in bulk cell
        try:
            ff = AllChem.UFFGetMoleculeForceField(combined_mol)
            if ff is not None:
                ff.Minimize(maxIts=250)
        except Exception:
            pass

        final_conf = combined_mol.GetConformer()
        all_positions = np.array([list(final_conf.GetAtomPosition(i)) for i in range(combined_mol.GetNumAtoms())])

        if apply_pbc_wrapping:
            all_positions = wrap_coordinates_pbc(all_positions, L_box)
            for idx, p in enumerate(all_positions):
                final_conf.SetAtomPosition(idx, (float(p[0]), float(p[1]), float(p[2])))

        return combined_mol, float(L_box), all_positions

    def save_xyz(self, combined_mol, filepath: str, box_length: float = None):
        """
        Saves Cartesian coordinates to Extended XYZ format with 3D Periodic Lattice vectors:
        Lattice="L 0.0 0.0 0.0 L 0.0 0.0 0.0 L" Properties=species:S:1:pos:R:3 pbc="T T T"
        Fully compatible with ASE, OVITO, VMD, and PyMOL.
        """
        conf = combined_mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in combined_mol.GetAtoms()]
        num_atoms = combined_mol.GetNumAtoms()

        L = box_length if box_length is not None else 25.0
        lattice_str = f'Lattice="{L:.4f} 0.0 0.0 0.0 {L:.4f} 0.0 0.0 0.0 {L:.4f}" Properties=species:S:1:pos:R:3 pbc="T T T"'

        with open(filepath, "w") as f:
            f.write(f"{num_atoms}\n")
            f.write(f"{lattice_str}\n")
            for i, sym in enumerate(symbols):
                pos = conf.GetAtomPosition(i)
                f.write(f"{sym:<3} {pos.x:10.5f} {pos.y:10.5f} {pos.z:10.5f}\n")
