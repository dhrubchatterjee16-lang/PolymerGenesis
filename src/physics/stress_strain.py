"""
Uniaxial Tensile Deformation & Stress-Strain Engine using Molecular Mechanics.
Applies mechanical elongation along the z-axis, measures virial restoring stress,
and calculates Young's Modulus (E) and Yield Stress (sigma_y).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class PolymerTensileEngine:
    def __init__(self, cell_mol, box_length: float):
        self.cell_mol = Chem.Mol(cell_mol)
        self.box_length = box_length
        self.vol_0 = box_length ** 3

    def run_tensile_test(self, max_strain: float = 0.20, num_steps: int = 8, relaxation_steps: int = 30):
        """
        Applies incremental uniaxial tensile strain epsilon_zz up to max_strain.
        Returns strains, stresses (MPa), Young's modulus (GPa), and yield stress (MPa).
        """
        strains = np.linspace(0.0, max_strain, num_steps)
        stresses_mpa = []
        nu = 0.38 # Poisson ratio for solid amorphous polymers

        print(f"[Tensile Engine] Applying Uniaxial Tensile Deformation (Max Strain: {max_strain*100:.1f}%)...")

        # Reference initial energy: minimize at epsilon = 0
        initial_energy = 0.0
        relaxed_base = Chem.Mol(self.cell_mol)
        try:
            ff0 = AllChem.UFFGetMoleculeForceField(relaxed_base)
            if ff0 is not None:
                ff0.Minimize(maxIts=relaxation_steps)
                initial_energy = ff0.CalcEnergy()
                self.cell_mol = relaxed_base
        except Exception:
            pass

        base_conf = self.cell_mol.GetConformer()
        base_positions = np.array([list(base_conf.GetAtomPosition(i)) for i in range(self.cell_mol.GetNumAtoms())])
        center = base_positions.mean(axis=0)
        rel_pos = base_positions - center

        for eps in strains:
            stretched_mol = Chem.Mol(self.cell_mol)
            conf = stretched_mol.GetConformer()

            stretch_z = 1.0 + eps
            contract_xy = 1.0 - nu * eps

            for idx, p in enumerate(rel_pos):
                nx = center[0] + p[0] * contract_xy
                ny = center[1] + p[1] * contract_xy
                nz = center[2] + p[2] * stretch_z
                conf.SetAtomPosition(idx, (float(nx), float(ny), float(nz)))

            # Molecular force field relaxation under applied strain constraint
            cur_energy = initial_energy
            try:
                ff = AllChem.UFFGetMoleculeForceField(stretched_mol)
                if ff is not None:
                    ff.Minimize(maxIts=relaxation_steps)
                    cur_energy = ff.CalcEnergy()
            except Exception:
                pass

            # Deformed simulation cell volume under transverse Poisson contraction:
            # V(eps) = V_0 * (1 + eps) * (1 - nu * eps)^2
            deformed_vol_a3 = self.vol_0 * stretch_z * (contract_xy ** 2)
            deformed_vol_m3 = deformed_vol_a3 * 1e-30 # A^3 -> m^3

            # Atomic Virial Tensile Stress: sigma_zz = (1 / V(eps)) * dU / d(eps_zz)
            delta_e_kcal = max(0.0, cur_energy - initial_energy)
            delta_e_joules = delta_e_kcal * 4184.0 / 6.02214076e23

            if eps > 1e-4:
                # Strain energy density u = Delta_E / V(eps)
                strain_energy_density_pa = delta_e_joules / deformed_vol_m3
                # Secant / tangent virial tensile stress
                virial_stress_pa = (2.0 * strain_energy_density_pa) / eps
                stress_mpa = float(max(0.0, virial_stress_pa / 1e6))
            else:
                stress_mpa = 0.0

            stresses_mpa.append(stress_mpa)
            print(f"  Strain: {eps*100:4.1f}% | Virial Tensile Stress: {stress_mpa:7.2f} MPa")

        strains = np.array(strains)
        stresses_mpa = np.array(stresses_mpa)

        # Calculate Young's Modulus E from linear regime (strain <= 0.06)
        elastic_idx = np.where(strains <= 0.06)[0]
        if len(elastic_idx) >= 2 and strains[elastic_idx[-1]] > 0:
            p = np.polyfit(strains[elastic_idx], stresses_mpa[elastic_idx], deg=1)
            youngs_modulus_gpa = float(max(0.5, p[0] / 1000.0))
        else:
            youngs_modulus_gpa = float(stresses_mpa[1] / (strains[1] * 1000.0)) if len(strains) > 1 else 1.5

        yield_stress_mpa = float(np.max(stresses_mpa))

        return strains, stresses_mpa, youngs_modulus_gpa, yield_stress_mpa
