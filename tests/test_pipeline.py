"""
End-to-End Comprehensive Test Suite for PolymerGenesis.
Tests all scientific, AI, generative, physics, and chemical modules.
"""

import os
import sys
import numpy as np
import torch
from rdkit import Chem

# Ensure scratch/PolymerGenesis is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features.featurizer import (
    smiles_to_mol,
    get_morgan_fingerprint,
    get_rdkit_descriptors,
    smiles_to_graph
)
from src.features.tokenizer import PolymerTokenizer
from src.utils.chemistry import (
    is_valid_polymer,
    compute_synthetic_accessibility,
    compute_tanimoto_similarity,
    compute_novelty,
    compute_internal_diversity
)
from src.utils.metrics import (
    compute_regression_metrics,
    compute_calibration_curve,
    compute_sparsification_curve
)
from src.chemistry.circularity import analyze_circularity
from src.chemistry.retrosynthesis import plan_retrosynthesis
from src.chemistry.vendor_resolver import resolve_monomer_vendor
from src.models.polymer_vae import PolymerVAE
from src.models.gnn_surrogate import PolymerGNN, PhysicsInformedPolymerGNN
from src.models.bayesian_opt import LatentBayesianOptimizer
from src.models.multi_fidelity_opt import MultiFidelityPolymerOptimizer
from src.models.pareto_opt import MultiObjectiveParetoOptimizer
from src.chemistry.reaction_grammar import (
    assemble_polyester,
    assemble_polyamide,
    assemble_polycarbonate,
    validate_carothers_stoichiometry,
    enumerate_combinatorial_library
)
from src.physics.chain_builder import PolymerChainBuilder
from src.physics.amorphous_cell import AmorphousCellBuilder
from src.physics.md_engine import PolymerMDEngine, calibrate_wlf_tg
from src.physics.stress_strain import PolymerTensileEngine
from src.automation.liquid_handler import (
    LiquidHandlerProtocolGenerator,
    ViscosityClass,
    DispenseAction,
    generate_opentrons_script,
    classify_synthesis_hardware
)
from src.automation.flow_reactor import MicrofluidicFlowScheduler
from src.automation.analytics_parser import GPCParser, DSCParser, parse_gpc_csv, parse_dsc_csv
from src.automation.delta_learning import DeltaLearningCalibrator




class TestFeaturizer:
    def test_smiles_to_mol(self):
        mol = smiles_to_mol("*CC*")
        assert mol is not None
        assert mol.GetNumAtoms() >= 2

        # Invalid SMILES handling
        invalid_mol = smiles_to_mol("Invalid>>>SMILES***")
        assert invalid_mol is None

        # Empty SMILES handling
        assert smiles_to_mol("") is None

    def test_morgan_fingerprint(self):
        fp = get_morgan_fingerprint("*CC*")
        assert isinstance(fp, np.ndarray)
        assert fp.shape == (2048,)
        assert fp.sum() > 0

        # Empty SMILES returns zero vector
        fp_empty = get_morgan_fingerprint("")
        assert fp_empty.shape == (2048,)
        assert fp_empty.sum() == 0.0

    def test_rdkit_descriptors(self):
        desc = get_rdkit_descriptors("*CC(*)c1ccccc1")
        assert isinstance(desc, np.ndarray)
        assert desc.shape == (12,)
        assert not np.isnan(desc).any()
        assert not np.isinf(desc).any()

    def test_smiles_to_graph(self):
        data = smiles_to_graph("*CC*", y_val=350.0)
        assert data is not None
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert data.y.item() == 350.0
        assert data.x.shape[1] == 30 # standard atom feature dim

    def test_featurizer_chirality(self):
        # Stereochemical feature dimension (30 + 4 chiral tags = 34)
        data_chiral = smiles_to_graph("*CC(*)c1ccccc1", include_chirality=True)
        assert data_chiral is not None
        assert data_chiral.x.shape[1] == 34



class TestChemistryAndCircularity:
    def test_valid_polymer_check(self):
        assert is_valid_polymer("*CC*") is True
        assert is_valid_polymer("*CC(*)c1ccccc1") is True
        assert is_valid_polymer("CC") is False # No wildcards
        assert is_valid_polymer("") is False

    def test_synthetic_accessibility(self):
        sa = compute_synthetic_accessibility("*CC*")
        assert 1.0 <= sa <= 10.0

    def test_diversity_and_novelty(self):
        sim = compute_tanimoto_similarity("*CC*", "*CCC*")
        assert 0.0 <= sim <= 1.0
        nov = compute_novelty(["*CC*", "*CF*"], {"*CC*"})
        assert nov == 0.5
        div = compute_internal_diversity(["*CC*", "*CO*", "*CS*"])
        assert div > 0.0

    def test_circularity_analysis(self):
        # Polyester: ester linkage
        res_ester = analyze_circularity("*C(=O)c1ccc(C(=O)OCCCCCC)cc1*")
        assert res_ester["circularity_score"] >= 80.0
        assert len(res_ester["detected_linkages"]) > 0
        assert "vitrimer_kinetics" in res_ester
        assert "thermodynamics" in res_ester
        assert "lca" in res_ester

        # Carbon-carbon backbone: baseline recyclability
        res_pe = analyze_circularity("*CC*")
        assert res_pe["circularity_score"] == 25.0

    def test_vitrimer_and_thermodynamics(self):
        # Schiff base / dynamic imine vitrimer
        res_imine = analyze_circularity("*c1ccc(C=Nc2ccccc2)cc1*")
        assert res_imine["vitrimer_kinetics"] is not None
        v_data = res_imine["vitrimer_kinetics"]
        assert 150.0 <= v_data["Tv_K"] <= 500.0
        assert v_data["malleability_score"] > 50.0

        # Dainton-Ivin Thermodynamics
        t_data = res_imine["thermodynamics"]
        assert 150.0 <= t_data["Tc_K"] <= 850.0

        # Bio-based carbon and LCA (FDCA furan polyester)
        res_pef = analyze_circularity("*C(=O)c1ccc(o1)C(=O)OCCOC(=O)*")
        lca = res_pef["lca"]
        assert lca["f_bio_C"] > 0.0 # Detected furan biogenic motif
        assert lca["net_GWP_kg_CO2_kg"] < 3.8 # Verified net CO2 reduction from biogenic offset

    def test_retrosynthesis(self):
        plan_ester = plan_retrosynthesis("*C(=O)c1ccc(C(=O)OCCCCCC)cc1*")
        assert "polymerization_type" in plan_ester
        assert len(plan_ester["precursor_monomers"]) > 0
        assert "laboratory_protocol" in plan_ester

    def test_vendor_resolver(self):
        res = resolve_monomer_vendor("C=CC(=O)OC") # Methyl acrylate
        assert "smiles" in res
        assert "sigma_search_url" in res

    def test_reaction_grammar_assembly(self):
        # 1. Test Carothers Stoichiometric Equation
        carothers = validate_carothers_stoichiometry(molar_ratio_r=1.0, conversion_p=0.99)
        assert carothers["degree_of_polymerization_Xn"] >= 95.0
        assert carothers["is_gelation_risk"] is False

        # 2. Test Combinatorial Assembly
        pe = assemble_polyester("c1ccc(cc1)", "CC")
        assert pe == "*C(=O)c1ccc(cc1)C(=O)OCCO*"
        assert is_valid_polymer(pe) is True

        pa = assemble_polyamide("CCCC", "CCCCCC")
        assert pa == "*C(=O)CCCCC(=O)NCCCCCCN*" or pa == "*C(=O)CCCC(=O)NCCCCCCN*"
        assert is_valid_polymer(pa) is True


        # 3. Test Library Enumeration
        lib = enumerate_combinatorial_library()
        assert len(lib) > 20
        assert all(is_valid_polymer(item["repeat_unit_smiles"]) for item in lib)




class TestGenerativeAndSurrogates:
    def test_tokenizer(self):
        tok = PolymerTokenizer()
        s = "*CC*"
        encoded = tok.encode(s, max_len=20)
        assert len(encoded) == 20
        decoded = tok.decode(encoded)
        assert decoded == s

    def test_polymer_vae(self):
        tok = PolymerTokenizer()
        vae = PolymerVAE(vocab_size=tok.vocab_size, embed_dim=32, hidden_dim=64, latent_dim=16)
        x = torch.tensor([tok.encode("*CC*", max_len=20), tok.encode("*CO*", max_len=20)], dtype=torch.long)
        logits, mu, log_var, z = vae(x, x)
        assert logits.shape == (2, 20, tok.vocab_size)
        assert mu.shape == (2, 16)
        assert log_var.shape == (2, 16)
        assert z.shape == (2, 16)

        samples = vae.sample(num_samples=2, tokenizer=tok, device="cpu")
        assert len(samples) == 2
        assert isinstance(samples[0], str)

    def test_gnn_surrogate(self):
        gnn = PolymerGNN(in_channels=30, hidden_dim=32, num_layers=2, dropout=0.2)
        d1 = smiles_to_graph("*CC*")
        d2 = smiles_to_graph("*CO*")
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([d1, d2])

        mu, log_var = gnn(batch.x, batch.edge_index, batch.batch)
        assert mu.shape == (2,)
        assert log_var.shape == (2,)

        mu_mcd, std_tot, std_epi, std_ale = gnn.predict_monte_carlo(batch.x, batch.edge_index, batch.batch, num_samples=5)
        assert len(mu_mcd) == 2
        assert len(std_tot) == 2
        assert np.all(std_tot >= 0.0)

    def test_physics_informed_gnn_fox_flory(self):
        # Physics-informed GNN verifying Fox-Flory relation: Tg(Mn) = Tg_inf - K / Mn
        p_gnn = PhysicsInformedPolymerGNN(in_channels=30, hidden_dim=32, num_layers=2, dropout=0.2)
        d = smiles_to_graph("*CC(*)c1ccccc1")
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([d])

        # 1. Asymptotic limit (Mn -> inf)
        mu_inf, _ = p_gnn(batch.x, batch.edge_index, batch.batch, m_n=None)

        # 2. High molecular weight (Mn = 200 kDa)
        mu_high, _ = p_gnn(batch.x, batch.edge_index, batch.batch, m_n=torch.tensor([200.0]))

        # 3. Low molecular weight (Mn = 10 kDa)
        mu_low, _ = p_gnn(batch.x, batch.edge_index, batch.batch, m_n=torch.tensor([10.0]))

        # Under Fox-Flory thermodynamics: Tg(low Mn) < Tg(high Mn) <= Tg_inf
        assert mu_low.item() < mu_high.item() <= mu_inf.item() + 1e-4



class TestOptimizationModules:
    def test_bayesian_optimizer(self):
        bo = LatentBayesianOptimizer(latent_dim=16, bounds_range=3.0)
        z_hist = torch.randn(6, 16)
        rewards = torch.tensor([1.2, 2.5, 0.8, 3.1, 2.0, 1.5])
        candidates = bo.optimize_step(z_hist, rewards, q=2)
        assert candidates.shape == (2, 16)
        assert torch.all(candidates >= -3.5) and torch.all(candidates <= 3.5)

    def test_multi_fidelity_optimizer(self):
        mf_bo = MultiFidelityPolymerOptimizer(latent_dim=16, bounds_range=3.0)
        # N=8 observations with D=16 latent coords + 1 fidelity coord (0.0 or 1.0)
        z_latent = torch.randn(8, 16)
        fidelities = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [1.0]])
        train_X = torch.cat([z_latent, fidelities], dim=-1)
        train_Y = torch.tensor([1.0, 1.2, 1.5, 0.9, 2.1, 2.4, 1.1, 2.7]).unsqueeze(-1)

        candidates = mf_bo.suggest_candidates(train_X, train_Y, q=2, target_fidelity=1.0)
        assert candidates.shape == (2, 17)
        # Fidelity dimension should be 1.0
        assert torch.all(candidates[:, 16] == 1.0)

    def test_multi_objective_pareto_optimizer(self):
        # 2-objective optimization: e.g. [Tg, Circularity]
        mo_bo = MultiObjectiveParetoOptimizer(latent_dim=16, num_objectives=2, bounds_range=3.0)
        z_hist = torch.randn(8, 16)
        # N=8 observations with M=2 objectives
        outcomes = torch.tensor([
            [500.0, 90.0],
            [520.0, 70.0],
            [480.0, 95.0],
            [540.0, 60.0],
            [510.0, 85.0],
            [490.0, 92.0],
            [530.0, 65.0],
            [470.0, 98.0]
        ])
        candidates = mo_bo.optimize_pareto_step(z_hist, outcomes, q=2)
        assert candidates.shape == (2, 16)

        # Test Pareto frontier identification
        pareto_mask = MultiObjectiveParetoOptimizer.identify_pareto_front(outcomes.numpy())
        assert pareto_mask.shape == (8,)
        assert np.sum(pareto_mask) >= 1



class TestPhysicsEngine:
    def test_chain_and_amorphous_cell(self):
        builder = PolymerChainBuilder(n_units=4)
        chain_mol = builder.build_chain("*CC*")
        assert chain_mol is not None
        assert chain_mol.GetNumAtoms() >= 8

        mol_3d = builder.generate_3d_conformer(chain_mol)
        assert mol_3d.GetNumConformers() > 0

        cell_builder = AmorphousCellBuilder(target_density=0.9)
        combined_mol, L_box, positions = cell_builder.pack_cell(mol_3d, num_chains=2)
        assert combined_mol.GetNumAtoms() == mol_3d.GetNumAtoms() * 2
        assert L_box > 10.0
        assert positions.shape == (combined_mol.GetNumAtoms(), 3)

    def test_md_annealing_and_tensile(self):
        builder = PolymerChainBuilder(n_units=3)
        chain_mol = builder.build_chain("*CC*")
        mol_3d = builder.generate_3d_conformer(chain_mol)

        cell_builder = AmorphousCellBuilder(target_density=0.9)
        combined_mol, L_box, _ = cell_builder.pack_cell(mol_3d, num_chains=2)

        # Quick 2-step annealing
        md_engine = PolymerMDEngine(combined_mol, box_length=L_box)
        traj, tg = md_engine.run_thermal_annealing(temp_start=400.0, temp_end=350.0, temp_step=50.0, num_sweeps=5)
        assert len(traj) == 2
        assert 300.0 <= tg <= 450.0

        # Quick 2-step tensile
        tensile_engine = PolymerTensileEngine(combined_mol, box_length=L_box)
        strains, stresses, E, yield_stress = tensile_engine.run_tensile_test(max_strain=0.05, num_steps=3, relaxation_steps=5)
        assert len(strains) == 3
        assert len(stresses) == 3
        assert E > 0.0
        assert yield_stress >= 0.0

    def test_tacticity_chain_builder(self):
        builder = PolymerChainBuilder(n_units=4)
        # Polystyrene repeat unit with asymmetric backbone carbon
        s = "*CC(*)c1ccccc1"

        iso_mol = builder.build_chain(s, tacticity="isotactic")
        assert iso_mol is not None
        assert iso_mol.GetNumAtoms() > 10

        syndio_mol = builder.build_chain(s, tacticity="syndiotactic")
        assert syndio_mol is not None

        atactic_mol = builder.build_chain(s, tacticity="atactic")
        assert atactic_mol is not None

    def test_wlf_timescale_calibration(self):
        # High-cooling-rate MD Tg = 500 K quenched at 10^10 K/s
        # Should calibrate down by ~50-80 K to experimental DSC at 0.33 K/s
        calib = calibrate_wlf_tg(md_tg=500.0, md_cooling_rate_k_per_s=1e10, exp_cooling_rate_k_per_s=0.33)
        assert "calibrated_exp_tg_K" in calib
        assert "delta_cooling_shift_K" in calib
        assert 420.0 <= calib["calibrated_exp_tg_K"] <= 470.0
        assert 30.0 <= calib["delta_cooling_shift_K"] <= 80.0



class TestEvaluationMetrics:
    def test_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.5, 19.5, 30.2, 39.8])
        reg = compute_regression_metrics(y_true, y_pred)
        assert reg["R2"] > 0.95
        assert reg["RMSE"] < 1.0

        calib = compute_calibration_curve(y_true, y_pred, np.array([0.5, 0.5, 0.5, 0.5]), num_bins=5)
        assert "MACE" in calib
        assert 0.0 <= calib["MACE"] <= 1.0

        spars = compute_sparsification_curve(y_true, y_pred, np.array([0.1, 0.2, 0.3, 0.4]), num_points=5)
        assert len(spars["rmse_curve"]) == 5


class TestAutomationAndSDL:
    def test_liquid_handler_protocol(self):
        lh = LiquidHandlerProtocolGenerator(protocol_name="TestProtocol")
        lh.add_step(
            DispenseAction(
                liquid_name="PEGDA_Monomer",
                source_well="A1",
                dest_well="B1",
                volume_ul=150.0,
                viscosity_class=ViscosityClass.MEDIUM_VISCOSITY
            )
        )
        script = lh.generate_python_script()
        assert "ot2_run" in script or "opentrons" in script or "run(protocol" in script
        assert "PEGDA_Monomer" in script
        assert "blow_out" in script

    def test_flow_reactor_scheduling(self):
        flow = MicrofluidicFlowScheduler(reactor_volume_ml=2.0)
        res = flow.schedule_flow_parameters(
            target_residence_time_min=10.0,
            monomer_conc_M=2.5,
            target_DPn=80.0,
            carrier_fraction=0.35,
            temperature_c=80.0
        )
        assert res["total_flow_rate_ul_min"] == 200.0
        assert res["carrier_flow_rate_ul_min"] == 70.0
        assert res["is_pressure_safe"] is True
        assert res["differential_pressure_bar"] < 15.0

    def test_gpc_analytics_parser(self):
        parser = GPCParser()
        vr, sig = parser.generate_synthetic_gpc(target_mn=25000.0, target_dispersity=1.35)
        results = parser.analyze_chromatogram(vr, sig)
        assert results["Mn"] > 5000.0
        assert results["Mw"] > results["Mn"]
        assert results["dispersity"] >= 1.0

    def test_dsc_analytics_parser(self):
        dsc = DSCParser(heating_rate_c_min=10.0)
        t, hf = dsc.generate_synthetic_dsc(target_tg_c=85.0, target_tm_c=165.0)
        res = dsc.analyze_thermogram(t, hf)
        assert abs(res["Tg_C"] - 85.0) < 15.0
        assert not np.isnan(res["Tm_C"])
        assert abs(res["Tm_C"] - 165.0) < 10.0

    def test_delta_learning_calibrator(self):
        calibrator = DeltaLearningCalibrator("Tg")
        for i in range(4):
            feat = np.array([0.1 * i, 0.2 * i, 0.5])
            y_sim = 80.0 + i * 10.0
            y_exp = y_sim + 12.0
            calibrator.add_experimental_point(feat, y_sim, y_exp)

        test_feat = np.array([0.2, 0.4, 0.5])
        y_sim_test = 100.0
        y_cal, std = calibrator.predict_calibrated(test_feat, y_sim_test)
        assert y_cal[0] > 105.0
        diag = calibrator.get_summary_metrics()
        assert diag["error_reduction_pct"] > 50.0

    def test_hardware_synthesis_classification(self):
        # 1. Vinyl / Acrylate solution polymerization -> Opentrons OT-2
        route_vinyl = classify_synthesis_hardware("Addition / Vinyl / Acrylic")
        assert route_vinyl["platform"] == "opentrons_ot2"
        assert route_vinyl["is_pipette_compatible"] is True

        # 2. Polyester melt polycondensation -> Microfluidic Flow Reactor
        route_ester = classify_synthesis_hardware("Step-Growth Polyester")
        assert route_ester["platform"] == "microfluidic_flow_reactor"
        assert route_ester["is_pipette_compatible"] is False

    def test_csv_analytics_parsing(self):
        import io
        # 1. Test GPC CSV parser with realistic trace
        gpc_parser = GPCParser()
        vr_syn, sig_syn = gpc_parser.generate_synthetic_gpc(target_mn=20000.0, num_points=40)
        lines_gpc = ["Retention_Volume_mL,RI_Signal_mV"]
        for v, s in zip(vr_syn, sig_syn):
            lines_gpc.append(f"{v:.3f},{s:.4f}")
        gpc_csv = io.StringIO("\n".join(lines_gpc))
        gpc_res = parse_gpc_csv(gpc_csv)
        assert gpc_res["Mn"] > 0.0
        assert gpc_res["dispersity"] >= 1.0

        # 2. Test DSC CSV parser with realistic trace
        dsc_parser = DSCParser()
        t_syn, hf_syn = dsc_parser.generate_synthetic_dsc(target_tg_c=80.0, num_points=50)
        lines_dsc = ["Temperature_C,Heat_Flow_mW"]
        for t, h in zip(t_syn, hf_syn):
            lines_dsc.append(f"{t:.2f},{h:.3f}")
        dsc_csv = io.StringIO("\n".join(lines_dsc))
        dsc_res = parse_dsc_csv(dsc_csv)
        assert 50.0 <= dsc_res["Tg_C"] <= 120.0


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING POLYMERGENESIS SYSTEM-WIDE TEST SUITE")
    print("=" * 70)

    test_classes = [
        TestFeaturizer,
        TestChemistryAndCircularity,
        TestGenerativeAndSurrogates,
        TestOptimizationModules,
        TestPhysicsEngine,
        TestEvaluationMetrics,
        TestAutomationAndSDL
    ]

    total_passed = 0
    total_failed = 0

    for cls in test_classes:
        instance = cls()
        print(f"\n[Running] {cls.__name__}...")
        for attr in dir(instance):
            if attr.startswith("test_"):
                test_fn = getattr(instance, attr)
                try:
                    test_fn()
                    print(f"  PASS: {attr}")
                    total_passed += 1
                except Exception as e:
                    print(f"  FAIL: {attr} -> {e}")
                    import traceback
                    traceback.print_exc()
                    total_failed += 1

    print("\n" + "=" * 70)
    print(f"TEST RUN COMPLETED: {total_passed} Passed | {total_failed} Failed")
    print("=" * 70)
    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

