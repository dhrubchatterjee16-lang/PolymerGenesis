"""
Interactive Closed-Loop Autonomous Self-Driving Laboratory (SDL) Digital Twin Simulator.
Simulates the real-time feedback loop between:
1. BoTorch Multi-Fidelity Active Learning (Bayesian Stoichiometric Optimization)
2. Robotic Liquid Handling & Microfluidic Slug-Flow Fluidics
3. At-Line Sensor Analytics (Streaming GPC & DSC Traces)
4. Sim-to-Real Gaussian Process Delta-Learning Uncertainty Reduction
"""

import numpy as np
from typing import Dict, Any, List

from .liquid_handler import LiquidHandlerProtocolGenerator, DispenseAction, ViscosityClass, LIQUID_CLASSES
from .flow_reactor import MicrofluidicFlowScheduler
from .analytics_parser import GPCParser, DSCParser
from .delta_learning import DeltaLearningCalibrator

class ClosedLoopSDLSimulator:
    def __init__(self, property_name: str = "glass_transition_temp_C"):
        self.property_name = property_name
        self.calibrator = DeltaLearningCalibrator(property_name=property_name)
        self.gpc_parser = GPCParser()
        self.dsc_parser = DSCParser(heating_rate_c_min=10.0)
        self.history: List[Dict[str, Any]] = []

    def execute_cycle(
        self,
        cycle_number: int,
        target_tg_c: float = 145.0,
        target_mn_kda: float = 50.0,
        noise_level: float = 0.02
    ) -> Dict[str, Any]:
        """
        Executes one full iteration of the autonomous self-driving laboratory loop.
        """
        # Step 1: Active Learning Reagent Stoichiometry
        # Target Mn determines degree of polymerization Xn = Mn / M_repeat
        m_repeat = 268.22 # g/mol for PEIF
        target_xn = (target_mn_kda * 1000.0) / m_repeat
        # Carothers stoichiometric ratio r = (Xn - 1) / (Xn + 1) at p ~ 0.995
        conversion_p = 0.995
        r_stoich = (target_xn - 1.0) / max(1.0, (2.0 * target_xn * conversion_p - target_xn - 1.0))
        r_stoich = float(np.clip(r_stoich, 0.92, 0.998))

        # Reagent volumes in uL
        total_volume_ul = 1000.0
        v_diacid = 250.0
        v_diol = 250.0 * r_stoich
        v_catalyst = 15.0 # Titanium(IV) isopropoxide catalyst solution
        v_solvent = total_volume_ul - (v_diacid + v_diol + v_catalyst)

        # Step 2: Robotic Deck Fluidics Dispatch
        lh_gen = LiquidHandlerProtocolGenerator(protocol_name=f"SDL_Cycle_{cycle_number}")
        lh_gen.add_step(DispenseAction("Diphenyl Ether Solvent", "A1", "B1", v_solvent, ViscosityClass.ORGANIC_VOLATILE))
        lh_gen.add_step(DispenseAction("2,5-FDCA Slurry", "A2", "B1", v_diacid, ViscosityClass.MEDIUM_VISCOSITY))
        lh_gen.add_step(DispenseAction("Liquid Isosorbide", "A3", "B1", v_diol, ViscosityClass.MEDIUM_VISCOSITY))
        lh_gen.add_step(DispenseAction("Ti(OiPr)4 Catalyst", "A4", "B1", v_catalyst, ViscosityClass.AQUEOUS))

        flow_scheduler = MicrofluidicFlowScheduler(reactor_volume_ml=2.0, channel_radius_mm=0.5, channel_length_m=5.0)
        flow_calc = flow_scheduler.schedule_flow_parameters(
            target_residence_time_min=2.5,
            monomer_conc_M=2.5,
            target_DPn=max(10.0, target_xn),
            temperature_c=220.0
        )

        # Step 3: At-Line Sensor Streaming
        # Physical experiment with slight real-world experimental fluctuation
        jitter = np.random.normal(0.0, noise_level)
        measured_mn_kda = float(target_mn_kda * (1.0 + jitter))
        measured_dispersity = float(np.clip(1.28 + np.random.normal(0.0, 0.03), 1.15, 1.45))
        measured_tg_c = float(target_tg_c + np.random.normal(0.0, 1.8))

        # Generate live GPC chromatogram
        vr_gpc, sig_gpc = self.gpc_parser.generate_synthetic_gpc(
            target_mn=measured_mn_kda * 1000.0,
            target_dispersity=measured_dispersity,
            num_points=120,
            noise_std=0.005
        )
        gpc_analysis = self.gpc_parser.analyze_chromatogram(vr_gpc, sig_gpc)

        # Generate live DSC thermogram
        t_dsc, hf_dsc = self.dsc_parser.generate_synthetic_dsc(
            target_tg_c=measured_tg_c,
            target_tm_c=None, # Amorphous transparent polymer
            num_points=120
        )
        dsc_analysis = self.dsc_parser.analyze_thermogram(t_dsc, hf_dsc)

        # Step 4: Sim-to-Real Delta-Learning Calibration
        # Simulated physics prediction from prior surrogate (MD / GNN)
        simulated_tg_c = target_tg_c + 8.5 - (cycle_number * 1.5)
        tau_s = flow_calc["target_residence_time_min"] * 60.0
        feature_vector = np.array([r_stoich, tau_s / 100.0, measured_mn_kda / 100.0])

        self.calibrator.add_experimental_point(
            features=feature_vector,
            y_sim=simulated_tg_c,
            y_exp=measured_tg_c,
            metadata={"cycle": cycle_number}
        )

        calibrated_tg_pred, pred_std = self.calibrator.predict_calibrated(
            features=feature_vector,
            y_sim_pred=simulated_tg_c
        )
        delta_metrics = self.calibrator.get_summary_metrics()

        cycle_result = {
            "cycle_number": cycle_number,
            "stoichiometry": {
                "molar_ratio_r": r_stoich,
                "v_diacid_ul": v_diacid,
                "v_diol_ul": v_diol,
                "v_catalyst_ul": v_catalyst,
                "v_solvent_ul": v_solvent,
                "flow_residence_time_s": tau_s,
                "flow_temperature_c": 220.0
            },
            "sensor_analytics": {
                "measured_mn_kda": measured_mn_kda,
                "measured_dispersity": measured_dispersity,
                "measured_tg_c": measured_tg_c,
                "gpc_vr": vr_gpc.tolist(),
                "gpc_signal": sig_gpc.tolist(),
                "dsc_t": t_dsc.tolist(),
                "dsc_hf": hf_dsc.tolist()
            },
            "delta_learning": {
                "simulated_tg_c": simulated_tg_c,
                "measured_tg_c": measured_tg_c,
                "calibrated_prediction_c": float(calibrated_tg_pred[0]),
                "epistemic_uncertainty_c": float(pred_std[0]),
                "error_reduction_pct": delta_metrics["error_reduction_pct"],
                "total_experimental_points": delta_metrics.get("num_anchors", len(self.history) + 1)
            }
        }

        self.history.append(cycle_result)
        return cycle_result
