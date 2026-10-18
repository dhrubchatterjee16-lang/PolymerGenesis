"""
Continuous Microfluidic Flow Reactor Scheduler & Safety Interlock for PolymerGenesis.
Implements:
1. Slug-flow multiphase droplet parameter scheduling (residence time tau, carrier oil FC-40).
2. Dynamic volumetric flow rate scheduling across monomer, catalyst, and carrier streams.
3. Hagen-Poiseuille pressure drop estimation and viscosity-induced clogging safety interlock.
"""

import math
import numpy as np

class MicrofluidicFlowScheduler:
    def __init__(self, reactor_volume_ml: float = 2.0, channel_radius_mm: float = 0.5, channel_length_m: float = 2.5):
        self.V_reactor_ml = reactor_volume_ml
        self.R_channel_m = channel_radius_mm * 1e-3 # mm to m
        self.L_channel_m = channel_length_m
        self.max_pressure_bar = 15.0 # Max safe operating pressure for PEEK / PTFE microchannels

    def schedule_flow_parameters(
        self,
        target_residence_time_min: float,
        monomer_conc_M: float,
        target_DPn: float,
        carrier_fraction: float = 0.35, # 35% FC-40 fluorinated carrier slug
        temperature_c: float = 75.0
    ) -> dict:
        """
        Calculates pump flow rates (uL/min) for monomer, catalyst, and carrier streams
        to achieve target DPn and residence time tau without channel fouling.
        """
        tau_min = max(0.5, float(target_residence_time_min))
        # Total flow rate Q_total = V_reactor / tau
        Q_total_ml_min = self.V_reactor_ml / tau_min
        Q_total_ul_min = Q_total_ml_min * 1000.0

        # Immiscible Fluorinert FC-40 carrier slug rate
        Q_carrier_ul_min = Q_total_ul_min * carrier_fraction
        Q_reaction_ul_min = Q_total_ul_min * (1.0 - carrier_fraction)

        # Catalyst / Initiator ratio determined by target DPn: [M] / [Init] = target_DPn
        # Monomer stream constitutes ~85-95% of reaction phase
        cat_fraction = 1.0 / max(2.0, target_DPn)
        Q_catalyst_ul_min = Q_reaction_ul_min * cat_fraction
        Q_monomer_ul_min = Q_reaction_ul_min - Q_catalyst_ul_min

        # Estimate reaction mixture viscosity as function of target DPn
        # Below entanglement: eta ~ DP^1.0; above entanglement (DP > 200): eta ~ DP^3.4
        if target_DPn < 150:
            viscosity_pa_s = 0.002 * (target_DPn / 50.0) ** 1.0
        else:
            viscosity_pa_s = 0.010 * (target_DPn / 150.0) ** 2.5

        # Estimate Hagen-Poiseuille differential pressure: Delta_P = (8 * eta * L * Q) / (pi * R^4)
        Q_m3_s = (Q_total_ml_min * 1e-6) / 60.0
        delta_p_pa = (8.0 * viscosity_pa_s * self.L_channel_m * Q_m3_s) / (math.pi * (self.R_channel_m ** 4))
        delta_p_bar = delta_p_pa / 1.0e5

        is_pressure_safe = delta_p_bar < self.max_pressure_bar
        recommended_action = "Continuous Flow Nominal" if is_pressure_safe else "Throttle Flow Rate / Inject Diluent Solvent"

        return {
            "target_residence_time_min": tau_min,
            "reaction_temperature_C": temperature_c,
            "total_flow_rate_ul_min": float(Q_total_ul_min),
            "carrier_fluid": "3M Fluorinert FC-40 (Immiscible Droplet Encapsulation)",
            "carrier_flow_rate_ul_min": float(Q_carrier_ul_min),
            "monomer_flow_rate_ul_min": float(Q_monomer_ul_min),
            "catalyst_flow_rate_ul_min": float(Q_catalyst_ul_min),
            "estimated_melt_viscosity_mPa_s": float(viscosity_pa_s * 1000.0),
            "differential_pressure_bar": float(delta_p_bar),
            "is_pressure_safe": is_pressure_safe,
            "safety_action": recommended_action
        }
