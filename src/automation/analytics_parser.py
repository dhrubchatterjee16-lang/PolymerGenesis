"""
Automated At-Line Analytics Data Parsers for PolymerGenesis Self-Driving Lab.
Implements:
1. Gel Permeation Chromatography (GPC / SEC):
   - Baseline correction (linear / polynomial).
   - Universal / narrow standard calibration (log M vs. retention volume).
   - Molecular weight distribution moments: Mn, Mw, Mz, and dispersity (D = Mw / Mn).
   - Synthetic GPC chromatogram generator for hardware-in-the-loop simulation.
2. Differential Scanning Calorimetry (DSC):
   - ASTM E1356 glass transition midpoint extraction via d(HF)/dT inflection analysis.
   - Melting temperature (Tm) and crystallization / melting enthalpy (Delta Hm, J/g).
   - Synthetic DSC thermogram generator for hardware-in-the-loop simulation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class GPCParser:
    """
    Parses and integrates Gel Permeation Chromatography (GPC) / Size Exclusion Chromatography (SEC) traces.
    """
    def __init__(self, calib_a: float = 11.5, calib_b: float = 0.55):
        """
        calib_a, calib_b: Parameters for narrow polystyrene calibration:
        log10(M) = calib_a - calib_b * V_r (where V_r is retention volume in mL).
        """
        self.calib_a = calib_a
        self.calib_b = calib_b

    def retention_volume_to_mw(self, vr: np.ndarray) -> np.ndarray:
        """Converts retention volume Vr to molecular weight M (Da)."""
        log_m = self.calib_a - self.calib_b * vr
        return 10.0 ** log_m

    def baseline_correct(self, signal: np.ndarray) -> np.ndarray:
        """Linear baseline correction from leading to trailing baseline regions."""
        n = len(signal)
        if n < 6:
            return np.maximum(0.0, signal - float(np.min(signal)))
        n_lead = max(1, min(int(0.05 * n), n // 6))
        n_trail = max(1, min(int(0.05 * n), n // 6))
        base_start = float(np.mean(signal[:n_lead]))
        base_end = float(np.mean(signal[-n_trail:]))
        baseline = np.linspace(base_start, base_end, n)
        corrected = signal - baseline
        return np.maximum(0.0, corrected)

    def analyze_chromatogram(
        self,
        retention_volumes_ml: Union[List[float], np.ndarray],
        detector_signal: Union[List[float], np.ndarray],
        baseline_corrected: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculates Mn, Mw, Mz, and Đ from retention volume & detector signal.
        """
        vr = np.asarray(retention_volumes_ml, dtype=float)
        sig = np.asarray(detector_signal, dtype=float)

        if not baseline_corrected:
            sig = self.baseline_correct(sig)

        # Convert Vr to molecular weight slices across full chromatogram
        m_all = self.retention_volume_to_mw(vr)

        # Thresholding to isolate chromatographic elution peak from solvent/permeation noise
        peak_max = np.max(sig)
        mask = (sig > 0.03 * peak_max) & (m_all >= 300.0)

        if np.sum(mask) < 3:
            mask = sig > 0.05 * peak_max

        vr_peak = vr[mask]
        sig_peak = sig[mask]
        m_slices = m_all[mask]

        # Statistical moments:
        # Mn = sum(h_i) / sum(h_i / M_i)
        # Mw = sum(h_i * M_i) / sum(h_i)
        # Mz = sum(h_i * M_i^2) / sum(h_i * M_i)
        sum_h = np.sum(sig_peak)
        sum_h_over_m = np.sum(sig_peak / m_slices)
        sum_h_times_m = np.sum(sig_peak * m_slices)
        sum_h_times_m2 = np.sum(sig_peak * (m_slices ** 2))

        mn = float(sum_h / max(1e-12, sum_h_over_m))
        mw = float(sum_h_times_m / max(1e-12, sum_h))
        mz = float(sum_h_times_m2 / max(1e-12, sum_h_times_m))
        dispersity = float(mw / max(1e-6, mn))

        peak_idx = np.argmax(sig)
        v_peak = float(vr[peak_idx])

        return {
            "Mn": mn,
            "Mw": mw,
            "Mz": mz,
            "dispersity": dispersity,
            "peak_retention_volume_ml": v_peak,
            "retention_volumes": vr,
            "signal_corrected": sig
        }

    def generate_synthetic_gpc(
        self,
        target_mn: float,
        target_dispersity: float = 1.3,
        num_points: int = 200,
        noise_std: float = 0.005
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates realistic synthetic GPC curve from target Mn & Đ for hardware-in-the-loop simulation.
        """
        # log-normal distribution parameters:
        # sigma^2 = ln(Mw / Mn) = ln(dispersity)
        # mu = ln(Mn) + 0.5 * sigma^2
        sigma_ln = np.sqrt(max(0.001, np.log(max(1.01, target_dispersity))))
        mu_ln = np.log(max(100.0, target_mn)) + 0.5 * (sigma_ln ** 2)

        # Retention volume range (e.g., 10 mL to 20 mL)
        vr = np.linspace(10.0, 20.0, num_points)
        m_vals = self.retention_volume_to_mw(vr)

        # Log-normal density evaluated at M
        pdf = (1.0 / (m_vals * sigma_ln * np.sqrt(2.0 * np.pi))) * np.exp(
            -((np.log(m_vals) - mu_ln) ** 2) / (2.0 * sigma_ln ** 2)
        )
        # Jacobian adjustment for dV / dM = -1 / (ln(10) * calib_b * M)
        detector_signal = pdf * m_vals
        detector_signal = detector_signal / (np.max(detector_signal) + 1e-12)

        # Add Gaussian baseline noise
        noise = np.random.normal(0, noise_std, num_points)
        signal_noisy = np.maximum(0.0, detector_signal + noise)
        return vr, signal_noisy


class DSCParser:
    """
    Parses and extracts thermal transitions (Tg, Tm, Delta Hm) from DSC thermograms.
    """
    def __init__(self, heating_rate_c_min: float = 10.0):
        self.heating_rate = heating_rate_c_min

    def smooth_signal(self, signal: np.ndarray, window: int = 5) -> np.ndarray:
        """Applies a moving average smoothing filter."""
        if len(signal) < window:
            return signal
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")

    def analyze_thermogram(
        self,
        temperature_c: Union[List[float], np.ndarray],
        heat_flow_mw: Union[List[float], np.ndarray],
        sample_mass_mg: float = 5.0
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extracts:
        - Tg: ASTM E1356 midpoint (inflection point of endothermic step, peak in d(HF)/dT).
        - Tm: Endothermic melting peak temperature.
        - Delta Hm: Melting enthalpy (J/g).
        """
        t = np.asarray(temperature_c, dtype=float)
        hf = np.asarray(heat_flow_mw, dtype=float)

        # Normalize heat flow to mW / mg (or W/g)
        norm_mass = max(0.1, sample_mass_mg)
        hf_norm = hf / norm_mass

        # Smooth signal
        hf_smooth = self.smooth_signal(hf_norm, window=7)

        # Numerical first derivative d(HF)/dT
        dhf_dt = np.gradient(hf_smooth, t)

        # 1. Identify melting peak (Tm) first: deepest endothermic trough in heat flow
        dip_idx = np.argmin(hf_smooth)
        min_hf = hf_smooth[dip_idx]
        baseline_ref = np.median(hf_smooth)

        # Check if the trough is a significant endotherm (> 0.25 mW/mg below baseline)
        if (baseline_ref - min_hf) > 0.25 and dip_idx > 10 and dip_idx < len(t) - 10:
            tm_c = float(t[dip_idx])
            # Search for Tg strictly below the melting transition envelope
            tg_search_mask = (t >= -50.0) & (t <= tm_c - 20.0)
        else:
            tm_c = None
            tg_search_mask = (t >= -50.0) & (t <= 280.0)

        # 2. Identify Tg: inflection point of the endothermic heat capacity step
        # During an endothermic step, d(HF)/dT exhibits a distinct negative minimum
        if np.sum(tg_search_mask) > 10:
            dhf_sub = dhf_dt[tg_search_mask]
            t_sub = t[tg_search_mask]
            tg_idx = np.argmin(dhf_sub)
            tg_c = float(t_sub[tg_idx])
        else:
            tg_c = float(np.median(t))

        # 3. Integrate melting peak for Delta Hm (J/g) if Tm was identified
        if tm_c is not None:
            melt_peak_mask = (t >= tm_c - 15.0) & (t <= tm_c + 15.0)
            t_peak = t[melt_peak_mask]
            hf_peak = hf_smooth[melt_peak_mask]
            if len(t_peak) > 5:
                baseline_linear = np.linspace(hf_peak[0], hf_peak[-1], len(t_peak))
                area_mw_c = np.trapz(np.maximum(0.0, baseline_linear - hf_peak), t_peak)
                delta_hm = float(area_mw_c / (self.heating_rate / 60.0))
            else:
                delta_hm = 0.0
        else:
            delta_hm = 0.0

        return {
            "Tg_C": tg_c,
            "Tm_C": tm_c if tm_c is not None else float("nan"),
            "Delta_Hm_J_per_g": delta_hm,
            "temperature_c": t,
            "heat_flow_norm": hf_norm,
            "dhf_dt": dhf_dt
        }

    def generate_synthetic_dsc(
        self,
        target_tg_c: float,
        target_tm_c: Optional[float] = None,
        delta_hm_target: float = 45.0,
        temp_min_c: float = 0.0,
        temp_max_c: float = 300.0,
        num_points: int = 300
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates synthetic DSC thermogram with glass transition step and endothermic melting peak.
        """
        t = np.linspace(temp_min_c, temp_max_c, num_points)
        # Baseline slope
        hf = -0.002 * t + 0.5

        # Glass transition step (sigmoidal heat capacity step drop)
        step_width = 8.0
        delta_cp = 0.15 # mW/mg step change
        tg_step = -delta_cp / (1.0 + np.exp(-(t - target_tg_c) / (step_width / 4.0)))
        hf += tg_step

        # Melting peak if semicrystalline
        if target_tm_c is not None and target_tm_c < temp_max_c:
            sigma_m = 6.0
            peak_depth = 0.6 # mW/mg
            tm_peak = -peak_depth * np.exp(-((t - target_tm_c) ** 2) / (2.0 * (sigma_m ** 2)))
            hf += tm_peak

        # Add detector noise
        noise = np.random.normal(0, 0.003, num_points)
        hf_noisy = hf + noise

        # Multiply by sample mass 5.0 mg to yield raw mW
        return t, hf_noisy * 5.0


def parse_gpc_csv(filepath_or_buffer, parser: Optional[GPCParser] = None) -> dict:
    """
    Parses standard analytical laboratory GPC export (Agilent ChemStation, Waters Empower, Shimadzu).
    Expects columns corresponding to retention volume/time and detector signal (RI / UV).
    """
    import pandas as pd
    if parser is None:
        parser = GPCParser()

    try:
        df = pd.read_csv(filepath_or_buffer, comment="#")
    except Exception:
        df = pd.read_csv(filepath_or_buffer, sep=r"\s+", comment="#")

    col_vol = None
    col_sig = None
    for c in df.columns:
        c_low = c.lower()
        if any(k in c_low for k in ["vol", "ml", "time", "ret"]):
            col_vol = c
        elif any(k in c_low for k in ["signal", "ri", "detector", "mv", "abs"]):
            col_sig = c

    if col_vol is None or col_sig is None:
        col_vol = df.columns[0]
        col_sig = df.columns[1]

    vr = df[col_vol].to_numpy(dtype=float)
    sig = df[col_sig].to_numpy(dtype=float)
    return parser.analyze_chromatogram(vr, sig)


def parse_dsc_csv(filepath_or_buffer, parser: Optional[DSCParser] = None, sample_mass_mg: float = 5.0) -> dict:
    """
    Parses standard analytical laboratory DSC export (TA Instruments TRIOS, Mettler Toledo STARe).
    Expects columns corresponding to Temperature (°C) and Heat Flow (mW or mW/mg).
    """
    import pandas as pd
    if parser is None:
        parser = DSCParser()

    try:
        df = pd.read_csv(filepath_or_buffer, comment="#")
    except Exception:
        df = pd.read_csv(filepath_or_buffer, sep=r"\s+", comment="#")

    col_temp = None
    col_hf = None
    for c in df.columns:
        c_low = c.lower()
        if any(k in c_low for k in ["temp", "celsius", "t_c"]):
            col_temp = c
        elif any(k in c_low for k in ["heat", "flow", "hf", "mw", "flux"]):
            col_hf = c

    if col_temp is None or col_hf is None:
        col_temp = df.columns[0]
        col_hf = df.columns[1]

    t = df[col_temp].to_numpy(dtype=float)
    hf = df[col_hf].to_numpy(dtype=float)
    return parser.analyze_thermogram(t, hf, sample_mass_mg=sample_mass_mg)
