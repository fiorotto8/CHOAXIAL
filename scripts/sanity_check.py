from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import ESS_flux
import JPARK_flux
import SNS_flux
from cevens import CEvNSCalculator, NeutrinoElectronCalculator, carbon12_target, cf4_electron_target, fluorine19_target
from detector_estimation import DEFAULT_CARBON_QF_CSV, DEFAULT_FLUORINE_QF_CSV, load_config, load_quenching_curve


def assert_close(name: str, value: float, expected: float, *, rtol: float = 1.0e-3) -> None:
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=rtol, abs_tol=0.0):
        raise AssertionError(f"{name}: got {value:.8e}, expected {expected:.8e}")


def check_michel_normalization(module) -> None:
    energy = np.linspace(0.0, module.E_NU_MAX_MEV, 20001)
    nue_norm = float(np.trapz(module.michel_spectrum_nue(energy), energy))
    numubar_norm = float(np.trapz(module.michel_spectrum_numubar(energy), energy))
    assert_close(f"{module.__name__} nue Michel normalization", nue_norm, 1.0)
    assert_close(f"{module.__name__} numubar Michel normalization", numubar_norm, 1.0)


def check_flux_modules() -> None:
    for module, beam_class in (
        (ESS_flux, ESS_flux.ESSBeamConfig),
        (SNS_flux, SNS_flux.SNSBeamConfig),
        (JPARK_flux, JPARK_flux.JPARCMLFBeamConfig),
    ):
        check_michel_normalization(module)
        beam = beam_class()
        distance_m = 20.0
        expected = beam.neutrino_yield_per_proton_per_flavor * module.isotropic_geometry_factor_cm2(distance_m)
        actual = module.prompt_numu_line_fluence_per_pot(distance_m, beam=beam)
        assert_close(f"{module.__name__} prompt fluence", actual, expected, rtol=1.0e-12)


def check_cross_sections() -> None:
    calc = CEvNSCalculator()
    fluorine = fluorine19_target(axial_model="hoferichter_19f_central")
    carbon = carbon12_target()

    for target in (carbon, fluorine):
        total = calc.differential_cross_section_cm2_per_kev(target, 30.0, 5.0)
        vector = calc.differential_vector_cross_section_cm2_per_kev(target, 30.0, 5.0)
        axial = calc.differential_axial_cross_section_cm2_per_kev(target, 30.0, 5.0)
        if total < 0.0 or vector < 0.0 or axial < 0.0:
            raise AssertionError(f"{target.name}: negative CEvNS component")
        assert_close(f"{target.name} vector+axial", vector + axial, total, rtol=1.0e-12)

    electron_calc = NeutrinoElectronCalculator(electron_target=cf4_electron_target())
    if electron_calc.differential_cross_section_cm2_per_kev("nue", 30.0, 1000.0) <= 0.0:
        raise AssertionError("nu-e cross section should be positive at the test point")


def check_detector_inputs() -> None:
    load_config(str(REPO_DIR / "configs" / "detector_config.json"))
    for path, label in ((DEFAULT_CARBON_QF_CSV, "carbon"), (DEFAULT_FLUORINE_QF_CSV, "fluorine")):
        curve = load_quenching_curve(path, label)
        if np.any(np.diff(curve.electron_equivalent_energy(curve.recoil_energy_kev)) <= 0.0):
            raise AssertionError(f"{label} QF mapping is not monotonic")


def main() -> None:
    check_flux_modules()
    check_cross_sections()
    check_detector_inputs()
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()

