from __future__ import annotations

from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import ESS_flux
from cevens import CEvNSCalculator, fluorine19_target


def main() -> None:
    beam = ESS_flux.ESSBeamConfig()
    distance_m = 20.0
    target = fluorine19_target(axial_model="hoferichter_19f_central")
    calc = CEvNSCalculator()

    enu_mev = 30.0
    recoil_kev = 5.0
    total = calc.differential_cross_section_cm2_per_kev(target, enu_mev, recoil_kev)
    vector = calc.differential_vector_cross_section_cm2_per_kev(target, enu_mev, recoil_kev)
    axial = calc.differential_axial_cross_section_cm2_per_kev(target, enu_mev, recoil_kev)
    prompt_flux = ESS_flux.prompt_numu_line_flux(distance_m, beam=beam)

    print(f"Target                         : {target.name}")
    print(f"E_nu                           : {enu_mev:.3f} MeV")
    print(f"E_r                            : {recoil_kev:.3f} keV")
    print(f"d sigma / dE_r total           : {total:.6e} cm^2/keV")
    print(f"d sigma / dE_r vector          : {vector:.6e} cm^2/keV")
    print(f"d sigma / dE_r axial           : {axial:.6e} cm^2/keV")
    print(f"ESS prompt nu_mu line flux      : {prompt_flux:.6e} cm^-2 s^-1")


if __name__ == "__main__":
    main()

