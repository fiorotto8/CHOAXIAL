from __future__ import annotations

from cohaxial.dar_flux import (
    ELEM_CHARGE_J,
    E_NU_MAX_MEV,
    E_NU_MU_PROMPT_MEV,
    M_MU_MEV,
    M_PI_MEV,
    StoppedPionDARFluxModel,
    binned_line_density as _binned_line_density,
    delayed_shape as _delayed_shape,
    isotropic_geometry_factor_cm2,
    michel_spectrum_nue,
    michel_spectrum_numubar,
    print_saved_files,
    save_binned_fluence_per_pot_csv,
    save_binned_flux_csv,
    save_point_fluence_per_pot_csv,
    save_point_flux_csv,
)


OUTPUT_DIR = "ess_flux_output"


class ESSBeamConfig:
    """ESS benchmark beam metadata for semi-analytic DAR source estimates.

    The timing fields are kept as metadata only.  This module models the
    energy-space stopped-pion DAR spectrum and does not fold the pulse profile
    into a time-dependent flux.
    """

    def __init__(
        self,
        beam_power_MW: float = 5.0,
        proton_energy_GeV: float = 2.0,
        repetition_rate_Hz: float = 14.0,
        pulse_length_s: float = 2.86e-3,
        neutrino_yield_per_proton_per_flavor: float = 0.3,
        yield_fractional_uncertainty: float = 0.0,
    ) -> None:
        self.beam_power_MW = beam_power_MW
        self.proton_energy_GeV = proton_energy_GeV
        self.repetition_rate_Hz = repetition_rate_Hz
        self.pulse_length_s = pulse_length_s
        self.neutrino_yield_per_proton_per_flavor = neutrino_yield_per_proton_per_flavor
        self.yield_fractional_uncertainty = yield_fractional_uncertainty

    @property
    def beam_power_W(self) -> float:
        return self.beam_power_MW * 1e6

    @property
    def proton_energy_J(self) -> float:
        return self.proton_energy_GeV * 1e9 * ELEM_CHARGE_J

    @property
    def protons_per_second(self) -> float:
        return self.beam_power_W / self.proton_energy_J

    @property
    def protons_per_pulse(self) -> float:
        return self.protons_per_second / self.repetition_rate_Hz

    @property
    def duty_factor(self) -> float:
        return self.repetition_rate_Hz * self.pulse_length_s

    @property
    def neutrinos_per_second_per_flavor(self) -> float:
        return self.protons_per_second * self.neutrino_yield_per_proton_per_flavor

    @property
    def neutrinos_per_pulse_per_flavor(self) -> float:
        return self.protons_per_pulse * self.neutrino_yield_per_proton_per_flavor

    @property
    def neutrinos_per_year_per_flavor(self) -> float:
        seconds_per_year = 365.25 * 24.0 * 3600.0
        return self.neutrinos_per_second_per_flavor * seconds_per_year


MODEL = StoppedPionDARFluxModel(
    label="ESS",
    output_dir=OUTPUT_DIR,
    file_prefix="ess",
    beam_factory=ESSBeamConfig,
)


def prompt_numu_line_flux(distance_m: float, beam: ESSBeamConfig | None = None) -> float:
    return MODEL.prompt_numu_line_flux(distance_m, beam=beam)


def prompt_numu_line_fluence_per_pot(distance_m: float, beam: ESSBeamConfig | None = None) -> float:
    return MODEL.prompt_numu_line_fluence_per_pot(distance_m, beam=beam)


def differential_flux_delayed(E_MeV, distance_m: float, flavor: str, beam: ESSBeamConfig | None = None):
    return MODEL.differential_flux_delayed(E_MeV, distance_m, flavor, beam=beam)


def differential_fluence_delayed_per_pot(E_MeV, distance_m: float, flavor: str, beam: ESSBeamConfig | None = None):
    return MODEL.differential_fluence_delayed_per_pot(E_MeV, distance_m, flavor, beam=beam)


def binned_prompt_numu_flux(E_edges_MeV, distance_m: float, beam: ESSBeamConfig | None = None):
    return MODEL.binned_prompt_numu_flux(E_edges_MeV, distance_m, beam=beam)


def binned_prompt_numu_fluence_per_pot(E_edges_MeV, distance_m: float, beam: ESSBeamConfig | None = None):
    return MODEL.binned_prompt_numu_fluence_per_pot(E_edges_MeV, distance_m, beam=beam)


def total_differential_flux(E_MeV, distance_m: float, beam: ESSBeamConfig | None = None):
    return MODEL.total_differential_flux(E_MeV, distance_m, beam=beam)


def total_differential_fluence_per_pot(E_MeV, distance_m: float, beam: ESSBeamConfig | None = None):
    return MODEL.total_differential_fluence_per_pot(E_MeV, distance_m, beam=beam)


def binned_total_flux(E_edges_MeV, distance_m: float, beam: ESSBeamConfig | None = None):
    return MODEL.binned_total_flux(E_edges_MeV, distance_m, beam=beam)


def binned_total_fluence_per_pot(E_edges_MeV, distance_m: float, beam: ESSBeamConfig | None = None):
    return MODEL.binned_total_fluence_per_pot(E_edges_MeV, distance_m, beam=beam)


def plot_point_fluxes(filename: str, E_MeV, phi_nue, phi_numubar, phi_delayed_sum) -> None:
    MODEL.plot_point_fluxes(filename, E_MeV, phi_nue, phi_numubar, phi_delayed_sum)


def plot_binned_total_flux(filename: str, binned_flux) -> None:
    MODEL.plot_binned_total_flux(filename, binned_flux)


def plot_point_fluence_per_pot(filename: str, E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum) -> None:
    MODEL.plot_point_fluence_per_pot(filename, E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum)


def plot_binned_total_fluence_per_pot(filename: str, binned_fluence) -> None:
    MODEL.plot_binned_total_fluence_per_pot(filename, binned_fluence)


def main() -> None:
    beam = ESSBeamConfig()
    distance_m = 20.0

    print("=== ESS DAR benchmark summary ===")
    print(f"Beam power                       : {beam.beam_power_MW:.3f} MW")
    print(f"Proton kinetic energy            : {beam.proton_energy_GeV:.3f} GeV")
    print(f"Repetition rate                  : {beam.repetition_rate_Hz:.3f} Hz")
    print(f"Pulse length                     : {beam.pulse_length_s * 1e3:.3f} ms")
    print("Timing note                      : metadata only; not folded into flux shape")
    print(f"Protons/s                        : {beam.protons_per_second:.6e}")
    print(f"Protons/pulse                    : {beam.protons_per_pulse:.6e}")
    print(f"Yield per proton per flavor      : {beam.neutrino_yield_per_proton_per_flavor:.3f}")
    print(f"Yield fractional uncertainty     : {beam.yield_fractional_uncertainty:.3%}")
    print(f"Neutrinos/s per flavor           : {beam.neutrinos_per_second_per_flavor:.6e}")
    print(f"Neutrinos/pulse per flavor       : {beam.neutrinos_per_pulse_per_flavor:.6e}")
    print(f"Neutrinos/year per flavor        : {beam.neutrinos_per_year_per_flavor:.6e}")
    print(f"Prompt nu_mu line energy         : {E_NU_MU_PROMPT_MEV:.6f} MeV")
    print(f"Delayed endpoint                 : {E_NU_MAX_MEV:.6f} MeV")
    print(f"Chosen baseline                  : {distance_m:.1f} m")
    print(f"Prompt nu_mu line flux           : {prompt_numu_line_flux(distance_m, beam=beam):.6e} /cm^2/s")
    print(f"Prompt nu_mu line fluence / POT  : {prompt_numu_line_fluence_per_pot(distance_m, beam=beam):.6e} /cm^2/POT")

    print_saved_files(MODEL.generate_standard_outputs(distance_m, beam=beam))


if __name__ == "__main__":
    main()
