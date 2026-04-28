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


TAU_PI_S = 26.033e-9
TAU_MU_S = 2.1969811e-6
OUTPUT_DIR = "sns_flux_output"


def _yield_poly_aluminum(proton_energy_GeV: float) -> float:
    p3, p2, p1, p0 = 0.28, -1.12, 1.79, -0.68
    E = proton_energy_GeV
    return p3 * E**3 + p2 * E**2 + p1 * E + p0


def _yield_poly_inconel(proton_energy_GeV: float) -> float:
    p3, p2, p1, p0 = 0.27, -1.09, 1.75, -0.67
    E = proton_energy_GeV
    return p3 * E**3 + p2 * E**2 + p1 * E + p0


class SNSBeamConfig:
    """SNS FTS beam metadata for the semi-analytic DAR benchmark.

    The PBW polynomial is an effective source-normalization model used only for
    the neutrino yield.  The beam spill and lifetimes are reported as metadata;
    they are not folded into the energy-space flux shape.
    """

    def __init__(
        self,
        beam_power_MW: float = 1.4,
        proton_energy_GeV: float = 1.0,
        repetition_rate_Hz: float = 60.0,
        beam_spill_fwhm_s: float = 350e-9,
        neutrino_yield_per_proton_per_flavor: float | None = None,
        pbw_model: str = "aluminum",
        use_energy_dependent_yield: bool = True,
        yield_fractional_uncertainty: float = 0.10,
    ) -> None:
        pbw_model = pbw_model.lower()
        if pbw_model not in {"aluminum", "inconel"}:
            raise ValueError("pbw_model must be 'aluminum' or 'inconel'")

        self.beam_power_MW = beam_power_MW
        self.proton_energy_GeV = proton_energy_GeV
        self.repetition_rate_Hz = repetition_rate_Hz
        self.beam_spill_fwhm_s = beam_spill_fwhm_s
        self.pbw_model = pbw_model
        self.use_energy_dependent_yield = use_energy_dependent_yield
        self.yield_fractional_uncertainty = yield_fractional_uncertainty

        if neutrino_yield_per_proton_per_flavor is None:
            if use_energy_dependent_yield:
                self.neutrino_yield_per_proton_per_flavor = self.total_neutrino_yield_per_proton / 3.0
            else:
                self.neutrino_yield_per_proton_per_flavor = 0.0874
        else:
            self.neutrino_yield_per_proton_per_flavor = neutrino_yield_per_proton_per_flavor

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
        return self.repetition_rate_Hz * self.beam_spill_fwhm_s

    @property
    def total_neutrino_yield_per_proton(self) -> float:
        if self.pbw_model == "aluminum":
            return _yield_poly_aluminum(self.proton_energy_GeV)
        return _yield_poly_inconel(self.proton_energy_GeV)

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

    @property
    def pot_per_year_at_5000h(self) -> float:
        return self.protons_per_second * 5000.0 * 3600.0

    @property
    def neutrinos_per_year_per_flavor_at_5000h(self) -> float:
        return self.neutrinos_per_second_per_flavor * 5000.0 * 3600.0


MODEL = StoppedPionDARFluxModel(
    label="SNS",
    output_dir=OUTPUT_DIR,
    file_prefix="sns",
    beam_factory=SNSBeamConfig,
)


def prompt_numu_line_flux(distance_m: float, beam: SNSBeamConfig | None = None) -> float:
    return MODEL.prompt_numu_line_flux(distance_m, beam=beam)


def prompt_numu_line_fluence_per_pot(distance_m: float, beam: SNSBeamConfig | None = None) -> float:
    return MODEL.prompt_numu_line_fluence_per_pot(distance_m, beam=beam)


def differential_flux_delayed(E_MeV, distance_m: float, flavor: str, beam: SNSBeamConfig | None = None):
    return MODEL.differential_flux_delayed(E_MeV, distance_m, flavor, beam=beam)


def differential_fluence_delayed_per_pot(E_MeV, distance_m: float, flavor: str, beam: SNSBeamConfig | None = None):
    return MODEL.differential_fluence_delayed_per_pot(E_MeV, distance_m, flavor, beam=beam)


def binned_prompt_numu_flux(E_edges_MeV, distance_m: float, beam: SNSBeamConfig | None = None):
    return MODEL.binned_prompt_numu_flux(E_edges_MeV, distance_m, beam=beam)


def binned_prompt_numu_fluence_per_pot(E_edges_MeV, distance_m: float, beam: SNSBeamConfig | None = None):
    return MODEL.binned_prompt_numu_fluence_per_pot(E_edges_MeV, distance_m, beam=beam)


def total_differential_flux(E_MeV, distance_m: float, beam: SNSBeamConfig | None = None):
    return MODEL.total_differential_flux(E_MeV, distance_m, beam=beam)


def total_differential_fluence_per_pot(E_MeV, distance_m: float, beam: SNSBeamConfig | None = None):
    return MODEL.total_differential_fluence_per_pot(E_MeV, distance_m, beam=beam)


def binned_total_flux(E_edges_MeV, distance_m: float, beam: SNSBeamConfig | None = None):
    return MODEL.binned_total_flux(E_edges_MeV, distance_m, beam=beam)


def binned_total_fluence_per_pot(E_edges_MeV, distance_m: float, beam: SNSBeamConfig | None = None):
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
    beam = SNSBeamConfig()
    distance_m = 20.0

    print("=== SNS FTS DAR benchmark summary ===")
    print(f"Beam power                       : {beam.beam_power_MW:.3f} MW")
    print(f"Proton kinetic energy            : {beam.proton_energy_GeV:.3f} GeV")
    print(f"Repetition rate                  : {beam.repetition_rate_Hz:.3f} Hz")
    print(f"Beam spill FWHM                  : {beam.beam_spill_fwhm_s * 1e9:.1f} ns")
    print(f"Pion lifetime                    : {TAU_PI_S * 1e9:.1f} ns")
    print(f"Muon lifetime                    : {TAU_MU_S * 1e6:.3f} us")
    print("Timing note                      : metadata only; not folded into flux shape")
    print(f"PBW model                        : {beam.pbw_model}")
    print(f"Protons/s                        : {beam.protons_per_second:.6e}")
    print(f"Protons/pulse                    : {beam.protons_per_pulse:.6e}")
    print(f"Yield per proton per flavor      : {beam.neutrino_yield_per_proton_per_flavor:.5f}")
    print(f"Total neutrino yield / POT       : {beam.total_neutrino_yield_per_proton:.5f}")
    print(f"Yield fractional uncertainty     : {beam.yield_fractional_uncertainty:.3%}")
    print(f"Neutrinos/s per flavor           : {beam.neutrinos_per_second_per_flavor:.6e}")
    print(f"Neutrinos/pulse per flavor       : {beam.neutrinos_per_pulse_per_flavor:.6e}")
    print(f"Neutrinos/year per flavor        : {beam.neutrinos_per_year_per_flavor:.6e}")
    print(f"POT/year at 5000 h               : {beam.pot_per_year_at_5000h:.6e}")
    print(f"Neutrinos/year/flavor at 5000 h  : {beam.neutrinos_per_year_per_flavor_at_5000h:.6e}")
    print(f"Prompt nu_mu line energy         : {E_NU_MU_PROMPT_MEV:.6f} MeV")
    print(f"Delayed endpoint                 : {E_NU_MAX_MEV:.6f} MeV")
    print(f"Chosen baseline                  : {distance_m:.1f} m")
    print(f"Prompt nu_mu line flux           : {prompt_numu_line_flux(distance_m, beam=beam):.6e} /cm^2/s")
    print(f"Prompt nu_mu line fluence / POT  : {prompt_numu_line_fluence_per_pot(distance_m, beam=beam):.6e} /cm^2/POT")

    print_saved_files(MODEL.generate_standard_outputs(distance_m, beam=beam))


if __name__ == "__main__":
    main()
