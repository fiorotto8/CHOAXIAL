from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np


ELEM_CHARGE_J = 1.602176634e-19  # J / eV
M_PI_MEV = 139.57039
M_MU_MEV = 105.6583755

E_NU_MU_PROMPT_MEV = (M_PI_MEV**2 - M_MU_MEV**2) / (2.0 * M_PI_MEV)
E_NU_MAX_MEV = M_MU_MEV / 2.0


class DARBeamLike(Protocol):
    neutrino_yield_per_proton_per_flavor: float
    neutrinos_per_second_per_flavor: float


def michel_spectrum_nue(E_MeV: np.ndarray | float) -> np.ndarray:
    """Normalized delayed nu_e spectrum from mu+ DAR in MeV^-1."""
    E = np.asarray(E_MeV, dtype=float)
    out = np.zeros_like(E)
    mask = (E >= 0.0) & (E <= E_NU_MAX_MEV)
    x = E[mask] / M_MU_MEV
    out[mask] = (192.0 / M_MU_MEV) * x**2 * (0.5 - x)
    return out


def michel_spectrum_numubar(E_MeV: np.ndarray | float) -> np.ndarray:
    """Normalized delayed anti-nu_mu spectrum from mu+ DAR in MeV^-1."""
    E = np.asarray(E_MeV, dtype=float)
    out = np.zeros_like(E)
    mask = (E >= 0.0) & (E <= E_NU_MAX_MEV)
    x = E[mask] / M_MU_MEV
    out[mask] = (64.0 / M_MU_MEV) * x**2 * (0.75 - x)
    return out


def isotropic_geometry_factor_cm2(distance_m: float) -> float:
    """Point-source dilution factor 1 / (4 pi L^2), with L converted to cm."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive.")
    r_cm = distance_m * 100.0
    return 1.0 / (4.0 * np.pi * r_cm**2)


def delayed_shape(E_MeV: np.ndarray | float, flavor: str) -> np.ndarray:
    if flavor == "nue":
        return michel_spectrum_nue(E_MeV)
    if flavor == "numubar":
        return michel_spectrum_numubar(E_MeV)
    raise ValueError("flavor must be 'nue' or 'numubar'")


def binned_line_density(
    E_edges_MeV: np.ndarray,
    line_energy_MeV: float,
    line_intensity: float,
) -> np.ndarray:
    """Represent a monochromatic line as a histogram-bin average density.

    The physical prompt nu_mu contribution is a delta-function line in energy.
    For binned outputs, the integrated line intensity is placed into the bin
    containing the line energy and divided by that bin width.
    """
    edges = np.asarray(E_edges_MeV, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("E_edges_MeV must be a one-dimensional edge array.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("E_edges_MeV must be strictly increasing.")

    vals = np.zeros(len(edges) - 1, dtype=float)
    idx = np.searchsorted(edges, line_energy_MeV, side="right") - 1
    if 0 <= idx < len(vals):
        width = edges[idx + 1] - edges[idx]
        vals[idx] = line_intensity / width
    return vals


@dataclass(frozen=True)
class StoppedPionDARFluxModel:
    """Reusable energy-space stopped-pion DAR flux model.

    Facility-specific scripts supply only their beam metadata and output labels.
    Timing fields live on the beam config objects, but this class does not fold
    timing into the energy spectrum.
    """

    label: str
    output_dir: str
    file_prefix: str
    beam_factory: Callable[[], DARBeamLike]
    prompt_energy_mev: float = E_NU_MU_PROMPT_MEV
    delayed_endpoint_mev: float = E_NU_MAX_MEV

    def beam_or_default(self, beam: DARBeamLike | None = None) -> DARBeamLike:
        return beam if beam is not None else self.beam_factory()

    def prompt_numu_line_flux(self, distance_m: float, beam: DARBeamLike | None = None) -> float:
        beam = self.beam_or_default(beam)
        return beam.neutrinos_per_second_per_flavor * isotropic_geometry_factor_cm2(distance_m)

    def prompt_numu_line_fluence_per_pot(
        self,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> float:
        beam = self.beam_or_default(beam)
        return beam.neutrino_yield_per_proton_per_flavor * isotropic_geometry_factor_cm2(distance_m)

    def differential_flux_delayed(
        self,
        E_MeV: np.ndarray | float,
        distance_m: float,
        flavor: str,
        beam: DARBeamLike | None = None,
    ) -> np.ndarray:
        beam = self.beam_or_default(beam)
        return (
            beam.neutrinos_per_second_per_flavor
            * isotropic_geometry_factor_cm2(distance_m)
            * delayed_shape(E_MeV, flavor)
        )

    def differential_fluence_delayed_per_pot(
        self,
        E_MeV: np.ndarray | float,
        distance_m: float,
        flavor: str,
        beam: DARBeamLike | None = None,
    ) -> np.ndarray:
        beam = self.beam_or_default(beam)
        return (
            beam.neutrino_yield_per_proton_per_flavor
            * isotropic_geometry_factor_cm2(distance_m)
            * delayed_shape(E_MeV, flavor)
        )

    def binned_prompt_numu_flux(
        self,
        E_edges_MeV: np.ndarray,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> np.ndarray:
        return binned_line_density(
            E_edges_MeV,
            self.prompt_energy_mev,
            self.prompt_numu_line_flux(distance_m, beam=beam),
        )

    def binned_prompt_numu_fluence_per_pot(
        self,
        E_edges_MeV: np.ndarray,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> np.ndarray:
        return binned_line_density(
            E_edges_MeV,
            self.prompt_energy_mev,
            self.prompt_numu_line_fluence_per_pot(distance_m, beam=beam),
        )

    def total_differential_flux(
        self,
        E_MeV: np.ndarray,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> dict[str, np.ndarray]:
        E = np.asarray(E_MeV, dtype=float)
        phi_nue = self.differential_flux_delayed(E, distance_m, "nue", beam=beam)
        phi_numubar = self.differential_flux_delayed(E, distance_m, "numubar", beam=beam)
        return {
            "E_MeV": E,
            "phi_nue": phi_nue,
            "phi_numubar": phi_numubar,
            "phi_delayed_sum": phi_nue + phi_numubar,
        }

    def total_differential_fluence_per_pot(
        self,
        E_MeV: np.ndarray,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> dict[str, np.ndarray]:
        E = np.asarray(E_MeV, dtype=float)
        Phi_nue = self.differential_fluence_delayed_per_pot(E, distance_m, "nue", beam=beam)
        Phi_numubar = self.differential_fluence_delayed_per_pot(E, distance_m, "numubar", beam=beam)
        return {
            "E_MeV": E,
            "Phi_nue": Phi_nue,
            "Phi_numubar": Phi_numubar,
            "Phi_delayed_sum": Phi_nue + Phi_numubar,
        }

    def binned_total_flux(
        self,
        E_edges_MeV: np.ndarray,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> dict[str, np.ndarray]:
        edges = np.asarray(E_edges_MeV, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        phi_numu_prompt_binned = self.binned_prompt_numu_flux(edges, distance_m, beam=beam)
        phi_nue = self.differential_flux_delayed(centers, distance_m, "nue", beam=beam)
        phi_numubar = self.differential_flux_delayed(centers, distance_m, "numubar", beam=beam)
        return {
            "E_low_MeV": edges[:-1],
            "E_high_MeV": edges[1:],
            "E_center_MeV": centers,
            "phi_numu_prompt_binned": phi_numu_prompt_binned,
            "phi_nue": phi_nue,
            "phi_numubar": phi_numubar,
            "phi_total": phi_numu_prompt_binned + phi_nue + phi_numubar,
        }

    def binned_total_fluence_per_pot(
        self,
        E_edges_MeV: np.ndarray,
        distance_m: float,
        beam: DARBeamLike | None = None,
    ) -> dict[str, np.ndarray]:
        edges = np.asarray(E_edges_MeV, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        Phi_numu_prompt_binned = self.binned_prompt_numu_fluence_per_pot(edges, distance_m, beam=beam)
        Phi_nue = self.differential_fluence_delayed_per_pot(centers, distance_m, "nue", beam=beam)
        Phi_numubar = self.differential_fluence_delayed_per_pot(centers, distance_m, "numubar", beam=beam)
        return {
            "E_low_MeV": edges[:-1],
            "E_high_MeV": edges[1:],
            "E_center_MeV": centers,
            "Phi_numu_prompt_binned": Phi_numu_prompt_binned,
            "Phi_nue": Phi_nue,
            "Phi_numubar": Phi_numubar,
            "Phi_total": Phi_numu_prompt_binned + Phi_nue + Phi_numubar,
        }

    def plot_point_fluxes(
        self,
        filename: str,
        E_MeV: np.ndarray,
        phi_nue: np.ndarray,
        phi_numubar: np.ndarray,
        phi_delayed_sum: np.ndarray,
    ) -> None:
        plot_point_fluxes(filename, E_MeV, phi_nue, phi_numubar, phi_delayed_sum, self.label)

    def plot_binned_total_flux(self, filename: str, binned_flux: dict[str, np.ndarray]) -> None:
        plot_binned_total_flux(filename, binned_flux, self.label)

    def plot_point_fluence_per_pot(
        self,
        filename: str,
        E_MeV: np.ndarray,
        Phi_nue: np.ndarray,
        Phi_numubar: np.ndarray,
        Phi_delayed_sum: np.ndarray,
    ) -> None:
        plot_point_fluence_per_pot(filename, E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum, self.label)

    def plot_binned_total_fluence_per_pot(
        self,
        filename: str,
        binned_fluence: dict[str, np.ndarray],
    ) -> None:
        plot_binned_total_fluence_per_pot(filename, binned_fluence, self.label)

    def generate_standard_outputs(
        self,
        distance_m: float,
        beam: DARBeamLike | None = None,
        *,
        n_point: int = 500,
        n_edges: int = 265,
    ) -> list[str]:
        beam = self.beam_or_default(beam)
        os.makedirs(self.output_dir, exist_ok=True)

        E = np.linspace(0.0, self.delayed_endpoint_mev, n_point)
        E_edges = np.linspace(0.0, self.delayed_endpoint_mev, n_edges)

        flux_point = self.total_differential_flux(E, distance_m, beam=beam)
        fluence_point = self.total_differential_fluence_per_pot(E, distance_m, beam=beam)
        flux_binned = self.binned_total_flux(E_edges, distance_m, beam=beam)
        fluence_binned = self.binned_total_fluence_per_pot(E_edges, distance_m, beam=beam)

        paths = {
            "flux_point_csv": f"{self.output_dir}/{self.file_prefix}_delayed_flux_point_grid.csv",
            "flux_binned_csv": f"{self.output_dir}/{self.file_prefix}_total_flux_binned.csv",
            "fluence_point_csv": f"{self.output_dir}/{self.file_prefix}_delayed_fluence_per_pot_point_grid.csv",
            "fluence_binned_csv": f"{self.output_dir}/{self.file_prefix}_total_fluence_per_pot_binned.csv",
            "flux_point_png": f"{self.output_dir}/{self.file_prefix}_delayed_flux_point_grid.png",
            "flux_binned_png": f"{self.output_dir}/{self.file_prefix}_total_flux_binned.png",
            "fluence_point_png": f"{self.output_dir}/{self.file_prefix}_delayed_fluence_per_pot_point_grid.png",
            "fluence_binned_png": f"{self.output_dir}/{self.file_prefix}_total_fluence_per_pot_binned.png",
        }

        save_point_flux_csv(
            paths["flux_point_csv"],
            flux_point["E_MeV"],
            flux_point["phi_nue"],
            flux_point["phi_numubar"],
            flux_point["phi_delayed_sum"],
        )
        save_binned_flux_csv(paths["flux_binned_csv"], flux_binned)
        save_point_fluence_per_pot_csv(
            paths["fluence_point_csv"],
            fluence_point["E_MeV"],
            fluence_point["Phi_nue"],
            fluence_point["Phi_numubar"],
            fluence_point["Phi_delayed_sum"],
        )
        save_binned_fluence_per_pot_csv(paths["fluence_binned_csv"], fluence_binned)

        self.plot_point_fluxes(
            paths["flux_point_png"],
            flux_point["E_MeV"],
            flux_point["phi_nue"],
            flux_point["phi_numubar"],
            flux_point["phi_delayed_sum"],
        )
        self.plot_binned_total_flux(paths["flux_binned_png"], flux_binned)
        self.plot_point_fluence_per_pot(
            paths["fluence_point_png"],
            fluence_point["E_MeV"],
            fluence_point["Phi_nue"],
            fluence_point["Phi_numubar"],
            fluence_point["Phi_delayed_sum"],
        )
        self.plot_binned_total_fluence_per_pot(paths["fluence_binned_png"], fluence_binned)

        print(f"Peak nue average differential flux      : {flux_point['phi_nue'].max():.6e} /cm^2/s/MeV")
        print(f"Peak numubar average differential flux  : {flux_point['phi_numubar'].max():.6e} /cm^2/s/MeV")
        print(f"Peak nue fluence per POT                : {fluence_point['Phi_nue'].max():.6e} /cm^2/POT/MeV")
        print(f"Peak numubar fluence per POT            : {fluence_point['Phi_numubar'].max():.6e} /cm^2/POT/MeV")

        return list(paths.values())


def save_point_flux_csv(
    filename: str,
    E_MeV: np.ndarray,
    phi_nue: np.ndarray,
    phi_numubar: np.ndarray,
    phi_delayed_sum: np.ndarray,
) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["E_MeV", "phi_nue_per_cm2_s_MeV", "phi_numubar_per_cm2_s_MeV", "phi_delayed_sum_per_cm2_s_MeV"])
        writer.writerows(zip(E_MeV, phi_nue, phi_numubar, phi_delayed_sum))


def save_binned_flux_csv(filename: str, binned_flux: dict[str, np.ndarray]) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "E_low_MeV",
            "E_high_MeV",
            "E_center_MeV",
            "phi_numu_prompt_per_cm2_s_MeV",
            "phi_nue_per_cm2_s_MeV",
            "phi_numubar_per_cm2_s_MeV",
            "phi_total_per_cm2_s_MeV",
        ])
        writer.writerows(zip(
            binned_flux["E_low_MeV"],
            binned_flux["E_high_MeV"],
            binned_flux["E_center_MeV"],
            binned_flux["phi_numu_prompt_binned"],
            binned_flux["phi_nue"],
            binned_flux["phi_numubar"],
            binned_flux["phi_total"],
        ))


def save_point_fluence_per_pot_csv(
    filename: str,
    E_MeV: np.ndarray,
    Phi_nue: np.ndarray,
    Phi_numubar: np.ndarray,
    Phi_delayed_sum: np.ndarray,
) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["E_MeV", "Phi_nue_per_cm2_POT_MeV", "Phi_numubar_per_cm2_POT_MeV", "Phi_delayed_sum_per_cm2_POT_MeV"])
        writer.writerows(zip(E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum))


def save_binned_fluence_per_pot_csv(filename: str, binned_fluence: dict[str, np.ndarray]) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "E_low_MeV",
            "E_high_MeV",
            "E_center_MeV",
            "Phi_numu_prompt_per_cm2_POT_MeV",
            "Phi_nue_per_cm2_POT_MeV",
            "Phi_numubar_per_cm2_POT_MeV",
            "Phi_total_per_cm2_POT_MeV",
        ])
        writer.writerows(zip(
            binned_fluence["E_low_MeV"],
            binned_fluence["E_high_MeV"],
            binned_fluence["E_center_MeV"],
            binned_fluence["Phi_numu_prompt_binned"],
            binned_fluence["Phi_nue"],
            binned_fluence["Phi_numubar"],
            binned_fluence["Phi_total"],
        ))


def plot_point_fluxes(
    filename: str,
    E_MeV: np.ndarray,
    phi_nue: np.ndarray,
    phi_numubar: np.ndarray,
    phi_delayed_sum: np.ndarray,
    source_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(E_MeV, phi_nue, label=r"$\nu_e$")
    plt.plot(E_MeV, phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(E_MeV, phi_delayed_sum, linestyle="--", label="delayed sum")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential flux [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title(f"{source_label} delayed neutrino differential flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_binned_total_flux(
    filename: str,
    binned_flux: dict[str, np.ndarray],
    source_label: str,
) -> None:
    centers = binned_flux["E_center_MeV"]
    phi_numu_prompt_binned = binned_flux["phi_numu_prompt_binned"]
    phi_nue = binned_flux["phi_nue"]
    phi_numubar = binned_flux["phi_numubar"]
    phi_total = binned_flux["phi_total"]

    plt.figure(figsize=(8, 5))
    plt.step(centers, phi_numu_prompt_binned, where="mid", label=r"prompt $\nu_\mu$")
    plt.plot(centers, phi_nue, label=r"$\nu_e$")
    plt.plot(centers, phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(centers, phi_total, linewidth=2, label="total")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential flux [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title(f"{source_label} total neutrino differential flux (binned)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    _set_positive_ylim(phi_numu_prompt_binned, phi_nue, phi_numubar, phi_total)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_point_fluence_per_pot(
    filename: str,
    E_MeV: np.ndarray,
    Phi_nue: np.ndarray,
    Phi_numubar: np.ndarray,
    Phi_delayed_sum: np.ndarray,
    source_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(E_MeV, Phi_nue, label=r"$\nu_e$")
    plt.plot(E_MeV, Phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(E_MeV, Phi_delayed_sum, linestyle="--", label="delayed sum")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential fluence [cm$^{-2}$ POT$^{-1}$ MeV$^{-1}$]")
    plt.title(f"{source_label} delayed neutrino differential fluence per POT")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_binned_total_fluence_per_pot(
    filename: str,
    binned_fluence: dict[str, np.ndarray],
    source_label: str,
) -> None:
    centers = binned_fluence["E_center_MeV"]
    Phi_numu_prompt_binned = binned_fluence["Phi_numu_prompt_binned"]
    Phi_nue = binned_fluence["Phi_nue"]
    Phi_numubar = binned_fluence["Phi_numubar"]
    Phi_total = binned_fluence["Phi_total"]

    plt.figure(figsize=(8, 5))
    plt.step(centers, Phi_numu_prompt_binned, where="mid", label=r"prompt $\nu_\mu$")
    plt.plot(centers, Phi_nue, label=r"$\nu_e$")
    plt.plot(centers, Phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(centers, Phi_total, linewidth=2, label="total")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential fluence [cm$^{-2}$ POT$^{-1}$ MeV$^{-1}$]")
    plt.title(f"{source_label} total neutrino differential fluence per POT (binned)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    _set_positive_ylim(Phi_numu_prompt_binned, Phi_nue, Phi_numubar, Phi_total)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def _set_positive_ylim(*series: np.ndarray) -> None:
    positive = np.concatenate([np.asarray(values, dtype=float) for values in series])
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)


def print_saved_files(paths: Iterable[str]) -> None:
    print("\nSaved files:")
    for path in paths:
        print(f"  {path}")

