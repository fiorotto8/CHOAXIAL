from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


AVOGADRO = 6.02214076e23
GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
CF4_MOLAR_MASS_KG_PER_MOL = 88.0043e-3
CF4_ELECTRONS_PER_MOLECULE = 42.0

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join("configs", "detector_config.json")
DEFAULT_INPUT_NUCLEAR_RATE_CSV = os.path.join(
    "cevens_rate_output",
    "cf4_differential_rate_per_molecule.csv",
)
DEFAULT_INPUT_ELECTRON_RATE_CSV = os.path.join(
    "cevens_rate_output",
    "cf4_electron_differential_rate_per_molecule.csv",
)
DEFAULT_OUTPUT_DIR = "detector_rate_output"
DEFAULT_CARBON_QF_CSV = os.path.join(REPO_DIR, "QF", "C_in_CF4.csv")
DEFAULT_FLUORINE_QF_CSV = os.path.join(REPO_DIR, "QF", "F_in_CF4.csv")

REQUIRED_NUCLEAR_COLUMNS = (
    "Er_keV",
    "dR_dEr_C_prompt_total_per_s_per_keV_per_C",
    "dR_dEr_C_nue_total_per_s_per_keV_per_C",
    "dR_dEr_C_numubar_total_per_s_per_keV_per_C",
    "dR_dEr_C_delayed_total_per_s_per_keV_per_C",
    "dR_dEr_C_total_per_s_per_keV_per_C",
    "dR_dEr_F_prompt_total_per_s_per_keV_per_F",
    "dR_dEr_F_nue_total_per_s_per_keV_per_F",
    "dR_dEr_F_numubar_total_per_s_per_keV_per_F",
    "dR_dEr_F_delayed_total_per_s_per_keV_per_F",
    "dR_dEr_F_total_per_s_per_keV_per_F",
    "dR_dEr_CF4_prompt_total_per_s_per_keV_per_molecule",
    "dR_dEr_CF4_nue_total_per_s_per_keV_per_molecule",
    "dR_dEr_CF4_numubar_total_per_s_per_keV_per_molecule",
    "dR_dEr_CF4_delayed_total_per_s_per_keV_per_molecule",
    "dR_dEr_CF4_total_per_s_per_keV_per_molecule",
    "dR_dEr_CF4_C_piece_per_s_per_keV_per_molecule",
    "dR_dEr_CF4_4F_piece_per_s_per_keV_per_molecule",
    "CF4_F_fraction",
)

OPTIONAL_NUCLEAR_COLUMNS = (
    "dR_dEr_F_total_vector_per_s_per_keV_per_F",
    "dR_dEr_F_total_axial_per_s_per_keV_per_F",
    "F_axial_fraction",
)

REQUIRED_ELECTRON_COLUMNS = (
    "Te_keV",
    "dR_dTe_CF4_numu_prompt_per_s_per_keV_per_molecule",
    "dR_dTe_CF4_nue_delayed_per_s_per_keV_per_molecule",
    "dR_dTe_CF4_numubar_delayed_per_s_per_keV_per_molecule",
    "dR_dTe_CF4_delayed_total_per_s_per_keV_per_molecule",
    "dR_dTe_CF4_total_per_s_per_keV_per_molecule",
    "CF4_electrons_per_molecule",
)


@dataclass(frozen=True)
class DetectorConfig:
    input_nuclear_rate_csv: str
    input_electron_rate_csv: str | None
    output_dir: str
    radius_m: float
    length_m: float
    fiducial_fraction: float
    pressure_pa: float
    pressure_label: str
    temperature_k: float
    energy_threshold_kev: float


@dataclass(frozen=True)
class QuenchingCurve:
    label: str
    recoil_energy_kev: np.ndarray
    quenching_factor: np.ndarray

    def evaluate(self, recoil_energy_kev: np.ndarray) -> np.ndarray:
        # Piecewise-linear interpolation inside the tabulated range, with the
        # endpoint QF values held fixed below/above the table.
        return np.interp(
            recoil_energy_kev,
            self.recoil_energy_kev,
            self.quenching_factor,
            left=self.quenching_factor[0],
            right=self.quenching_factor[-1],
        )

    def electron_equivalent_energy(self, recoil_energy_kev: np.ndarray) -> np.ndarray:
        return recoil_energy_kev * self.evaluate(recoil_energy_kev)

    def recoil_threshold_for_ee(self, threshold_kevee: float) -> float:
        if threshold_kevee <= 0.0:
            return 0.0

        eee_grid_kev = self.electron_equivalent_energy(self.recoil_energy_kev)
        if threshold_kevee >= eee_grid_kev[-1]:
            return float("inf")

        return float(np.interp(threshold_kevee, eee_grid_kev, self.recoil_energy_kev))


def positive_float(value: object, label: str) -> float:
    out = float(value)
    if out <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return out


def read_radius_m(geometry: dict) -> float:
    if "radius_m" in geometry:
        return positive_float(geometry["radius_m"], "geometry.radius_m")
    if "diameter_m" in geometry:
        return 0.5 * positive_float(geometry["diameter_m"], "geometry.diameter_m")
    raise ValueError("geometry must define either radius_m or diameter_m.")


def read_length_m(geometry: dict) -> float:
    if "length_m" in geometry:
        return positive_float(geometry["length_m"], "geometry.length_m")
    if "height_m" in geometry:
        return positive_float(geometry["height_m"], "geometry.height_m")
    raise ValueError("geometry must define either length_m or height_m.")


def read_pressure_pa(gas: dict) -> tuple[float, str]:
    pressure_keys = (
        ("pressure_pa", 1.0, "Pa"),
        ("pressure_kpa", 1.0e3, "kPa"),
        ("pressure_mbar", 100.0, "mbar"),
        ("pressure_bar", 1.0e5, "bar"),
        ("pressure_torr", 133.32236842105263, "Torr"),
        ("pressure_atm", 101325.0, "atm"),
    )

    matches: list[tuple[float, str]] = []
    for key, factor, label in pressure_keys:
        if key in gas:
            raw_value = positive_float(gas[key], f"gas.{key}")
            matches.append((raw_value * factor, f"{raw_value:.6g} {label}"))

    if not matches:
        raise ValueError(
            "gas must define one pressure key: pressure_pa, pressure_kpa, "
            "pressure_mbar, pressure_bar, pressure_torr, or pressure_atm."
        )
    if len(matches) > 1:
        raise ValueError("gas must define exactly one pressure key.")

    return matches[0]


def load_config(
    config_path: str,
    nuclear_rate_csv_override: str | None = None,
    electron_rate_csv_override: str | None = None,
    output_dir_override: str | None = None,
) -> DetectorConfig:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    geometry = raw.get("geometry", {})
    gas = raw.get("gas", {})
    analysis = raw.get("analysis", {})

    radius_m = read_radius_m(geometry)
    length_m = read_length_m(geometry)

    fiducial_fraction = float(raw.get("fiducial_fraction", 1.0))
    if not 0.0 < fiducial_fraction <= 1.0:
        raise ValueError("fiducial_fraction must satisfy 0 < fiducial_fraction <= 1.")

    pressure_pa, pressure_label = read_pressure_pa(gas)
    temperature_k = positive_float(gas.get("temperature_K", 293.15), "gas.temperature_K")

    energy_threshold_kev = float(analysis.get("energy_threshold_kev", 0.0))
    if energy_threshold_kev < 0.0:
        raise ValueError("analysis.energy_threshold_kev must be non-negative.")

    input_nuclear_rate_csv = nuclear_rate_csv_override or raw.get(
        "input_nuclear_rate_csv",
        raw.get("input_rate_csv", DEFAULT_INPUT_NUCLEAR_RATE_CSV),
    )

    if electron_rate_csv_override is not None:
        input_electron_rate_csv = electron_rate_csv_override
    else:
        input_electron_rate_csv = raw.get("input_electron_rate_csv")
        if input_electron_rate_csv is None:
            input_electron_rate_csv = DEFAULT_INPUT_ELECTRON_RATE_CSV

    output_dir = output_dir_override or raw.get("output_dir", DEFAULT_OUTPUT_DIR)

    return DetectorConfig(
        input_nuclear_rate_csv=input_nuclear_rate_csv,
        input_electron_rate_csv=input_electron_rate_csv,
        output_dir=output_dir,
        radius_m=radius_m,
        length_m=length_m,
        fiducial_fraction=fiducial_fraction,
        pressure_pa=pressure_pa,
        pressure_label=pressure_label,
        temperature_k=temperature_k,
        energy_threshold_kev=energy_threshold_kev,
    )


def load_rate_table(
    filename: str,
    required_columns: tuple[str, ...],
    optional_columns: tuple[str, ...] = (),
) -> Dict[str, np.ndarray]:
    with open(filename, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{filename} does not contain a CSV header.")

        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(float(row[name]))

    table = {name: np.asarray(values, dtype=float) for name, values in columns.items()}

    missing = [name for name in required_columns if name not in table]
    if missing:
        raise ValueError(f"Missing required columns in {filename}: {', '.join(missing)}")

    energy_column = required_columns[0]
    energy_grid = table[energy_column]
    if energy_grid.size < 2:
        raise ValueError(f"{filename} must contain at least two recoil-energy bins.")
    if np.any(np.diff(energy_grid) <= 0.0):
        raise ValueError(f"{energy_column} in {filename} must be strictly increasing.")

    for name in required_columns:
        if not np.all(np.isfinite(table[name])):
            raise ValueError(f"Required column {name} contains non-finite values in {filename}.")

    for name in optional_columns:
        if name in table:
            table[name] = np.nan_to_num(table[name], nan=0.0, posinf=0.0, neginf=0.0)

    return table


def load_quenching_curve(filename: str, label: str) -> QuenchingCurve:
    data = np.loadtxt(filename, delimiter=",", dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{filename} must contain exactly two numeric columns: energy_keV, QF.")

    recoil_energy_kev = np.asarray(data[:, 0], dtype=float)
    quenching_factor = np.asarray(data[:, 1], dtype=float)

    if recoil_energy_kev.size < 2:
        raise ValueError(f"{filename} must contain at least two tabulated QF points.")
    if np.any(~np.isfinite(recoil_energy_kev)) or np.any(~np.isfinite(quenching_factor)):
        raise ValueError(f"{filename} contains non-finite QF values.")
    if np.any(np.diff(recoil_energy_kev) <= 0.0):
        raise ValueError(f"QF recoil energies must be strictly increasing in {filename}.")
    if np.any(quenching_factor <= 0.0):
        raise ValueError(f"QF values must stay positive in {filename}.")

    if recoil_energy_kev[0] > 0.0:
        # Extend the first tabulated QF down to zero recoil energy so the
        # low-energy extrapolation stays explicit and well-defined.
        recoil_energy_kev = np.concatenate(([0.0], recoil_energy_kev))
        quenching_factor = np.concatenate(([quenching_factor[0]], quenching_factor))

    eee_grid_kev = recoil_energy_kev * quenching_factor
    if np.any(np.diff(eee_grid_kev) <= 0.0):
        raise ValueError(f"Electron-equivalent mapping must be strictly increasing in {filename}.")

    return QuenchingCurve(
        label=label,
        recoil_energy_kev=recoil_energy_kev,
        quenching_factor=quenching_factor,
    )


def cylinder_volume_m3(radius_m: float, length_m: float) -> float:
    return math.pi * radius_m * radius_m * length_m


def ideal_gas_moles(pressure_pa: float, volume_m3: float, temperature_k: float) -> float:
    return pressure_pa * volume_m3 / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)


def hard_threshold(
    energy_grid_kev: np.ndarray,
    differential_rate: np.ndarray,
    threshold_kev: float,
) -> np.ndarray:
    return np.where(energy_grid_kev >= threshold_kev, differential_rate, 0.0)


def integrate_spectrum(energy_grid_kev: np.ndarray, differential_rate: np.ndarray) -> float:
    return float(np.trapz(differential_rate, energy_grid_kev))


def integrate_above_threshold(
    energy_grid_kev: np.ndarray,
    differential_rate: np.ndarray,
    threshold_kev: float,
) -> float:
    if threshold_kev <= energy_grid_kev[0]:
        return integrate_spectrum(energy_grid_kev, differential_rate)
    if threshold_kev >= energy_grid_kev[-1]:
        return 0.0

    threshold_value = float(np.interp(threshold_kev, energy_grid_kev, differential_rate))
    mask = energy_grid_kev > threshold_kev

    energy_tail = np.concatenate(([threshold_kev], energy_grid_kev[mask]))
    rate_tail = np.concatenate(([threshold_value], differential_rate[mask]))
    return integrate_spectrum(energy_tail, rate_tail)


def append_per_year_spectra(spectra: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(spectra)
    for key in list(spectra.keys()):
        if key.endswith("_per_s"):
            out[key.replace("_per_s", "_per_year")] = spectra[key] * SECONDS_PER_YEAR
    return out


def build_nuclear_species_detector_spectra(
    table: Dict[str, np.ndarray],
    molecules_fiducial: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    carbon_scale = molecules_fiducial
    fluorine_scale = 4.0 * molecules_fiducial

    carbon_spectra = {
        "energy_kev": table["Er_keV"],
        "prompt_per_s": carbon_scale * table["dR_dEr_C_prompt_total_per_s_per_keV_per_C"],
        "nue_per_s": carbon_scale * table["dR_dEr_C_nue_total_per_s_per_keV_per_C"],
        "numubar_per_s": carbon_scale * table["dR_dEr_C_numubar_total_per_s_per_keV_per_C"],
        "delayed_per_s": carbon_scale * table["dR_dEr_C_delayed_total_per_s_per_keV_per_C"],
        "total_per_s": carbon_scale * table["dR_dEr_C_total_per_s_per_keV_per_C"],
    }

    fluorine_spectra = {
        "energy_kev": table["Er_keV"],
        "prompt_per_s": fluorine_scale * table["dR_dEr_F_prompt_total_per_s_per_keV_per_F"],
        "nue_per_s": fluorine_scale * table["dR_dEr_F_nue_total_per_s_per_keV_per_F"],
        "numubar_per_s": fluorine_scale * table["dR_dEr_F_numubar_total_per_s_per_keV_per_F"],
        "delayed_per_s": fluorine_scale * table["dR_dEr_F_delayed_total_per_s_per_keV_per_F"],
        "total_per_s": fluorine_scale * table["dR_dEr_F_total_per_s_per_keV_per_F"],
    }

    if "dR_dEr_F_total_vector_per_s_per_keV_per_F" in table:
        fluorine_spectra["vector_per_s"] = fluorine_scale * table["dR_dEr_F_total_vector_per_s_per_keV_per_F"]
    if "dR_dEr_F_total_axial_per_s_per_keV_per_F" in table:
        fluorine_spectra["axial_per_s"] = fluorine_scale * table["dR_dEr_F_total_axial_per_s_per_keV_per_F"]

    return append_per_year_spectra(carbon_spectra), append_per_year_spectra(fluorine_spectra)


def combine_nuclear_species_detector_spectra(
    carbon_spectra: Dict[str, np.ndarray],
    fluorine_spectra: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    energy_grid_kev = carbon_spectra["energy_kev"]

    spectra = {"energy_kev": energy_grid_kev}
    for component in ("prompt", "nue", "numubar", "delayed", "total"):
        for suffix in ("per_s", "per_year"):
            key = f"{component}_{suffix}"
            spectra[key] = carbon_spectra[key] + fluorine_spectra[key]

    spectra["c_piece_per_s"] = carbon_spectra["total_per_s"]
    spectra["c_piece_per_year"] = carbon_spectra["total_per_year"]
    spectra["f_piece_per_s"] = fluorine_spectra["total_per_s"]
    spectra["f_piece_per_year"] = fluorine_spectra["total_per_year"]

    spectra["cf4_f_fraction"] = np.divide(
        spectra["f_piece_per_s"],
        spectra["total_per_s"],
        out=np.zeros_like(spectra["total_per_s"]),
        where=spectra["total_per_s"] > 0.0,
    )

    if "vector_per_s" in fluorine_spectra and "axial_per_s" in fluorine_spectra:
        spectra["f_vector_per_s"] = fluorine_spectra["vector_per_s"]
        spectra["f_vector_per_year"] = fluorine_spectra["vector_per_year"]
        spectra["f_axial_per_s"] = fluorine_spectra["axial_per_s"]
        spectra["f_axial_per_year"] = fluorine_spectra["axial_per_year"]
        spectra["f_axial_fraction"] = np.divide(
            spectra["f_axial_per_s"],
            spectra["f_piece_per_s"],
            out=np.zeros_like(spectra["f_piece_per_s"]),
            where=spectra["f_piece_per_s"] > 0.0,
        )

    return spectra


def build_nuclear_detector_spectra(
    table: Dict[str, np.ndarray],
    molecules_fiducial: float,
) -> Dict[str, np.ndarray]:
    carbon_spectra, fluorine_spectra = build_nuclear_species_detector_spectra(table, molecules_fiducial)
    return combine_nuclear_species_detector_spectra(carbon_spectra, fluorine_spectra)


def build_thresholded_recoil_species_spectra(
    spectra: Dict[str, np.ndarray],
    qf_curve: QuenchingCurve,
    threshold_kevee: float,
) -> Dict[str, np.ndarray]:
    threshold_recoil_kev = qf_curve.recoil_threshold_for_ee(threshold_kevee)
    energy_grid_kev = spectra["energy_kev"]

    thresholded = {"energy_kev": energy_grid_kev}
    for key, values in spectra.items():
        if key == "energy_kev":
            continue
        thresholded[key] = hard_threshold(energy_grid_kev, values, threshold_recoil_kev)

    return thresholded


def transform_species_recoil_spectra_to_ee(
    spectra: Dict[str, np.ndarray],
    qf_curve: QuenchingCurve,
) -> Dict[str, np.ndarray]:
    recoil_energy_kev = spectra["energy_kev"]
    energy_kevee = qf_curve.electron_equivalent_energy(recoil_energy_kev)
    d_eee_d_er = np.gradient(energy_kevee, recoil_energy_kev)

    if np.any(d_eee_d_er <= 0.0):
        raise ValueError(f"{qf_curve.label} QF produced a non-monotonic keVee mapping.")

    transformed = {"energy_kevee": energy_kevee}
    for key, values in spectra.items():
        if key == "energy_kev":
            continue
        transformed[key] = np.divide(
            values,
            d_eee_d_er,
            out=np.zeros_like(values),
            where=d_eee_d_er > 0.0,
        )

    return transformed


def interpolate_spectrum(
    target_grid: np.ndarray,
    source_grid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(target_grid, source_grid, values, left=0.0, right=0.0)


def build_nuclear_ee_detector_spectra(
    carbon_spectra: Dict[str, np.ndarray],
    fluorine_spectra: Dict[str, np.ndarray],
    carbon_qf: QuenchingCurve,
    fluorine_qf: QuenchingCurve,
) -> Dict[str, np.ndarray]:
    carbon_ee = transform_species_recoil_spectra_to_ee(carbon_spectra, carbon_qf)
    fluorine_ee = transform_species_recoil_spectra_to_ee(fluorine_spectra, fluorine_qf)

    energy_kevee = np.unique(
        np.concatenate((carbon_ee["energy_kevee"], fluorine_ee["energy_kevee"]))
    )
    spectra = {"energy_kevee": energy_kevee}

    for component in ("prompt", "nue", "numubar", "delayed", "total"):
        for suffix in ("per_s", "per_year"):
            key = f"{component}_{suffix}"
            spectra[key] = interpolate_spectrum(
                energy_kevee,
                carbon_ee["energy_kevee"],
                carbon_ee[key],
            ) + interpolate_spectrum(
                energy_kevee,
                fluorine_ee["energy_kevee"],
                fluorine_ee[key],
            )

    spectra["c_piece_per_s"] = interpolate_spectrum(
        energy_kevee,
        carbon_ee["energy_kevee"],
        carbon_ee["total_per_s"],
    )
    spectra["c_piece_per_year"] = interpolate_spectrum(
        energy_kevee,
        carbon_ee["energy_kevee"],
        carbon_ee["total_per_year"],
    )
    spectra["f_piece_per_s"] = interpolate_spectrum(
        energy_kevee,
        fluorine_ee["energy_kevee"],
        fluorine_ee["total_per_s"],
    )
    spectra["f_piece_per_year"] = interpolate_spectrum(
        energy_kevee,
        fluorine_ee["energy_kevee"],
        fluorine_ee["total_per_year"],
    )
    spectra["cf4_f_fraction"] = np.divide(
        spectra["f_piece_per_s"],
        spectra["total_per_s"],
        out=np.zeros_like(spectra["total_per_s"]),
        where=spectra["total_per_s"] > 0.0,
    )

    if "vector_per_s" in fluorine_ee and "axial_per_s" in fluorine_ee:
        spectra["f_vector_per_s"] = interpolate_spectrum(
            energy_kevee,
            fluorine_ee["energy_kevee"],
            fluorine_ee["vector_per_s"],
        )
        spectra["f_vector_per_year"] = interpolate_spectrum(
            energy_kevee,
            fluorine_ee["energy_kevee"],
            fluorine_ee["vector_per_year"],
        )
        spectra["f_axial_per_s"] = interpolate_spectrum(
            energy_kevee,
            fluorine_ee["energy_kevee"],
            fluorine_ee["axial_per_s"],
        )
        spectra["f_axial_per_year"] = interpolate_spectrum(
            energy_kevee,
            fluorine_ee["energy_kevee"],
            fluorine_ee["axial_per_year"],
        )
        spectra["f_axial_fraction"] = np.divide(
            spectra["f_axial_per_s"],
            spectra["f_piece_per_s"],
            out=np.zeros_like(spectra["f_piece_per_s"]),
            where=spectra["f_piece_per_s"] > 0.0,
        )

    return spectra


def validate_nuclear_ee_spectra(
    recoil_spectra: Dict[str, np.ndarray],
    ee_spectra: Dict[str, np.ndarray],
) -> None:
    recoil_total_rate = integrate_spectrum(recoil_spectra["energy_kev"], recoil_spectra["total_per_s"])
    ee_total_rate = integrate_spectrum(ee_spectra["energy_kevee"], ee_spectra["total_per_s"])

    if not math.isfinite(ee_total_rate) or ee_total_rate < 0.0:
        raise ValueError("The nuclear keVee spectrum produced a non-finite or negative total rate.")

    if recoil_total_rate > 0.0:
        fractional_shift = abs(ee_total_rate / recoil_total_rate - 1.0)
        if fractional_shift > 1.0e-2:
            raise ValueError(
                "The nuclear keVee remapping changed the total rate by more than 1%. "
                "Check the QF tables and the keVee interpolation."
            )


def build_electron_detector_spectra(
    table: Dict[str, np.ndarray],
    molecules_fiducial: float,
) -> Dict[str, np.ndarray]:
    scale = molecules_fiducial

    spectra = {
        "energy_kev": table["Te_keV"],
        "prompt_per_s": scale * table["dR_dTe_CF4_numu_prompt_per_s_per_keV_per_molecule"],
        "nue_per_s": scale * table["dR_dTe_CF4_nue_delayed_per_s_per_keV_per_molecule"],
        "numubar_per_s": scale * table["dR_dTe_CF4_numubar_delayed_per_s_per_keV_per_molecule"],
        "delayed_per_s": scale * table["dR_dTe_CF4_delayed_total_per_s_per_keV_per_molecule"],
        "total_per_s": scale * table["dR_dTe_CF4_total_per_s_per_keV_per_molecule"],
        "electrons_per_molecule": table["CF4_electrons_per_molecule"],
    }

    return append_per_year_spectra(spectra)


def write_nuclear_detector_csv(
    filename: str,
    spectra: Dict[str, np.ndarray],
    thresholded_spectra: Dict[str, np.ndarray],
) -> None:
    energy_grid_kev = spectra["energy_kev"]

    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        header = [
            "Er_keV",
            "dR_dEr_CF4_prompt_per_s_per_keV_detector",
            "dR_dEr_CF4_nue_per_s_per_keV_detector",
            "dR_dEr_CF4_numubar_per_s_per_keV_detector",
            "dR_dEr_CF4_delayed_per_s_per_keV_detector",
            "dR_dEr_CF4_total_per_s_per_keV_detector",
            "dR_dEr_CF4_total_above_threshold_per_s_per_keV_detector",
            "dN_dEr_CF4_prompt_per_keV_per_year_detector",
            "dN_dEr_CF4_prompt_above_threshold_per_keV_per_year_detector",
            "dN_dEr_CF4_delayed_per_keV_per_year_detector",
            "dN_dEr_CF4_delayed_above_threshold_per_keV_per_year_detector",
            "dN_dEr_CF4_total_per_keV_per_year_detector",
            "dN_dEr_CF4_total_above_threshold_per_keV_per_year_detector",
            "dN_dEr_CF4_C_piece_per_keV_per_year_detector",
            "dN_dEr_CF4_4F_piece_per_keV_per_year_detector",
            "CF4_F_fraction",
        ]

        include_f_split = "f_vector_per_year" in spectra and "f_axial_per_year" in spectra
        if include_f_split:
            header.extend([
                "dN_dEr_19F_vector_piece_per_keV_per_year_detector",
                "dN_dEr_19F_axial_piece_per_keV_per_year_detector",
                "F_axial_fraction",
            ])

        writer.writerow(header)

        for index, energy_kev in enumerate(energy_grid_kev):
            row = [
                energy_kev,
                spectra["prompt_per_s"][index],
                spectra["nue_per_s"][index],
                spectra["numubar_per_s"][index],
                spectra["delayed_per_s"][index],
                spectra["total_per_s"][index],
                thresholded_spectra["total_per_s"][index],
                spectra["prompt_per_year"][index],
                thresholded_spectra["prompt_per_year"][index],
                spectra["delayed_per_year"][index],
                thresholded_spectra["delayed_per_year"][index],
                spectra["total_per_year"][index],
                thresholded_spectra["total_per_year"][index],
                spectra["c_piece_per_year"][index],
                spectra["f_piece_per_year"][index],
                spectra["cf4_f_fraction"][index],
            ]

            if include_f_split:
                row.extend([
                    spectra["f_vector_per_year"][index],
                    spectra["f_axial_per_year"][index],
                    spectra.get("f_axial_fraction", np.zeros_like(energy_grid_kev))[index],
                ])

            writer.writerow(row)


def write_nuclear_detector_ee_csv(
    filename: str,
    cfg: DetectorConfig,
    spectra: Dict[str, np.ndarray],
) -> None:
    energy_grid_kevee = spectra["energy_kevee"]
    threshold_kevee = cfg.energy_threshold_kev

    total_per_s_thresholded = hard_threshold(energy_grid_kevee, spectra["total_per_s"], threshold_kevee)
    total_per_year_thresholded = hard_threshold(energy_grid_kevee, spectra["total_per_year"], threshold_kevee)
    prompt_per_year_thresholded = hard_threshold(energy_grid_kevee, spectra["prompt_per_year"], threshold_kevee)
    delayed_per_year_thresholded = hard_threshold(energy_grid_kevee, spectra["delayed_per_year"], threshold_kevee)

    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        header = [
            "Eee_keVee",
            "dR_dEee_CF4_prompt_per_s_per_keVee_detector",
            "dR_dEee_CF4_nue_per_s_per_keVee_detector",
            "dR_dEee_CF4_numubar_per_s_per_keVee_detector",
            "dR_dEee_CF4_delayed_per_s_per_keVee_detector",
            "dR_dEee_CF4_total_per_s_per_keVee_detector",
            "dR_dEee_CF4_total_above_threshold_per_s_per_keVee_detector",
            "dN_dEee_CF4_prompt_per_keVee_per_year_detector",
            "dN_dEee_CF4_prompt_above_threshold_per_keVee_per_year_detector",
            "dN_dEee_CF4_delayed_per_keVee_per_year_detector",
            "dN_dEee_CF4_delayed_above_threshold_per_keVee_per_year_detector",
            "dN_dEee_CF4_total_per_keVee_per_year_detector",
            "dN_dEee_CF4_total_above_threshold_per_keVee_per_year_detector",
            "dN_dEee_CF4_C_piece_per_keVee_per_year_detector",
            "dN_dEee_CF4_4F_piece_per_keVee_per_year_detector",
            "CF4_F_fraction",
        ]

        include_f_split = "f_vector_per_year" in spectra and "f_axial_per_year" in spectra
        if include_f_split:
            header.extend([
                "dN_dEee_19F_vector_piece_per_keVee_per_year_detector",
                "dN_dEee_19F_axial_piece_per_keVee_per_year_detector",
                "F_axial_fraction",
            ])

        writer.writerow(header)

        for index, energy_kevee in enumerate(energy_grid_kevee):
            row = [
                energy_kevee,
                spectra["prompt_per_s"][index],
                spectra["nue_per_s"][index],
                spectra["numubar_per_s"][index],
                spectra["delayed_per_s"][index],
                spectra["total_per_s"][index],
                total_per_s_thresholded[index],
                spectra["prompt_per_year"][index],
                prompt_per_year_thresholded[index],
                spectra["delayed_per_year"][index],
                delayed_per_year_thresholded[index],
                spectra["total_per_year"][index],
                total_per_year_thresholded[index],
                spectra["c_piece_per_year"][index],
                spectra["f_piece_per_year"][index],
                spectra["cf4_f_fraction"][index],
            ]

            if include_f_split:
                row.extend([
                    spectra["f_vector_per_year"][index],
                    spectra["f_axial_per_year"][index],
                    spectra.get("f_axial_fraction", np.zeros_like(energy_grid_kevee))[index],
                ])

            writer.writerow(row)


def write_electron_detector_csv(
    filename: str,
    cfg: DetectorConfig,
    spectra: Dict[str, np.ndarray],
) -> None:
    energy_grid_kev = spectra["energy_kev"]
    threshold_kev = cfg.energy_threshold_kev

    total_per_s_thresholded = hard_threshold(energy_grid_kev, spectra["total_per_s"], threshold_kev)
    total_per_year_thresholded = hard_threshold(energy_grid_kev, spectra["total_per_year"], threshold_kev)
    prompt_per_year_thresholded = hard_threshold(energy_grid_kev, spectra["prompt_per_year"], threshold_kev)
    delayed_per_year_thresholded = hard_threshold(energy_grid_kev, spectra["delayed_per_year"], threshold_kev)

    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "Te_keV",
            "dR_dTe_CF4_numu_prompt_per_s_per_keV_detector",
            "dR_dTe_CF4_nue_delayed_per_s_per_keV_detector",
            "dR_dTe_CF4_numubar_delayed_per_s_per_keV_detector",
            "dR_dTe_CF4_delayed_total_per_s_per_keV_detector",
            "dR_dTe_CF4_total_per_s_per_keV_detector",
            "dR_dTe_CF4_total_above_threshold_per_s_per_keV_detector",
            "dN_dTe_CF4_numu_prompt_per_keV_per_year_detector",
            "dN_dTe_CF4_numu_prompt_above_threshold_per_keV_per_year_detector",
            "dN_dTe_CF4_delayed_total_per_keV_per_year_detector",
            "dN_dTe_CF4_delayed_total_above_threshold_per_keV_per_year_detector",
            "dN_dTe_CF4_total_per_keV_per_year_detector",
            "dN_dTe_CF4_total_above_threshold_per_keV_per_year_detector",
            "CF4_electrons_per_molecule",
        ])

        for index, energy_kev in enumerate(energy_grid_kev):
            writer.writerow([
                energy_kev,
                spectra["prompt_per_s"][index],
                spectra["nue_per_s"][index],
                spectra["numubar_per_s"][index],
                spectra["delayed_per_s"][index],
                spectra["total_per_s"][index],
                total_per_s_thresholded[index],
                spectra["prompt_per_year"][index],
                prompt_per_year_thresholded[index],
                spectra["delayed_per_year"][index],
                delayed_per_year_thresholded[index],
                spectra["total_per_year"][index],
                total_per_year_thresholded[index],
                spectra["electrons_per_molecule"][index],
            ])


def plot_nuclear_detector_rates(
    filename: str,
    spectra: Dict[str, np.ndarray],
) -> None:
    energy_grid_kev = spectra["energy_kev"]

    plt.figure(figsize=(8, 5))
    plt.plot(energy_grid_kev, spectra["prompt_per_year"], label=r"prompt $\nu_\mu$")
    plt.plot(energy_grid_kev, spectra["delayed_per_year"], label="delayed total")
    plt.plot(energy_grid_kev, spectra["total_per_year"], linewidth=2, label="total")

    plt.xlabel("Nuclear recoil energy [keV]")
    plt.ylabel("Expected events / (keV year)")
    plt.title("CF4 detector CEvNS recoil spectrum (raw recoil-energy view)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    positive = np.concatenate((
        spectra["prompt_per_year"],
        spectra["delayed_per_year"],
        spectra["total_per_year"],
    ))
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_nuclear_detector_ee_rates(
    filename: str,
    cfg: DetectorConfig,
    spectra: Dict[str, np.ndarray],
) -> None:
    energy_grid_kevee = spectra["energy_kevee"]
    threshold_kevee = cfg.energy_threshold_kev

    plt.figure(figsize=(8, 5))
    plt.plot(energy_grid_kevee, spectra["prompt_per_year"], label=r"prompt $\nu_\mu$")
    plt.plot(energy_grid_kevee, spectra["delayed_per_year"], label="delayed total")
    plt.plot(energy_grid_kevee, spectra["total_per_year"], linewidth=2, label="total")
    plt.axvline(
        threshold_kevee,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"threshold = {threshold_kevee:.2f} keVee",
    )

    plt.xlabel("Electron-equivalent energy [keVee]")
    plt.ylabel("Expected events / (keVee year)")
    plt.title("CF4 detector CEvNS spectrum in measured energy")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    positive = np.concatenate((
        spectra["prompt_per_year"],
        spectra["delayed_per_year"],
        spectra["total_per_year"],
    ))
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_electron_detector_rates(
    filename: str,
    cfg: DetectorConfig,
    spectra: Dict[str, np.ndarray],
) -> None:
    energy_grid_kev = spectra["energy_kev"]
    threshold_kev = cfg.energy_threshold_kev

    plt.figure(figsize=(8, 5))
    plt.plot(energy_grid_kev, spectra["prompt_per_year"], label=r"prompt $\nu_\mu e$")
    plt.plot(energy_grid_kev, spectra["delayed_per_year"], label="delayed total")
    plt.plot(energy_grid_kev, spectra["total_per_year"], linewidth=2, label="total")
    plt.axvline(
        threshold_kev,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"threshold = {threshold_kev:.2f} keVee",
    )

    plt.xlabel("Electron recoil energy [keV]")
    plt.ylabel("Expected events / (keV year)")
    plt.title(r"CF4 detector $\nu$-e recoil spectrum")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    positive = np.concatenate((
        spectra["prompt_per_year"],
        spectra["delayed_per_year"],
        spectra["total_per_year"],
    ))
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def build_nuclear_summary(
    spectra: Dict[str, np.ndarray],
    threshold_kev: float,
) -> dict:
    energy_grid_kev = spectra["energy_kevee"] if "energy_kevee" in spectra else spectra["energy_kev"]

    prompt_rate_above = integrate_above_threshold(energy_grid_kev, spectra["prompt_per_s"], threshold_kev)
    delayed_rate_above = integrate_above_threshold(energy_grid_kev, spectra["delayed_per_s"], threshold_kev)
    total_rate_above = integrate_above_threshold(energy_grid_kev, spectra["total_per_s"], threshold_kev)
    c_piece_above = integrate_above_threshold(energy_grid_kev, spectra["c_piece_per_s"], threshold_kev)
    f_piece_above = integrate_above_threshold(energy_grid_kev, spectra["f_piece_per_s"], threshold_kev)

    out = {
        "prompt_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["prompt_per_s"]),
        "delayed_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["delayed_per_s"]),
        "total_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["total_per_s"]),
        "prompt_rate_above_threshold_per_s": prompt_rate_above,
        "delayed_rate_above_threshold_per_s": delayed_rate_above,
        "total_rate_above_threshold_per_s": total_rate_above,
        "prompt_events_per_year_above_threshold": prompt_rate_above * SECONDS_PER_YEAR,
        "delayed_events_per_year_above_threshold": delayed_rate_above * SECONDS_PER_YEAR,
        "total_events_per_year_above_threshold": total_rate_above * SECONDS_PER_YEAR,
        "c_piece_events_per_year_above_threshold": c_piece_above * SECONDS_PER_YEAR,
        "f_piece_events_per_year_above_threshold": f_piece_above * SECONDS_PER_YEAR,
        "prompt_fraction_above_threshold": prompt_rate_above / total_rate_above if total_rate_above > 0.0 else 0.0,
        "delayed_fraction_above_threshold": delayed_rate_above / total_rate_above if total_rate_above > 0.0 else 0.0,
        "cf4_f_fraction_above_threshold": f_piece_above / total_rate_above if total_rate_above > 0.0 else 0.0,
    }

    if "f_vector_per_s" in spectra and "f_axial_per_s" in spectra:
        f_vector_above = integrate_above_threshold(energy_grid_kev, spectra["f_vector_per_s"], threshold_kev)
        f_axial_above = integrate_above_threshold(energy_grid_kev, spectra["f_axial_per_s"], threshold_kev)
        f_total_above = f_vector_above + f_axial_above
        out["f_vector_events_per_year_above_threshold"] = f_vector_above * SECONDS_PER_YEAR
        out["f_axial_events_per_year_above_threshold"] = f_axial_above * SECONDS_PER_YEAR
        out["f_axial_fraction_above_threshold"] = f_axial_above / f_total_above if f_total_above > 0.0 else 0.0

    return out


def build_electron_summary(
    spectra: Dict[str, np.ndarray],
    threshold_kev: float,
) -> dict:
    energy_grid_kev = spectra["energy_kev"]

    prompt_rate_above = integrate_above_threshold(energy_grid_kev, spectra["prompt_per_s"], threshold_kev)
    nue_rate_above = integrate_above_threshold(energy_grid_kev, spectra["nue_per_s"], threshold_kev)
    numubar_rate_above = integrate_above_threshold(energy_grid_kev, spectra["numubar_per_s"], threshold_kev)
    delayed_rate_above = integrate_above_threshold(energy_grid_kev, spectra["delayed_per_s"], threshold_kev)
    total_rate_above = integrate_above_threshold(energy_grid_kev, spectra["total_per_s"], threshold_kev)

    return {
        "prompt_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["prompt_per_s"]),
        "nue_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["nue_per_s"]),
        "numubar_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["numubar_per_s"]),
        "delayed_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["delayed_per_s"]),
        "total_rate_full_per_s": integrate_spectrum(energy_grid_kev, spectra["total_per_s"]),
        "prompt_rate_above_threshold_per_s": prompt_rate_above,
        "nue_rate_above_threshold_per_s": nue_rate_above,
        "numubar_rate_above_threshold_per_s": numubar_rate_above,
        "delayed_rate_above_threshold_per_s": delayed_rate_above,
        "total_rate_above_threshold_per_s": total_rate_above,
        "prompt_events_per_year_above_threshold": prompt_rate_above * SECONDS_PER_YEAR,
        "nue_events_per_year_above_threshold": nue_rate_above * SECONDS_PER_YEAR,
        "numubar_events_per_year_above_threshold": numubar_rate_above * SECONDS_PER_YEAR,
        "delayed_events_per_year_above_threshold": delayed_rate_above * SECONDS_PER_YEAR,
        "total_events_per_year_above_threshold": total_rate_above * SECONDS_PER_YEAR,
        "prompt_fraction_above_threshold": prompt_rate_above / total_rate_above if total_rate_above > 0.0 else 0.0,
        "delayed_fraction_above_threshold": delayed_rate_above / total_rate_above if total_rate_above > 0.0 else 0.0,
    }


def build_summary(
    cfg: DetectorConfig,
    total_volume_m3: float,
    fiducial_volume_m3: float,
    moles_fiducial: float,
    molecules_fiducial: float,
    mass_fiducial_kg: float,
    nuclear_ee_spectra: Dict[str, np.ndarray],
    electron_spectra: Dict[str, np.ndarray] | None,
    carbon_qf: QuenchingCurve,
    fluorine_qf: QuenchingCurve,
) -> dict:
    nuclear_summary = build_nuclear_summary(nuclear_ee_spectra, cfg.energy_threshold_kev)
    electron_summary = (
        build_electron_summary(electron_spectra, cfg.energy_threshold_kev)
        if electron_spectra is not None
        else None
    )

    electrons_per_molecule = (
        float(electron_spectra["electrons_per_molecule"][0])
        if electron_spectra is not None
        else CF4_ELECTRONS_PER_MOLECULE
    )

    combined_total_events_per_year_above_threshold = nuclear_summary["total_events_per_year_above_threshold"]
    if electron_summary is not None:
        combined_total_events_per_year_above_threshold += electron_summary["total_events_per_year_above_threshold"]

    summary = {
        "config": {
            "input_nuclear_rate_csv": cfg.input_nuclear_rate_csv,
            "input_electron_rate_csv": cfg.input_electron_rate_csv,
            "output_dir": cfg.output_dir,
            "radius_m": cfg.radius_m,
            "length_m": cfg.length_m,
            "fiducial_fraction": cfg.fiducial_fraction,
            "pressure_pa": cfg.pressure_pa,
            "pressure_label": cfg.pressure_label,
            "temperature_k": cfg.temperature_k,
            "energy_threshold_kev": cfg.energy_threshold_kev,
            "energy_threshold_kevee": cfg.energy_threshold_kev,
            "threshold_space": "electron_equivalent_energy",
        },
        "derived_detector": {
            "total_volume_m3": total_volume_m3,
            "fiducial_volume_m3": fiducial_volume_m3,
            "cf4_density_kg_per_m3": mass_fiducial_kg / fiducial_volume_m3 if fiducial_volume_m3 > 0.0 else 0.0,
            "cf4_mass_fiducial_kg": mass_fiducial_kg,
            "cf4_mass_fiducial_g": mass_fiducial_kg * 1.0e3,
            "cf4_moles_fiducial": moles_fiducial,
            "cf4_molecules_fiducial": molecules_fiducial,
            "cf4_electrons_per_molecule": electrons_per_molecule,
            "electron_targets_fiducial": molecules_fiducial * electrons_per_molecule,
        },
        "quenching_factor": {
            "applied_to_nuclear_recoils": True,
            "carbon_qf_csv": os.path.relpath(DEFAULT_CARBON_QF_CSV, REPO_DIR),
            "fluorine_qf_csv": os.path.relpath(DEFAULT_FLUORINE_QF_CSV, REPO_DIR),
            "carbon_qf_energy_range_kev": [
                float(carbon_qf.recoil_energy_kev[0]),
                float(carbon_qf.recoil_energy_kev[-1]),
            ],
            "fluorine_qf_energy_range_kev": [
                float(fluorine_qf.recoil_energy_kev[0]),
                float(fluorine_qf.recoil_energy_kev[-1]),
            ],
        },
        "nuclear_integrated_rates": nuclear_summary,
        "electron_integrated_rates": electron_summary,
        "combined_integrated_rates": {
            "total_events_per_year_above_threshold": combined_total_events_per_year_above_threshold,
            "electron_over_nuclear_above_threshold": (
                electron_summary["total_events_per_year_above_threshold"]
                / nuclear_summary["total_events_per_year_above_threshold"]
                if electron_summary is not None and nuclear_summary["total_events_per_year_above_threshold"] > 0.0
                else 0.0
            ),
        },
    }

    return summary


def write_summary_json(filename: str, summary: dict) -> None:
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def print_summary(summary: dict) -> None:
    cfg = summary["config"]
    det = summary["derived_detector"]
    qf = summary.get("quenching_factor") or {}
    nuclear = summary["nuclear_integrated_rates"]
    electron = summary["electron_integrated_rates"]
    combined = summary["combined_integrated_rates"]

    print("=== Detector configuration ===")
    print(f"Nuclear rate CSV                : {cfg['input_nuclear_rate_csv']}")
    print(f"Electron rate CSV               : {cfg['input_electron_rate_csv']}")
    print(f"Cylinder radius                 : {cfg['radius_m']:.6f} m")
    print(f"Cylinder length                 : {cfg['length_m']:.6f} m")
    print(f"Fiducial fraction               : {cfg['fiducial_fraction']:.6f}")
    print(f"CF4 pressure                    : {cfg['pressure_label']}")
    print(f"Gas temperature                 : {cfg['temperature_k']:.3f} K")
    print(f"Measured energy threshold       : {cfg['energy_threshold_kevee']:.3f} keVee")
    if qf:
        print(f"Carbon QF table                 : {qf['carbon_qf_csv']}")
        print(f"Fluorine QF table               : {qf['fluorine_qf_csv']}")
    print()

    print("=== Derived CF4 bookkeeping ===")
    print(f"Total cylinder volume           : {det['total_volume_m3']:.6e} m^3")
    print(f"Fiducial volume                 : {det['fiducial_volume_m3']:.6e} m^3")
    print(f"CF4 density                     : {det['cf4_density_kg_per_m3']:.6e} kg/m^3")
    print(f"CF4 fiducial mass               : {det['cf4_mass_fiducial_g']:.6e} g")
    print(f"CF4 fiducial moles              : {det['cf4_moles_fiducial']:.6e} mol")
    print(f"CF4 fiducial molecules          : {det['cf4_molecules_fiducial']:.6e}")
    print(f"Electrons per CF4 molecule      : {det['cf4_electrons_per_molecule']:.0f}")
    print(f"Fiducial electron targets       : {det['electron_targets_fiducial']:.6e}")
    print()

    print("=== Integrated CEvNS rates (threshold in keVee) ===")
    print(f"Prompt rate above threshold     : {nuclear['prompt_rate_above_threshold_per_s']:.6e} s^-1")
    print(f"Delayed rate above threshold    : {nuclear['delayed_rate_above_threshold_per_s']:.6e} s^-1")
    print(f"Total rate above threshold      : {nuclear['total_rate_above_threshold_per_s']:.6e} s^-1")
    print(f"Prompt events/year above thr    : {nuclear['prompt_events_per_year_above_threshold']:.6e}")
    print(f"Delayed events/year above thr   : {nuclear['delayed_events_per_year_above_threshold']:.6e}")
    print(f"Total events/year above thr     : {nuclear['total_events_per_year_above_threshold']:.6e}")
    print(f"4F contribution above thr       : {nuclear['cf4_f_fraction_above_threshold']:.6f}")
    print(f"Prompt fraction above thr       : {nuclear['prompt_fraction_above_threshold']:.6f}")
    print(f"Delayed fraction above thr      : {nuclear['delayed_fraction_above_threshold']:.6f}")
    if "f_axial_fraction_above_threshold" in nuclear:
        print(f"19F axial fraction above thr    : {nuclear['f_axial_fraction_above_threshold']:.6f}")
    print()

    if electron is not None:
        print("=== Integrated nu-e rates ===")
        print(f"Prompt rate above threshold     : {electron['prompt_rate_above_threshold_per_s']:.6e} s^-1")
        print(f"Delayed rate above threshold    : {electron['delayed_rate_above_threshold_per_s']:.6e} s^-1")
        print(f"Total rate above threshold      : {electron['total_rate_above_threshold_per_s']:.6e} s^-1")
        print(f"Prompt events/year above thr    : {electron['prompt_events_per_year_above_threshold']:.6e}")
        print(f"Delayed events/year above thr   : {electron['delayed_events_per_year_above_threshold']:.6e}")
        print(f"Total events/year above thr     : {electron['total_events_per_year_above_threshold']:.6e}")
        print(f"Prompt fraction above thr       : {electron['prompt_fraction_above_threshold']:.6f}")
        print(f"Delayed fraction above thr      : {electron['delayed_fraction_above_threshold']:.6f}")
        print()

    print("=== Combined totals (threshold in keVee) ===")
    print(f"All channels events/year > thr  : {combined['total_events_per_year_above_threshold']:.6e}")
    print(f"nu-e / CEvNS above threshold    : {combined['electron_over_nuclear_above_threshold']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scale the flux-folded CF4 differential rates into detector-level spectra "
            "using a cylindrical ideal-gas CF4 configuration."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the detector configuration JSON file.",
    )
    parser.add_argument(
        "--input-csv",
        "--input-nuclear-csv",
        dest="input_nuclear_csv",
        default=None,
        help="Optional override for the nuclear rate CSV.",
    )
    parser.add_argument(
        "--input-electron-csv",
        default=None,
        help="Optional override for the electron-scattering rate CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the detector output directory.",
    )
    args = parser.parse_args()

    cfg = load_config(
        config_path=args.config,
        nuclear_rate_csv_override=args.input_nuclear_csv,
        electron_rate_csv_override=args.input_electron_csv,
        output_dir_override=args.output_dir,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    nuclear_table = load_rate_table(
        cfg.input_nuclear_rate_csv,
        REQUIRED_NUCLEAR_COLUMNS,
        OPTIONAL_NUCLEAR_COLUMNS,
    )
    carbon_qf = load_quenching_curve(DEFAULT_CARBON_QF_CSV, "carbon")
    fluorine_qf = load_quenching_curve(DEFAULT_FLUORINE_QF_CSV, "fluorine")

    electron_table = None
    if cfg.input_electron_rate_csv:
        electron_table = load_rate_table(
            cfg.input_electron_rate_csv,
            REQUIRED_ELECTRON_COLUMNS,
        )

    total_volume_m3 = cylinder_volume_m3(cfg.radius_m, cfg.length_m)
    fiducial_volume_m3 = cfg.fiducial_fraction * total_volume_m3
    moles_fiducial = ideal_gas_moles(cfg.pressure_pa, fiducial_volume_m3, cfg.temperature_k)
    molecules_fiducial = moles_fiducial * AVOGADRO
    mass_fiducial_kg = moles_fiducial * CF4_MOLAR_MASS_KG_PER_MOL

    carbon_nuclear_spectra, fluorine_nuclear_spectra = build_nuclear_species_detector_spectra(
        nuclear_table,
        molecules_fiducial,
    )
    nuclear_spectra = combine_nuclear_species_detector_spectra(
        carbon_nuclear_spectra,
        fluorine_nuclear_spectra,
    )
    thresholded_recoil_spectra = combine_nuclear_species_detector_spectra(
        build_thresholded_recoil_species_spectra(
            carbon_nuclear_spectra,
            carbon_qf,
            cfg.energy_threshold_kev,
        ),
        build_thresholded_recoil_species_spectra(
            fluorine_nuclear_spectra,
            fluorine_qf,
            cfg.energy_threshold_kev,
        ),
    )
    nuclear_ee_spectra = build_nuclear_ee_detector_spectra(
        carbon_nuclear_spectra,
        fluorine_nuclear_spectra,
        carbon_qf,
        fluorine_qf,
    )
    validate_nuclear_ee_spectra(nuclear_spectra, nuclear_ee_spectra)
    electron_spectra = (
        build_electron_detector_spectra(electron_table, molecules_fiducial)
        if electron_table is not None
        else None
    )

    nuclear_csv_file = os.path.join(cfg.output_dir, "cf4_detector_differential_rate.csv")
    nuclear_plot_file = os.path.join(cfg.output_dir, "cf4_detector_differential_rate.png")
    nuclear_ee_csv_file = os.path.join(cfg.output_dir, "cf4_detector_differential_rate_ee.csv")
    nuclear_ee_plot_file = os.path.join(cfg.output_dir, "cf4_detector_differential_rate_ee.png")
    electron_csv_file = os.path.join(cfg.output_dir, "cf4_detector_electron_differential_rate.csv")
    electron_plot_file = os.path.join(cfg.output_dir, "cf4_detector_electron_differential_rate.png")
    summary_file = os.path.join(cfg.output_dir, "cf4_detector_summary.json")

    write_nuclear_detector_csv(nuclear_csv_file, nuclear_spectra, thresholded_recoil_spectra)
    plot_nuclear_detector_rates(nuclear_plot_file, nuclear_spectra)
    write_nuclear_detector_ee_csv(nuclear_ee_csv_file, cfg, nuclear_ee_spectra)
    plot_nuclear_detector_ee_rates(nuclear_ee_plot_file, cfg, nuclear_ee_spectra)

    if electron_spectra is not None:
        write_electron_detector_csv(electron_csv_file, cfg, electron_spectra)
        plot_electron_detector_rates(electron_plot_file, cfg, electron_spectra)

    summary = build_summary(
        cfg=cfg,
        total_volume_m3=total_volume_m3,
        fiducial_volume_m3=fiducial_volume_m3,
        moles_fiducial=moles_fiducial,
        molecules_fiducial=molecules_fiducial,
        mass_fiducial_kg=mass_fiducial_kg,
        nuclear_ee_spectra=nuclear_ee_spectra,
        electron_spectra=electron_spectra,
        carbon_qf=carbon_qf,
        fluorine_qf=fluorine_qf,
    )
    write_summary_json(summary_file, summary)
    print_summary(summary)

    print()
    print("Saved files:")
    print(f"  {nuclear_csv_file}")
    print(f"  {nuclear_plot_file}")
    print(f"  {nuclear_ee_csv_file}")
    print(f"  {nuclear_ee_plot_file}")
    if electron_spectra is not None:
        print(f"  {electron_csv_file}")
        print(f"  {electron_plot_file}")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()
