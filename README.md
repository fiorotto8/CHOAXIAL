# COHAXIAL

COHAXIAL is a practical research sandbox for first-pass neutrino-scattering
rate estimates in light gas targets.  The current workflow is centered on
CF4 and supports:

- Standard-Model coherent elastic neutrino-nucleus scattering (CEvNS) on
  `12C` and `19F`;
- Standard-Model neutrino-electron elastic scattering on the 42 electrons in a
  CF4 molecule;
- stopped-pion decay-at-rest (DAR) source benchmarks for ESS, SNS FTS, and
  J-PARC MLF-like configurations;
- flux-folded recoil spectra per CF4 molecule;
- ideal-gas detector scaling, quenching-factor remapping to keVee, and simple
  threshold scans.

This repository is not a polished public package or a final detector design
calculation.  It is a daily-use, work-in-progress research repository intended
to keep the source assumptions, target composition, cross-section ingredients,
and detector normalization inspectable while exploring whether a source/target
concept is in the right rate regime.

## Quick Start

From WSL, use the repository root:

```bash
cd /mnt/c/Users/david/MyDrive/WORK/COHAXIAL
python3 -m pip install -r requirements.txt
```

Run the lightweight sanity checks:

```bash
python3 scripts/sanity_check.py
```

Run a single CEvNS point:

```bash
python3 cevens.py --target 19F --enu-mev 30 --er-kev 5 --json
```

Generate a source benchmark:

```bash
python3 ESS_flux.py
python3 SNS_flux.py
python3 JPARK_flux.py
```

Fold a source flux with the CF4 interaction model:

```bash
python3 rate_estimation.py --source ess
```

Useful alternatives:

```bash
python3 rate_estimation.py --source sns --fluorine-axial-model hoferichter_19f_fast
python3 rate_estimation.py --source jparc --distance-m 24 --output-dir cevens_rate_output_jparc
```

`rate_estimation.py` currently defaults to `hoferichter_19f_central`; use
`hoferichter_19f_fast` for the leading transverse-response model without the
central HMS correction.

Scale molecule-normalized rates to the example detector:

```bash
python3 detector_estimation.py --config configs/detector_config.json
```

Scan measured-energy thresholds in keVee:

```bash
python3 scan_detector_threshold.py
```

Expected outputs are CSV and PNG products in ignored generated-output folders
such as `ess_flux_output/`, `sns_flux_output/`, `jparc_mlf_flux_output/`,
`cevens_rate_output/`, `detector_rate_output/`, and
`detector_threshold_scan_output/`.

## Repository Layout

```text
.
├── cevens.py                     # CEvNS and nu-e interaction kernels
├── ESS_flux.py                   # ESS DAR source entry point
├── SNS_flux.py                   # SNS FTS DAR source entry point
├── JPARK_flux.py                 # J-PARC MLF DAR source entry point
├── rate_estimation.py            # flux-folded CF4 spectra per molecule
├── detector_estimation.py        # detector scaling and quenching remap
├── scan_detector_threshold.py    # repeated detector threshold scan
├── cohaxial/
│   └── dar_flux.py               # shared stopped-pion DAR flux helpers
├── configs/
│   └── detector_config.json      # example cylindrical CF4 detector config
├── data/
│   └── quenching/
│       ├── C_in_CF4.csv          # carbon QF table in CF4
│       └── F_in_CF4.csv          # fluorine QF table in CF4
├── examples/
│   └── quick_point_check.py      # tiny point-calculation example
├── scripts/
│   └── sanity_check.py           # no-pytest validation smoke test
├── GMO_exam/
│   └── simple_estimates.ipynb    # exploratory notebook, outside main chain
├── Paper/                        # optional local references, ignored by git
└── requirements.txt
```

The root Python files are intentionally kept as user-facing scripts because
that is the established working style of this repository.  The only package-like
folder is `cohaxial/`, which currently holds shared code that would otherwise be
duplicated across the three source scripts.

Generated products are ignored by git.  The quenching-factor CSVs under
`data/quenching/` are input data and are intentionally unignored.

## Workflow Overview

The main analysis chain is:

```text
DAR source metadata
  -> phi(E_nu)
  -> fold with d sigma / dE
  -> rate per CF4 molecule
  -> ideal-gas detector scaling
  -> quenching remap and threshold integration
```

1. `ESS_flux.py`, `SNS_flux.py`, or `JPARK_flux.py` defines a beam metadata
   object and exposes a common stopped-pion DAR flux API.  The duplicated flux
   logic lives in `cohaxial/dar_flux.py`.
2. `cevens.py` defines the nuclear targets, form factors, CEvNS calculator, and
   neutrino-electron calculator.
3. `rate_estimation.py` chooses one DAR source, builds `12C`, `19F`, and CF4
   electron targets, then integrates flux times cross section over neutrino
   energy.  Outputs are rates per molecule, not detector counts.
4. `detector_estimation.py` reads the per-molecule CSVs, parses a detector JSON,
   computes the fiducial CF4 inventory with the ideal gas law, applies C/F
   quenching-factor maps to the nuclear spectra, and writes detector-level
   spectra plus a compact summary JSON.
5. `scan_detector_threshold.py` reruns `detector_estimation.py` over a grid of
   measured-energy thresholds and summarizes prompt/delayed, carbon/fluorine,
   and fluorine vector/axial contributions.

## Physics And Mathematical Model

### Units

The code uses practical phenomenology units:

- neutrino energy `E_nu`: MeV;
- nuclear recoil energy `E_r`: keV;
- electron recoil energy `T_e`: keV;
- measured nuclear energy `E_ee`: keVee;
- nuclear and electron masses: GeV internally;
- CEvNS and neutrino-electron differential cross sections:
  cm^2 / keV;
- delayed flux: neutrinos / (cm^2 s MeV);
- prompt line flux: neutrinos / (cm^2 s);
- delayed per-POT fluence: neutrinos / (cm^2 POT MeV);
- molecule-normalized rates: s^-1 keV^-1 molecule^-1;
- detector spectra: events / (keV year) or events / (keVee year).

### Source Of Physics Inputs

The local `Paper/` folder was used to trace the current mathematical
implementation.  The active formulas are drawn mainly from:

- `Paper/AccoppiamentoAssiale_Paper_2603.05281v1 (1).pdf`, a 2026 axial-vector
  CEvNS sensitivity study that motivates fluorine-based targets and uses an
  ESS-like DAR normalization;
- `Paper/CrossSections/HMS_axialFitparams.pdf`, Hoferichter, Menendez, and
  Schwenk, PRD 102 (2020), for the CEvNS vector/axial split, `19F`
  shell-model axial response coefficients, hadronic constants, and two-body
  current correction conventions;
- `Paper/CrossSections/HELM_vectorModel.pdf`, Helm, Phys. Rev. 104 (1956), for
  the folded-density idea behind the Helm form factor.

The local `Klein-Nystrand_vectorModel.pdf` and `AXialAprox_*.pdf` files are
useful background references, but their formulas are not the active default
implementation.  Klein-Nystrand is an alternative vector form-factor reference
mentioned by the axial-vector paper.  The `AXialAprox_*` PDFs describe older
dark-matter spin-dependent form-factor conventions, not the CEvNS `19F`
response used by the default workflow.

Important numerical constants and where they enter:

| Quantity | Code value | Where used | Provenance / reason |
| --- | ---: | --- | --- |
| `G_F` | `1.1663787e-5 GeV^-2` | all weak cross sections | Standard PDG value used as a fixed SM input |
| `sin^2(theta_W)` | `0.23857` | weak charge and nu-e couplings | Low-energy weak-mixing input; the axial paper also states that PDG electroweak values are used |
| `amu` | `0.93149410242 GeV` | nuclear masses | CODATA-style conversion; masses are approximate `A * amu` in this sandbox |
| `m_e` | `0.510998950 MeV` | nu-e scattering | PDG electron mass |
| `m_pi+` | `139.57039 MeV` | DAR prompt line | PDG charged-pion mass |
| `m_mu+` | `105.6583755 MeV` | DAR endpoint and Michel spectra | PDG muon mass |
| `g_A` | `1.27641` | HMS `19F` axial response | HMS Table I |
| `g_A^{s,N}` | `0` fast, `-0.085` central | strange axial contribution | Fast model omits strangeness by design; central value from HMS Table I |
| `<r_A^2>` | `0.46 fm^2` | central axial `delta_0` | HMS Table I |
| `rho` | `0.10 fm^-3` | central two-body correction | midpoint of the HMS Table V range `0.09...0.11 fm^-3` |
| `c_1,c_3,c_4,c_6` | `-1.20,-4.45,2.96,5.01 GeV^-1` | central two-body correction | HMS Table V central values |
| `c_D` | `(-6.08 + 0.30)/2 = -2.89` | central two-body correction | midpoint of the HMS Table V allowed range; this is a pragmatic single-value choice, not an uncertainty scan |
| `F_pi` | `0.09228 GeV` | central two-body correction | HMS convention |
| `g_piNN` | `sqrt(4 pi 13.7)` | central two-body correction | HMS/Klos convention used by the implementation comments |

### CEvNS

`cevens.py` implements a vector plus pure-axial CEvNS split:

```math
\frac{d\sigma}{dE_r}
= \frac{G_F^2 m_N}{4\pi}
\left[
\left(1-\frac{m_N E_r}{2E_\nu^2}-\frac{E_r}{E_\nu}\right)
Q_W^2 |F_W(q^2)|^2
+
\left(1+\frac{m_N E_r}{2E_\nu^2}-\frac{E_r}{E_\nu}\right)
F_A(q^2)
\right],
```

with

```math
q^2 = 2m_N E_r,
\qquad
Q_W = Z(1 - 4\sin^2\theta_W) - N.
```

Implementation notes:

- `12C` is treated as spin zero, so `F_A = 0`.
- `19F` is the only built-in target with nonzero axial options.
- The calculator keeps total, vector-only, and axial-only components separate.
- Cross sections are clipped at zero after kinematic checks to avoid tiny
  negative numerical artifacts at endpoints.
- Vector-axial interference terms are not included.  The axial-vector paper
  notes that these terms vanish or are recoil-suppressed for the intended
  CEvNS use case, so the sandbox follows the independent vector plus pure-axial
  decomposition.
- Nuclear masses are simple `A * amu` approximations.  This is adequate for
  first-pass rate comparisons but should be replaced by isotope masses for a
  precision calculation.

### Vector Form Factor

The default weak vector form factor is the Helm form:

```math
F_W(q) =
3\frac{j_1(qR_n)}{qR_n}
\exp\left[-\frac{(qs)^2}{2}\right],
```

where

```math
R_n^2 = c^2 + \frac{7}{3}\pi^2 a^2 - 5s^2,
```

and the implemented defaults are:

```math
c = 1.23 A^{1/3} - 0.60\ \mathrm{fm},\quad
a = 0.52\ \mathrm{fm},\quad
s = 0.90\ \mathrm{fm}.
```

`--pointlike` in `cevens.py` uses `F_W = 1` instead.

The Helm choice is used because it is the standard compact phenomenological
weak-charge form factor in the local axial CEvNS paper and in much CEvNS
rate-estimate work.  The `Paper/` folder also contains a Klein-Nystrand vector
form-factor reference, but no Klein-Nystrand implementation currently exists in
the code.

### 19F Axial Response

The generic axial form-factor decomposition is:

```math
F_A(q^2) =
\frac{8\pi}{2J+1}
\left[
(g_A^s)^2 S_{00}(q^2)
- g_A g_A^s S_{01}(q^2)
+ g_A^2 S_{11}(q^2)
\right].
```

For `19F`, the built-in Hoferichter-Menendez-Schwenk options use the transverse
shell-model polynomial responses:

```math
F(u) = e^{-u/2}\sum_i c_i u^i,
\qquad
u = \frac{q^2 b^2}{2},
\qquad
b = 1.7623\ \mathrm{fm}.
```

Here `q` in `u` is evaluated in `fm^-1`; the code converts from `q` in GeV
using `1 fm = 5.067730716 GeV^-1`.

The CEvNS axial term uses the transverse spin response `S^T_ij`.  In the HMS
notation this corresponds to the `Sigma-prime` response coefficients.  The
`Sigma-double-prime` longitudinal coefficients are present in HMS Table VIII and
are kept visible in `cevens.py` for future extensions, but they are not summed
into the CEvNS `F_A` used by the current workflow.

For `19F`, HMS Table VIII gives only the `L=1` multipole for the active
CEvNS transverse response.  The code evaluates:

```math
F_{\Sigma',+} = F_{\Sigma'}^p + F_{\Sigma'}^n,
\qquad
F_{\Sigma',-} = F_{\Sigma'}^p - F_{\Sigma'}^n,
```

and then:

```math
S_{00}^T = F_{\Sigma',+}^2,
\qquad
S_{11}^T = \left[(1+\delta_0)F_{\Sigma',-}\right]^2,
\qquad
S_{01}^T = 2(1+\delta_0)F_{\Sigma',+}F_{\Sigma',-}.
```

The active `19F` coefficients are:

| Response | `c_0` | `c_1` | `c_2` |
| --- | ---: | ---: | ---: |
| `F_{Sigma'}^p` | `0.269513` | `-0.18098` | `0.0296873` |
| `F_{Sigma'}^n` | `-0.00113172` | `0.00038188` | `0.000744991` |

At `q^2 = 0`, the fast model gives `F_A(0) ~= 1.50`, consistent with the HMS
Table IV spin expectation values for `19F`, `<S_p> = 0.478` and
`<S_n> = -0.002`, through the HMS zero-momentum normalization:

```math
F_A(0) =
\frac{4}{3}g_A^2\frac{J+1}{J}
\left(\langle S_p\rangle-\langle S_n\rangle\right)^2
```

for `J = 1/2` when strangeness and two-body corrections are omitted.

Available options:

- `hoferichter_19f_fast`: Table VIII transverse response polynomials,
  `g_A = 1.27641`, `g_A^{s,N} = 0`, and
  `delta_0 = 0`.
- `hoferichter_19f_central`: same response basis, `g_A^{s,N} = -0.085`,
  and one central HMS Eq. (86) `delta_0(q^2)` prescription using the constants
  present in `cevens.py`.
- `none`: disables the axial term.
- `toy`: legacy/debug spin-expectation model with a simple falloff.  It is not
  a proposal-grade nuclear-response model.

The full uncertainty band from nuclear inputs is not implemented.

Decision logic for the two HMS options:

- The fast option matches the leading-response spirit of the local axial-vector
  CEvNS study: shell-model `19F` transverse coefficients, no subleading
  two-body-current correction, and no strange axial term.  It is useful for
  rapid scans and comparisons with the first-pass analytic estimates.
- The central option is a single exploratory HMS central-value correction.  It
  applies the axial-radius and two-body-current `delta_0` correction with HMS
  Table I/V central inputs.  The HMS uncertainty band depends on ranges in
  `rho`, `c_D`, and other nuclear-current assumptions; that band is not yet
  propagated.
- The toy option keeps the older `<S_p>, <S_n>` interface for debugging.  Its
  defaults, `<S_p>=0.475`, `<S_n>=0`, and `Lambda_A=0.35 GeV`, should not be
  used as a reference CEvNS model without separate validation.  They are kept
  only because they are convenient for checking vector/axial plumbing.

### Neutrino-Electron Scattering

The free-electron elastic-scattering kernel is:

```math
\frac{d\sigma}{dT_e}
=
\frac{2G_F^2 m_e}{\pi}
\left[
g_\mathrm{lead}^2
+ g_\mathrm{sub}^2(1-y)^2
- \frac{g_L g_R m_e T_e}{E_\nu^2}
\right],
\qquad
y = \frac{T_e}{E_\nu}.
```

For neutrinos, `g_lead = g_L` and `g_sub = g_R`; for antineutrinos those
placements are interchanged.  The neutral-current couplings are:

```math
g_L = -\frac{1}{2} + \sin^2\theta_W,
\qquad
g_R = \sin^2\theta_W.
```

For electron flavor, the charged-current contribution is included through:

```math
g_L \rightarrow g_L + 1.
```

For CF4, the molecule-level electron channel multiplies the per-electron rate
by:

```math
Z_\mathrm{tot} = 6 + 4\times 9 = 42.
```

### Stopped-Pion DAR Source

The source scripts model the standard stopped-pion chain:

```math
\pi^+ \rightarrow \mu^+ + \nu_\mu,
\qquad
\mu^+ \rightarrow e^+ + \nu_e + \bar{\nu}_\mu.
```

The prompt `nu_mu` is monochromatic:

```math
E_{\nu_\mu}^{\mathrm{prompt}}
=
\frac{m_\pi^2 - m_\mu^2}{2m_\pi}
\simeq 29.79\ \mathrm{MeV}.
```

The delayed Michel spectra extend to:

```math
E_{\max} = \frac{m_\mu}{2} \simeq 52.83\ \mathrm{MeV}.
```

In code, delayed shapes are normalized differential spectra in MeV^-1.  The
prompt line is an integrated intensity.  For binned output only, the prompt
line is placed into the bin containing the line energy and divided by that bin
width; this is a numerical histogram convention, not physical broadening.

With beam power `P`, proton kinetic energy `E_p`, and yield per proton per
flavor `y_nu`, the source normalization is:

```math
\dot N_p = \frac{P}{E_p},
\qquad
\dot N_\nu = \dot N_p\,y_\nu.
```

The transport model is point-source dilution:

```math
G(L) = \frac{1}{4\pi L^2},
```

with `L` converted to cm in the implementation.  Thus:

```math
\phi(E) = \dot N_\nu\,G(L)\,f(E)
```

for average delayed flux, and

```math
\Phi(E) = y_\nu\,G(L)\,f(E)
```

for delayed per-POT fluence.

Current source defaults:

| Source | Beam power | Proton energy | Repetition metadata | Yield default | Script baseline |
| --- | ---: | ---: | --- | --- | ---: |
| ESS | 5.0 MW | 2.0 GeV | 14 Hz, 2.86 ms pulse | 0.3 / proton / flavor | 20 m |
| SNS FTS | 1.4 MW | 1.0 GeV | 60 Hz, 350 ns spill FWHM | PBW polynomial, about 0.09 / proton / flavor at 1 GeV | 20 m |
| J-PARC MLF | 1.0 MW | 3.0 GeV | 25 Hz, two 100 ns bunches separated by 600 ns | 0.48 / proton / flavor | 24 m |

Timing fields are metadata only.  No time-window efficiency, duty-factor
background rejection, or pulse-shape convolution is applied.

Source-normalization provenance:

- The `0.3 / proton / flavor` ESS yield at `2 GeV` is explicitly stated in the
  local axial-vector CEvNS paper, which cites the ESS CEvNS study
  arXiv:1911.00762.  That paper uses an ESS-inspired detector benchmark with
  `1.3 MW`, `20 m`, `2 keV`, `50 kg`, and `3 years`; this repository uses the
  same yield idea but keeps its own editable beam and detector configuration.
- The ESS script default `5 MW`, `2 GeV`, `14 Hz`, and `2.86 ms` pulse length
  are facility-design benchmark metadata used for rate scaling and output
  labels.  Timing is not used in the energy-spectrum calculation.
- The SNS polynomial labelled `PBW` is an effective yield model already present
  in the repository.  The README now documents its behavior, but the exact
  source paper or simulation note for the polynomial coefficients still needs
  to be attached before formal use.
- The J-PARC `0.48 / proton / flavor` default is a local benchmark assumption.
  The local axial-vector paper discusses J-PARC as a relevant facility, but the
  exact provenance of this number is still a validation gap in the codebase.

### Flux Folding

`rate_estimation.py` computes nuclear recoil spectra per target nucleus:

```math
\frac{dR}{dE_r}
=
\int dE_\nu\,
\phi(E_\nu)
\frac{d\sigma}{dE_r}(E_\nu,E_r),
```

with separate prompt, delayed `nu_e`, and delayed `anti-nu_mu` pieces.

For CF4:

```math
\mathrm{CF}_4 = 1\times {}^{12}\mathrm{C} + 4\times {}^{19}\mathrm{F}.
```

The code writes both per-nucleus components and per-molecule sums.

Numerical integration currently uses trapezoidal integration (`numpy.trapz`) on
linear recoil and neutrino-energy grids.  The default delayed neutrino grid is
from `1e-6 MeV` to the Michel endpoint.  The default nuclear recoil grid is
0 to 120 keV, and the default electron recoil grid is 0 to 60000 keV.

### Detector Scaling And Quenching

`detector_estimation.py` uses cylindrical ideal-gas bookkeeping:

```math
V = \pi R^2 L,
\qquad
n_\mathrm{mol} = \frac{PV}{R_\mathrm{gas}T},
\qquad
N_\mathrm{molecules} = n_\mathrm{mol} N_A f_\mathrm{fiducial}.
```

Detector spectra are:

```math
\frac{dN}{dE}
=
N_\mathrm{molecules}
\frac{dR}{dE}
t_\mathrm{year}.
```

For nuclear recoils, species-specific quenching-factor curves are read from
`data/quenching/C_in_CF4.csv` and `data/quenching/F_in_CF4.csv` from TRIM [MIGDAL experiment](https://arxiv.org/pdf/2207.08284).  The code maps:

```math
E_{ee}(E_r) = QF(E_r)\,E_r.
```

The QF is linearly interpolated inside the table and held constant outside the
tabulated range.  The nuclear threshold in `analysis.energy_threshold_kev` is
interpreted as a measured-energy threshold in keVee for the detector workflow.

The example detector geometry, pressure, temperature, fiducial fraction, live
time, and threshold are local working assumptions in `configs/detector_config.json`.

## Code And Tool Reference

`cevens.py`

- Purpose: target definitions and scattering kernels.
- Inputs: CLI target, neutrino energy, recoil energy, form-factor options.
- Outputs: printed or JSON single-point cross-section diagnostics.
- Main classes/functions: `NuclearTarget`, `HelmFormFactor`,
  `CEvNSCalculator`, `NeutrinoElectronCalculator`, `carbon12_target`,
  `fluorine19_target`, `cf4_electron_target`.
- Example:

```bash
python3 cevens.py --target 19F --enu-mev 30 --er-kev 5 --axial-model hoferichter_19f_central --json
```

`cohaxial/dar_flux.py`

- Purpose: shared stopped-pion DAR spectra, point-source geometry, binned line
  representation, CSV writers, and plots.
- Inputs: source-specific beam object, baseline, energy grids.
- Outputs: dictionaries of point-grid or binned flux/fluence components.
- User-facing entry points are the source scripts, not this helper module.

`ESS_flux.py`, `SNS_flux.py`, `JPARK_flux.py`

- Purpose: source-specific beam metadata plus root-level compatibility API for
  the shared DAR flux implementation.
- Inputs: hard-coded benchmark defaults, with beam classes editable/importable
  from Python.
- Outputs: source flux and fluence CSV/PNG files.
- Example:

```bash
python3 ESS_flux.py
```

`rate_estimation.py`

- Purpose: fold selected average source flux with CEvNS and neutrino-electron
  kernels to produce CF4 spectra per molecule.
- Inputs: CLI options for source, distance, axial model, output directory, and
  grid sizes.
- Outputs:
  - `cf4_differential_rate_per_molecule.csv`
  - `cf4_electron_differential_rate_per_molecule.csv`
  - component plots for CF4 composition and fluorine axial fraction.
- Example:

```bash
python3 rate_estimation.py --source ess --fluorine-axial-model hoferichter_19f_central
```

`detector_estimation.py`

- Purpose: convert per-molecule spectra to detector-level spectra using an
  ideal-gas CF4 detector config and QF remapping.
- Inputs: detector JSON, molecule-normalized rate CSVs, QF CSVs.
- Outputs: raw recoil spectra, keVee nuclear spectra, electron spectra, and
  `cf4_detector_summary.json`.
- Example:

```bash
python3 detector_estimation.py --config configs/detector_config.json
```

`scan_detector_threshold.py`

- Purpose: repeat detector scaling over a threshold grid.
- Inputs: detector config, threshold grid options.
- Outputs: threshold-scan CSV/PNG files.
- Example:

```bash
python3 scan_detector_threshold.py --threshold-min 0 --threshold-max 15 --n-thresholds 16
```

`scripts/sanity_check.py`

- Purpose: lightweight validation without adding a test framework.
- Checks: Michel spectrum normalizations, prompt fluence normalization, CEvNS
  vector+axial consistency, positive nu-e point, config parsing, and QF
  monotonicity.
- Example:

```bash
python3 scripts/sanity_check.py
```

`examples/quick_point_check.py`

- Purpose: minimal import-based point calculation.
- Example:

```bash
python3 examples/quick_point_check.py
```

## Validation And Sanity Checks

The repository currently has no formal pytest suite.  The intended quick checks
are:

```bash
python3 -m py_compile cohaxial/dar_flux.py ESS_flux.py SNS_flux.py JPARK_flux.py cevens.py rate_estimation.py detector_estimation.py scan_detector_threshold.py scripts/sanity_check.py examples/quick_point_check.py
python3 scripts/sanity_check.py
python3 cevens.py --target 19F --enu-mev 30 --er-kev 5 --json
python3 ESS_flux.py
python3 rate_estimation.py --source ess --n-er 61 --n-te 101 --n-enu 200 --output-dir cevens_rate_output_check
python3 detector_estimation.py --config configs/detector_config.json --input-csv cevens_rate_output_check/cf4_differential_rate_per_molecule.csv --input-electron-csv cevens_rate_output_check/cf4_electron_differential_rate_per_molecule.csv --output-dir detector_rate_output_check
```

The shortened-grid rate command is for smoke testing.  Use the defaults for
production-like exploratory numbers.

## Extending The Repository

Add a new source:

1. Create a root script or module with a beam metadata class exposing
   `neutrino_yield_per_proton_per_flavor` and
   `neutrinos_per_second_per_flavor`.
2. Instantiate `StoppedPionDARFluxModel` from `cohaxial/dar_flux.py`.
3. Add the source to `SOURCE_MODELS` in `rate_estimation.py`.
4. Document yield, beam, baseline, and provenance in this README.

Add a new nuclear target:

1. Add a factory in `cevens.py` returning a `NuclearTarget`.
2. Specify `Z`, `N`, mass, spin `J`, vector form factor, and axial form factor.
3. Keep units explicit: mass in GeV, recoil in keV, q^2 in GeV^2.
4. Add a sanity point and document any nuclear-response reference or gap.

Add a new axial model:

1. Implement a callable `F_A(q2_gev2)`.
2. Prefer `GenericAxialFormFactor` plus explicit `S_00`, `S_01`, `S_11` inputs
   if structure functions are available.
3. For CEvNS, check whether the reference gives transverse-only `S^T_ij` or
   total spin-dependent responses; do not silently mix dark-matter and CEvNS
   response conventions.
4. Add metadata to the target factory so output JSON/diagnostics explain the
   model choice.

Add a detector-response feature:

1. Keep the existing raw recoil and keVee views distinct.
2. Add response, efficiency, or smearing after the per-molecule-to-detector
   scaling, unless the physics specifically belongs earlier.
3. Preserve the summary JSON keys where possible, or add new keys rather than
   changing old meanings silently.

Add a plotting routine:

1. Use existing CSV outputs as inputs when possible.
2. Keep plot scripts separate from physics kernels.
3. Put generated products in ignored output directories.

## References Present In The Repository

Local PDFs and references already present in comments or notes:

- D. Aristizabal Sierra, P. M. Candela, V. De Romeri, D. K. Papoulias, and
  L. Trincado S., "Axial-vector neutral-current measurements in coherent
  elastic neutrino-nucleus scattering experiments", arXiv:2603.05281v1.  Local
  file: `Paper/AccoppiamentoAssiale_Paper_2603.05281v1 (1).pdf`.  Used here
  for the fluorine-target motivation, vector/axial CEvNS decomposition,
  Helm/Klein-Nystrand context, and ESS-like `0.3` neutrino/POT/flavor
  normalization.
- R. H. Helm, "Inelastic and Elastic Scattering of 187-MeV Electrons from
  Selected Even-Even Nuclei", Phys. Rev. 104, 1466 (1956).  Local file:
  `Paper/CrossSections/HELM_vectorModel.pdf`.  Used as the historical basis for
  the folded-density Helm form-factor model.
- S. Klein and J. Nystrand, "Exclusive vector meson production in relativistic
  heavy ion collisions", Phys. Rev. C 60, 014903 (1999), arXiv:hep-ph/9902259.
  Local file: `Paper/CrossSections/Klein-Nystrand_vectorModel.pdf`.  Present as
  an alternative vector form-factor reference, not currently implemented.
- M. Hoferichter, J. Menendez, and A. Schwenk, "Coherent elastic
  neutrino-nucleus scattering: EFT analysis and nuclear responses", Phys. Rev.
  D 102, 074018 (2020), arXiv:2007.08529.  Local file:
  `Paper/CrossSections/HMS_axialFitparams.pdf`.  Used directly for the `19F`
  axial implementation.
- P. Klos, J. Menendez, D. Gazit, and A. Schwenk, Phys. Rev. D 88, 083516
  (2013), Erratum Phys. Rev. D 89, 029901 (2014).  Cited by HMS and by the code
  comments for the two-body-current exchange integrals used in the central
  correction.
- D. Baxter et al., "Coherent Elastic Neutrino-Nucleus Scattering at the
  European Spallation Source", JHEP 02 (2020) 123, arXiv:1911.00762.  Cited in
  the local axial-vector paper for the ESS-like yield and detector context.
- G. Belanger et al., Comput. Phys. Commun. 180 (2009) 747-767, and the related
  local spin-dependent approximation PDFs under `Paper/CrossSections/AXialAprox_*`.
  These are dark-matter spin-dependent references.  They are kept as background
  but are not the default CEvNS `19F` axial model.
- Neutrino-electron scattering references already present in repository notes:
  Vogel and Engel, Phys. Rev. D 39 (1989) 3378, plus PDG neutrino cross-section
  conventions.

Documentation/validation gaps:

- The exact provenance of the SNS PBW polynomial coefficients should be checked
  against the intended SNS/COHERENT benchmark source.
- The J-PARC default yield `0.48 / proton / flavor` should be tied to the final
  source-normalization reference before use in a formal result.
- The QF CSV tables are used as input data, but their measurement/simulation
  provenance is not documented in machine-readable metadata.
- The local `Paper/` folder is ignored by git; references there are useful for
  the local researcher but should not be assumed to exist in a fresh clone.

## Known Limitations And WIP Notes

- No detector backgrounds, beam-related backgrounds, or cosmogenic backgrounds.
- No timing cuts, pulse-window optimization, or time-profile convolution.
- No detector response beyond hard threshold and QF remapping.
- No efficiency model, trigger model, reconstruction smearing, or resolution.
- No uncertainty propagation for source yield, form factors, quenching factors,
  detector geometry, gas state, or live time.
- No full hadron-production, target-transport, shielding, hall-geometry, or
  extended-source source simulation.
- The `19F` central axial option is a single central-value implementation, not
  an uncertainty scan.
- `12C` is fixed as spin-zero with no axial term.
- Recoil and neutrino-energy grids are linear and should be convergence-checked
  for any final plot or number.
- Generated outputs are useful working products, not curated reference data.
