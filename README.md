# CEvNS Feasibility Toolkit

A lightweight Python repository for fast feasibility studies of coherent elastic neutrino-nucleus scattering (CEvNS), with an initial focus on light nuclei relevant to CF4-based detector concepts.

At the current stage, the repository contains two main physics modules:

- a **CEvNS cross-section module** with explicit **vector** and **axial** contributions
- an **ESS pion-DAR neutrino flux generator**

and one minimal example script showing how to use the CEvNS module for `12C` and `19F`.

This is therefore already a useful **physics-kernel repository** for interaction-level and source-level studies, but it is **not yet** a full end-to-end event-rate framework. In particular, the current code does **not yet** fold together:

- neutrino flux
- target inventory
- detector mass / pressure / live time
- threshold and acceptance
- recoil smearing / detector response
- backgrounds

into final observable spectra or sensitivity projections.

---

## Current scope

The repository presently covers two layers of the CEvNS problem:

### 1. Interaction physics

The `cevens.py` module computes differential CEvNS cross sections with a clean separation between:

- nuclear target definition
- vector form factor
- axial form factor
- kinematics
- final cross-section evaluation

It includes built-in examples for:

- `12C`
- `19F`
- stoichiometric mixtures such as `CF4`

### 2. Neutrino source modeling

The `ESS_flux.py` module generates neutrino fluxes for an **ESS-like pion decay-at-rest source**, including:

- prompt monochromatic `νμ` from `π+ -> μ+ νμ`
- delayed `νe` and `\barνμ` Michel spectra from `μ+` decay
- isotropic geometric dilution with distance
- CSV export
- plot generation

---

## Repository contents

A minimal expected layout is:

```text
.
├── cevens.py
├── ESS_flux.py
├── example_cevens_scan.py
└── README.md