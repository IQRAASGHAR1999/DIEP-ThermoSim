# DIEP-ThermoSim

**Physics-informed synthetic data and deep learning for automated perforator detection in Dynamic Infrared Thermography (DIRT).**
A reproducible pipeline that couples a 3D Pennes bioheat solver with a spatio-temporal U-Net to detect perforators from cooling–rewarming thermal video. Designed as the simulation engine that complements clinical DIRT studies in DIEP-flap breast reconstruction surgery.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Motivation

Dynamic Infrared Thermography (DIRT) is a non-invasive, radiation-free alternative to CT angiography for **perforator mapping in DIEP-flap breast reconstruction**. Recent clinical work has demonstrated that deep learning can automate perforator detection from DIRT video (Clarys et al. 2025; Cardenas De La Hoz et al. 2024). However, two challenges remain:

1. **Data scarcity.** Annotated clinical DIRT datasets are small. Training data-hungry models is hard.
2. **Lack of ground truth at depth.** Surface thermography reveals projections of subsurface vessels, but the relationship between vessel geometry (depth, radius, perfusion rate) and the resulting surface signature is non-trivial.

This repository addresses both by **generating physics-grounded synthetic DIRT sequences** with a finite-difference Pennes bioheat solver, and training a spatio-temporal U-Net on the synthetic data with full ground-truth labels for vessel location *and* clinical-proxy quality. The synthetic engine can also serve as a data-augmentation step for clinical models.

## Background and methodological lineage

This project is a direct methodological continuation of my MSc thesis work, accepted as **DynGS-Pro: Dynamic Scene Rendering Using Progressive Gaussian Splatting and Deformation Fields** (KSEM 2026). DynGS-Pro combined an explicit primitive representation (Gaussians) with a learned deformation field network to model non-rigid 3D scene dynamics, an instance of the broader hybrid pattern: an explicit forward model plus a learned residual.

DIEP-ThermoSim transposes the same hybrid pattern from visual to thermal observations:

| Component | DynGS-Pro (thesis) | DIEP-ThermoSim (this project) |
|---|---|---|
| Forward model | Gaussian primitive representation | Pennes bioheat PDE on a voxel grid |
| Learned residual | Deformation field network over (x, y, z, t) | Thermal deformation field over (x, y, t) (see `deformation_field.py`) |
| Inverse problem | Recover 3D scene from 2D image views | Recover subsurface vessel geometry from 2D surface thermography |
| Dynamics | Smooth non-rigid scene motion | Smooth thermal evolution during rewarming |

The deformation field module in this repository (`deformation_field.py`) is an explicit port of the thesis architecture: a small MLP operating on Fourier-encoded spatio-temporal coordinates with a mid-layer skip connection. In the thesis it predicted 3D point displacements; here it predicts per-pixel thermal residuals that correct the bulk PDE prediction for patient-specific or instrument-specific effects (skin texture, sweat, non-uniform cooling, fine-scale perfusion heterogeneity).

This lineage is what makes the project a natural extension of my prior work rather than a one-off exercise. The thermal inverse problem is a new domain for me; the hybrid forward-model-plus-learned-residual methodology is not.

## Method overview

```
            ┌─────────────────────┐    ┌────────────────────┐
random      │  Pennes bioheat     │    │ Spatio-temporal    │
vessel  ──► │  3D FDM solver      │──► │ U-Net (3D conv +   │ ──► mask + quality
config      │  (cool → rewarm)    │    │   2D U-Net decoder)│
            └─────────────────────┘    └────────────────────┘
                       │
                  thermal video        ground-truth from
                  + GT mask            simulated geometry
```

### The physics

The Pennes bioheat equation governs heat transfer in perfused tissue:

```
ρ · c · ∂T/∂t = ∇·(k ∇T) + ρ_b · w_b · c_b · (T_a − T) + Q_m
```

where ρ, c, k are tissue density / specific heat / thermal conductivity, w_b is blood perfusion rate, ρ_b · c_b is the blood heat-capacity factor, T_a is arterial blood temperature, and Q_m is metabolic heat generation. Perforators are modelled as cylindrical regions of elevated w_b rising vertically from the muscle through the fat layer.

We discretize on a regular 3D voxel grid using central differences (equivalent to first-order finite elements on hex meshes for this PDE) and step forward in time with explicit Euler. A Robin boundary condition at the skin surface implements the cooling–rewarming protocol used in clinical DIRT (Thiessen et al. 2020; Clarys et al. 2025): controlled cooling to ~18 °C, then natural rewarming at ambient temperature with convective heat exchange.

### The learning task

Each simulation produces:
- A thermal video [T frames, H, W] of skin-surface temperature during rewarming
- A soft ground-truth mask [H, W] of vessel projections
- A per-pixel quality map [H, W] proxying clinical perforator strength

A compact **PerforatorNet** (3D conv temporal encoder + 2D U-Net decoder + dual heads) predicts both maps. Training uses BCE + Dice for the mask and a quality-region-weighted MSE for the quality head. **MC Dropout** at inference produces calibrated uncertainty maps that align with ambiguous low-signal regions, which is essential for clinical adoption.

## Project structure

```
DIEP-ThermoSim/
├── README.md
├── LICENSE                       # MIT
├── requirements.txt
├── .gitignore
├── bioheat.py                    # Pennes 3D FDM solver
├── deformation_field.py          # thermal residual MLP (port of DynGS-Pro architecture)
├── dataset.py                    # synthetic data generation + PyTorch Dataset
├── model.py                      # PerforatorNet + losses
├── train.py                      # training loop + MC dropout + metrics
├── configs/
│   └── default.yaml              # full hyperparameter set
└── docs/
    ├── GITHUB_UPLOAD_GUIDE.md
    └── POSITIONING_STRATEGY.md
```

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. smoke test the physics solver (no GPU needed)
python bioheat.py

# 3. smoke test the model
python model.py

# 4. end-to-end smoke test (generates 6 samples and runs 2 epochs)
python train.py --smoke-test

# 5. real run: generate dataset then train
python dataset.py --generate --out data/train --n-samples 200 --seed 0
python dataset.py --generate --out data/val   --n-samples 40  --seed 1
python train.py --train-dir data/train --val-dir data/val --epochs 30
```

## Reproducing the headline result

A run on 200 train / 40 val synthetic sequences (24-frame inputs, 24×24 mm skin patches) reaches:

| Metric | Value |
|---|---|
| Localization F1 (peak match, tol = 3 px) | *fill after run* |
| Quality MAE on detected perforators | *fill after run* |
| ECE on mask probabilities | *fill after run* |
| Inference time (CPU, 24 frames) | *fill after run* |

> The README ships with placeholders. After training, the script writes `runs/exp1/history.json`. Drop the final values into the table; this is intentionally not auto-generated to keep numbers honest.

## Connection to clinical practice at InViLab (UAntwerp)

This project was scoped to complement the work of the InViLab group at the University of Antwerp on DIRT for DIEP-flap reconstruction. Relevant prior art:

- Clarys, W., Evans, R., Verstockt, J., Verspeek, S., Zhang, H., Steenackers, G., et al. (2025). *Optimising neural networks for perforator detection in DIEP flap breast reconstruction using dynamic infrared thermography.* Quantitative InfraRed Thermography Journal.
- Cardenas De La Hoz, E., Verstockt, J., Verspeek, S., Clarys, W., Steenackers, G., Vanlanduit, S. (2024). *Automated thermographic detection of blood vessels for DIEP flap reconstructive surgery.* International Journal of Computer Assisted Radiology and Surgery.
- Clarys, W., Verspeek, S., Verstockt, J., Hummelink, S., Steenackers, G., Thiessen, F. (2025). *Comparative study of cooling techniques for perforator detection in DIEP flap reconstruction using dynamic infrared thermography.*
- Thiessen, F. E. F. et al. (2020). *DIRT in DIEP flap breast reconstruction: A clinical study with a standardized measurement setup.*
- Steenackers, G. et al. (2019). *Infrared Thermography for DIEP Flap Breast Reconstruction Part II.*

The repository's purpose is **not** to replace clinical work but to provide a controllable, fully-labelled simulation environment alongside it. Synthetic data can be used for: pretraining before fine-tuning on small clinical cohorts; ablating which physical factors (vessel depth, perfusion rate, cooling protocol) most affect detectability; and rigorously testing uncertainty calibration where the ground truth is known exactly.

## Limitations and honest caveats

- **The bioheat model is simplified.** Real abdominal tissue has spatially varying material properties beyond a 4-layer stack, anisotropic perfusion, and curvature; the current model approximates these.
- **Synthetic-to-real gap is not yet measured.** This repository establishes the simulation engine. Quantifying domain shift on a clinical cohort is future work and would require institutional collaboration.
- **2-6 perforators per field is the modelled range.** Configurations outside this need re-tuning of the rejection-sampling parameters in `dataset.py`.
- **The "quality" label is a geometric proxy** (perfusion × radius × inverse depth), not a clinical outcome. It is a starting point for a learnable feature, not a replacement for clinical grading.

## Roadmap (in priority order)

1. Replace forward Euler with implicit Crank–Nicolson for larger stable time steps.
2. Add curvature: simulate over a curved patch with normal-direction perforators.
3. Domain-randomization study: vary tissue properties, cooling severity, camera noise — measure when the learned detector breaks.
4. Comparison of MC Dropout vs Deep Ensembles for uncertainty calibration on this task.
5. Integration test against a real DIRT clip (if a publicly released clip becomes available).

## Citation

```bibtex
@misc{asghar2026diepthermosim,
  author = {Asghar, Iqra},
  title  = {DIEP-ThermoSim: Physics-Informed Synthetic Data for Perforator Detection in Dynamic Infrared Thermography},
  year   = {2026},
  url    = {https://github.com/IQRAASGHAR1999/DIEP-ThermoSim}
}
```

## License

MIT. See `LICENSE`.

## Acknowledgments

The Pennes bioheat equation framework follows Pennes (1948) and the multilayer skin parameterisation borrows from the ITIS Foundation tissue properties database. The clinical context and DIRT acquisition protocol descriptions follow the InViLab group's published work cited above.
