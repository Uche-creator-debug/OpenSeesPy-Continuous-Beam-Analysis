# Three-Span Continuous Beam Analysis — OpenSeesPy

An elastic Euler–Bernoulli continuous beam model with three spans (3 m, 3 m, 8 m) and four supports (fixed, roller, roller, pin).

## Features

- Solves nodal reactions and element-end resisting forces.
- Checks vertical force and moment equilibrium and bending moment continuity.
- Produces shear force and bending moment diagrams with key critical moment markers.
- Computes a within-span vertical deflection profile from beam interpolation plus the uniformly distributed load contribution.
- Studies the effect of changing Element 3's section depth by ±20%.
- Saves four publication-ready PNG figures in `continuous_beam_results/`.

## Setup

```bash
python -m pip install -r requirements.txt
python Continuous_Beam_Complete.py
```

Run in the same Python environment as Spyder if using Spyder. The program generates its outputs beside the script. Inputs are editable near the top of the file.

## Units and sign convention

- Model: N, mm, N/mm², N/mm, N·mm.
- Output figures: kN, kN·m, m, mm.
- Bending moment: sagging positive, hogging negative.
- Deflection: vertical upward positive; downward is negative.

## Model assumptions

Linear elastic, small displacement, 2D Euler–Bernoulli beam behaviour; no shear deformation, material nonlinearity, geometric nonlinearity, or detailed design-code checks. The deflection plot contains an analytical uniformly distributed load particular solution as well as interpolated nodal DOFs; it is not obtained simply by plotting the four vertical support displacements.
