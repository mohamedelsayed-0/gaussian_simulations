# Invariant Reduction for Symmetric TMSV Gaussian Channels

This repository derives closed-form entanglement survival conditions for two-mode squeezed vacuum (TMSV) states transmitted through symmetric phase-insensitive Gaussian channels by working directly from the symplectic-invariant expression for the smallest partially-transposed eigenvalue.

In the symmetric setting the spectrum reduces to the scalar quantity

$$
\tilde{\nu}_- = a' - c'
$$

so the PPT condition becomes an explicit, invertible inequality in the physical channel parameters.

---

## Main results

### Thermal-loss channel

Entanglement survives iff

$$
\eta e^{-2r} + (1-\eta)(2N_{\rm th}+1) < 1
$$

This gives the minimal input squeezing directly as a function of the noise.

### Symmetric quantum-limited amplification

$$
g e^{-2r} + (g-1) < 1
$$

This yields a sharp transition:

$$
g \ge 2 \quad \Rightarrow \quad \text{entanglement is impossible for any squeezing}
$$

$$
1 < g < 2 \quad \Rightarrow \quad r > \frac{1}{2}\ln\!\left(\frac{g}{2-g}\right)
$$

At g = 2 the amplifier injects one shot-noise unit per mode, creating a hard noise budget that arbitrarily large squeezing cannot overcome.

---

## Why this is useful

- Eliminates repeated symplectic-spectrum evaluation in symmetric settings
- Converts the entanglement condition into a parameter-level design constraint
- Makes the threshold analytically invertible for resource estimation

---

## Scope

The main reduction applies to symmetric phase-insensitive Gaussian channels acting independently on both modes of a TMSV input state. The numerical scripts compare the closed-form behavior with full partially-transposed symplectic-spectrum evaluation and include one non-TMSV check as supporting context.

---

## Reproducing figures

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the figures:

```bash
sh scripts/reproduce_figures.sh
```

The scripts write generated plots to `figs/`.

Additional cascade utilities:

```bash
python scripts/cascade_figures.py
python scripts/validate_cascade.py
python scripts/n_max_table.py
```

These generate the cascade figures, validate the scalar cascade formulas
against exact symplectic-eigenvalue computation, and print the
loss-compensating span table for manuscript use.

---

## Repository structure

```text
short_derivation/       invariant reduction and closed-form thresholds
long_note/              numerical and analytic study
figs/                   generated plots
benchmark-simulation/   code to benchmark and analyse results
scripts/                reproducibility entry points
```

---

## Notes

This repository contains the derivation notes, numerical study, generated figures,
and benchmark scripts for the Gaussian-channel entanglement calculations.

---

## Citation and license

Citation metadata is provided in `CITATION.cff`. The repository code is released under the MIT License; see `LICENSE`.

---

## Author

Mohamed Elsayed  
Engineering Science, University of Toronto  
https://elsayedmohamed.com/
