# Invariant Reduction for Symmetric TMSV Gaussian Channels

This project derives closed-form entanglement survival conditions for two-mode squeezed vacuum (TMSV) states transmitted through symmetric phase-insensitive Gaussian channels by working directly from the symplectic-invariant expression for the smallest partially-transposed eigenvalue.

In the symmetric setting the spectrum reduces to the scalar quantity

nu_tilde_minus = a' - c'

so the PPT condition becomes an explicit, invertible inequality in the physical channel parameters.

---

## Main results

Thermal-loss channel

Entanglement survives iff

eta * exp(-2r) + (1 - eta) * (2*N_th + 1) < 1

This gives the minimal input squeezing directly as a function of the noise.

---

Symmetric quantum-limited amplification

g * exp(-2r) + (g - 1) < 1

This yields a sharp transition:

g >= 2  ->  entanglement is impossible for any squeezing

1 < g < 2  ->  r > (1/2) * ln( g / (2 - g) )

At g = 2 the amplifier injects one shot-noise unit per mode, creating a hard noise budget that arbitrarily large squeezing cannot overcome.

---

## Why this is useful

- Eliminates repeated symplectic-spectrum evaluation in symmetric settings
- Converts the entanglement condition into a parameter-level design constraint
- Makes the threshold analytically invertible for resource estimation

---

## Repository structure

short_derivation/   invariant reduction and closed-form thresholds  
long_note/          numerical and analytic study  
figs/               generated plots  
benchmark-simulation/    code to benchmark and analyse results

---

## Status

Ongoing independent research project.

Current direction:

- extending beyond symmetric channels
- identifying cases where the invariant reduction remains analytically tractable
- developing the results toward a publishable manuscript

---

## Author

Mohamed Elsayed  
Engineering Science, University of Toronto  
https://elsayedmohamed.com/
