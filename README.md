# Entanglement Noise Budgets for Gaussian Channel Cascades

This repository supports a paper on closed-form entanglement survival budgets
for two-mode squeezed vacuum (TMSV) states transmitted through phase-insensitive
Gaussian channel cascades.

The main deterministic setting is symmetric transmission: the same channel acts
on both modes at each stage. In that case the partially transposed symplectic
eigenvalue reduces to a scalar affine recurrence

$$
\tilde{\nu}_-^{(m+1)} = \tau_{m+1}\tilde{\nu}_-^{(m)} + n_{m+1},
\qquad
\tilde{\nu}_-^{(0)} = e^{-2r},
$$

where $\tau$ is the channel gain/transmissivity and $n$ is the channel added
noise. For an ordered chain,

$$
\tilde{\nu}_- =
\left(\prod_{i=1}^n \tau_i\right)e^{-2r}
+ \sum_{j=1}^n n_j\prod_{k=j+1}^n\tau_k.
$$

The repository treats this as a link-budget tool: it turns the PPT condition
$\tilde{\nu}_-<1$ into explicit constraints on amplifier gain, span count,
thermal margin, channel ordering, and stochastic outage probability.

## Paper-facing results

- Thermal-loss threshold and its inverted squeezing requirement.
- Quantum-limited amplifier cutoff at $g=2$ and the $n$-amplifier budget
  $g<2^{1/n}$.
- Strict and finite-squeezing span limits for loss-compensating links.
- Optimal-ordering rule for mixed cascades, verified against brute-force
  permutation sweeps.
- Random-span entanglement-outage estimates for fluctuating transmissivity.
- Asymmetric and one-sided cascade recurrence for follow-up analysis.
- Operational metrics derived from the same budget: log-negativity and coherent
  teleportation fidelity.

These formulas do not claim that deterministic amplifiers beat loss for quantum
communication. They quantify the noise budget and no-go behavior of that
architecture.

## Reproducing results

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full workflow:

```bash
make reproduce
```

Useful individual targets:

```bash
make figures
make validate
make tests
make tables
make assets
```

The scripts write figures to `figs/` and CSV data to `data/`.

## Repository structure

```text
scripts/        scalar formulas, validation scripts, and figure generators
tests/          pytest checks against exact symplectic-spectrum computation
figs/           paper-facing generated figures
data/           generated CSV data used by figures and tables
archive/        old notes, derivations, prototype scripts, and legacy figures
```

The archived files are preserved for reference but are not part of the current
paper workflow.

## Citation and license

Citation metadata is provided in `CITATION.cff`. The code is released under the
MIT License; see `LICENSE`.

## Author

Mohamed Elsayed  
Engineering Science, University of Toronto  
https://elsayedmohamed.com/
