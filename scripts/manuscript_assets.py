#!/usr/bin/env python3
"""Print LaTeX-ready figure includes and captions for manuscript assets."""

from __future__ import annotations


FIGURES = [
    {
        "path": "figs/n_amplifier_cascade.png",
        "label": "fig:namp_cascade",
        "caption": (
            "Amplifier cascade survival curves for g=1.3 and n=1 through 5. "
            "The per-amplifier cutoff tightens as the number of amplifiers increases."
        ),
    },
    {
        "path": "figs/amplifier_cutoff_vs_n.png",
        "label": "fig:cutoff_vs_n",
        "caption": (
            "Hard per-amplifier cutoff 2^(1/n) as a function of cascade length. "
            "The cutoff approaches unity as n increases."
        ),
    },
    {
        "path": "figs/span_limit.png",
        "label": "fig:span_limit",
        "caption": (
            "Loss-compensating span limit. The vertical markers show the strict span boundary; "
            "integer n_max values exclude exact PPT-boundary cases."
        ),
    },
    {
        "path": "figs/finite_squeezing_span_limit.png",
        "label": "fig:finite_squeezing",
        "caption": (
            "Finite-squeezing span limit as a function of squeezing budget and per-span transmissivity."
        ),
    },
    {
        "path": "figs/ordering_effect_gap.png",
        "label": "fig:ordering",
        "caption": (
            "Ordering effect for a mixed channel cascade. "
            "The gap between best and worst orderings appears because earlier noise is transformed by later channels."
        ),
    },
    {
        "path": "figs/thermal_sensitivity.png",
        "label": "fig:sensitivity",
        "caption": (
            "Thermal-noise sensitivity of the squeezing threshold. "
            "Sensitivity diverges as the channel approaches the no-entanglement boundary."
        ),
    },
    {
        "path": "figs/stochastic_outage.png",
        "label": "fig:stochastic_outage",
        "caption": (
            "Entanglement-outage probability for random loss-compensating spans. "
            "The CLT curve converts fluctuating transmissivity into a required squeezing budget."
        ),
    },
    {
        "path": "figs/asymmetric_cascade.png",
        "label": "fig:asymmetric_cascade",
        "caption": (
            "One-sided asymmetric cascade recurrence compared with exact partially transposed "
            "symplectic-spectrum computation."
        ),
    },
    {
        "path": "figs/operational_rates.png",
        "label": "fig:operational_metrics",
        "caption": (
            "Log-negativity and coherent-state teleportation fidelity computed from the scalar "
            "cascade budget."
        ),
    },
]


def main() -> None:
    for figure in FIGURES:
        print(r"\begin{figure}[t]")
        print(r"\centering")
        print(fr"\includegraphics[width=0.78\linewidth]{{{figure['path']}}}")
        print(fr"\caption{{{figure['caption']}}}")
        print(fr"\label{{{figure['label']}}}")
        print(r"\end{figure}")
        print()


if __name__ == "__main__":
    main()
