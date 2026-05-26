#!/usr/bin/env python3
"""Print LaTeX-ready figure includes and captions for manuscript assets."""

from __future__ import annotations


FIGURES = [
    {
        "path": "figs/thermal_loss_eta_0.6_Nth_0.5.png",
        "label": "fig:thermal-loss-threshold",
        "caption": (
            "Thermal-loss threshold validation for eta=0.6 and N_th=0.5. "
            "The exact log-negativity onset agrees with the closed-form survival condition."
        ),
    },
    {
        "path": "figs/amplifier_g_1.5.png",
        "label": "fig:single-amplifier-threshold",
        "caption": (
            "Single quantum-limited amplifier threshold for g=1.5. "
            "The vertical threshold marks the minimum squeezing required for entanglement survival."
        ),
    },
    {
        "path": "figs/n_amplifier_cascade.png",
        "label": "fig:n-amplifier-cascade",
        "caption": (
            "Amplifier cascade survival curves for g=1.3 and n=1 through 5. "
            "The per-amplifier cutoff tightens as the number of amplifiers increases."
        ),
    },
    {
        "path": "figs/amplifier_cutoff_vs_n.png",
        "label": "fig:amplifier-cutoff-vs-n",
        "caption": (
            "Hard per-amplifier cutoff 2^(1/n) as a function of cascade length. "
            "The cutoff approaches unity as n increases."
        ),
    },
    {
        "path": "figs/span_limit.png",
        "label": "fig:span-limit",
        "caption": (
            "Loss-compensating span limit. The vertical markers show the strict span boundary; "
            "integer n_max values exclude exact PPT-boundary cases."
        ),
    },
    {
        "path": "figs/finite_squeezing_span_limit.png",
        "label": "fig:finite-squeezing-span-limit",
        "caption": (
            "Finite-squeezing span limit as a function of squeezing budget and per-span transmissivity."
        ),
    },
    {
        "path": "figs/ordering_effect_gap.png",
        "label": "fig:ordering-effect-gap",
        "caption": (
            "Ordering effect for a mixed channel cascade. "
            "The gap between best and worst orderings appears because earlier noise is transformed by later channels."
        ),
    },
    {
        "path": "figs/thermal_sensitivity.png",
        "label": "fig:thermal-sensitivity",
        "caption": (
            "Thermal-noise sensitivity of the squeezing threshold. "
            "Sensitivity diverges as the channel approaches the no-entanglement boundary."
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
