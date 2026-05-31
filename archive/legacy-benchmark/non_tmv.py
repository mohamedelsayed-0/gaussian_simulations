import os
import numpy as np
import matplotlib.pyplot as plt

# import the existing functions from main script 
from gaussian_en_bound import (
    apply_two_mode_phase_insensitive,
    exact_log_negativity,
    reference_bound,
)

# Non-TMSV input: two-mode squeezed thermal state 

def cm_two_mode_squeezed_thermal(r: float, nbar1: float = 0.5, nbar2: float = 0.5) -> np.ndarray:
    """
    Two-mode squeezed thermal state (TMSTS) covariance matrix in shot-noise units.
    Quadrature ordering: (q1, p1, q2, p2).

    Construction:
      V_th = diag(v1, v1, v2, v2), with v_i = 2*nbar_i + 1,
      then apply two-mode squeezing symplectic S2(r):
        S = [[cosh r I,  sinh r Z],
             [sinh r Z,  cosh r I]]
      with Z = diag(1, -1).
    """
    I2 = np.eye(2)
    Z = np.diag([1.0, -1.0])

    v1 = 2.0 * nbar1 + 1.0
    v2 = 2.0 * nbar2 + 1.0
    Vth = np.diag([v1, v1, v2, v2])

    ch = np.cosh(r)
    sh = np.sinh(r)
    S = np.block([
        [ch * I2, sh * Z],
        [sh * Z,  ch * I2],
    ])

    return S @ Vth @ S.T


def plot_non_tmsv_tmsts_thermal_loss(
    eta: float = 0.6,
    Nth: float = 0.1,
    nbar_in: float = 0.5,
    r_max: float = 1.5,
    npts: int = 301,
    figdir: str = "./figs",
) -> str:
    """
    Generates one non-TMSV validation figure:
      input: TMSTS with thermal occupancy nbar_in per mode,
      channel: symmetric thermal loss (eta, Nth),
      outputs: exact EN vs reference bound E_ref.

    Saves into figdir and returns the saved path.
    """
    os.makedirs(figdir, exist_ok=True)

    r_grid = np.linspace(0.0, r_max, npts)

    # symmetric thermal loss channel parameters
    tau = eta
    nu = (1.0 - eta) * (2.0 * Nth + 1.0)

    EN = np.empty_like(r_grid)
    Eref = np.empty_like(r_grid)

    for i, r in enumerate(r_grid):
        V0 = cm_two_mode_squeezed_thermal(r, nbar1=nbar_in, nbar2=nbar_in)
        Vout = apply_two_mode_phase_insensitive(V0, tau, nu, tau, nu)

        EN[i] = exact_log_negativity(Vout)
        Eref[i] = reference_bound(Vout)

    # Plot
    plt.figure()
    plt.plot(r_grid, EN, label=r"Exact $E_N$")
    plt.plot(r_grid, Eref, label=r"Reference bound $E_{\rm ref}$")
    plt.xlabel(r"squeezing $r$")
    plt.ylabel(r"$E_N$ (nats)")
    plt.title(
        rf"Non-TMSV input (TMSTS), $\bar n={nbar_in}$; thermal loss $\eta={eta}$, $N_{{th}}={Nth}$"
    )
    plt.legend()

    outpath = os.path.join(
        figdir, f"non_tmsv_tmsts_eta_{eta}_Nth_{Nth}_nbar_{nbar_in}.png"
    )
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    return outpath


def main():
    out = plot_non_tmsv_tmsts_thermal_loss(
        eta=0.6,
        Nth=0.1,
        nbar_in=0.5,
        r_max=1.5,
        npts=301,
        figdir="./figs",
    )
    print(f"[NON-TMSV] Saved: {out}")


if __name__ == "__main__":
    main()
