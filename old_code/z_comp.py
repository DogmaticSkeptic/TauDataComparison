#!/usr/bin/env python3
"""
Compare Z boson distributions versus the sum of its two taus,
using vector.obj and explicit loop over the first N events.
Produces a 2×4 grid:
  • Top row: shaded histograms of Z and τ₁+τ₂ with √N error bars.
  • Bottom row: shaded histograms of per-event residuals (Z − (τ₁+τ₂)) with √N error bars.
Y-axis for all panels is “Entries”. Saves to plots/z_vs_taus.png.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector

def hist_with_errors(ax, data, bins, color, label):
    """
    Draw a shaded histogram of 'data' on ax with sqrt(N) error bars.
    """
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    errs = np.sqrt(counts)

    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width, color=color, alpha=0.3, label=label)
    ax.errorbar(centers, counts, yerr=errs, fmt='none',
                ecolor=color, capsize=2, linewidth=1)

def main(csv_file, n_events=10000, bins=60):
    df = pd.read_csv(csv_file)
    N  = min(n_events, len(df))

    # Allocate arrays for Z and tau-sum kinematics
    Z       = {k: np.empty(N) for k in ("pt","eta","phi","mass")}
    tau_sum = {k: np.empty(N) for k in ("pt","eta","phi","mass")}

    # Build per-event Lorentz vectors and fill arrays
    for i in range(N):
        vZ = vector.obj(
            pt   = df.at[i, 'Z_pt'],
            eta  = df.at[i, 'Z_eta'],
            phi  = df.at[i, 'Z_phi'],
            mass = df.at[i, 'Z_mass']
        )
        v1 = vector.obj(
            pt   = df.at[i, 'TauLeptonic_pt'],
            eta  = df.at[i, 'TauLeptonic_eta'],
            phi  = df.at[i, 'TauLeptonic_phi'],
            mass = df.at[i, 'TauLeptonic_mass']
        )
        v2 = vector.obj(
            pt   = df.at[i, 'TauHadronic_pt'],
            eta  = df.at[i, 'TauHadronic_eta'],
            phi  = df.at[i, 'TauHadronic_phi'],
            mass = df.at[i, 'TauHadronic_mass']
        )
        vsum = v1 + v2

        Z['pt'][i]   = vZ.pt;      tau_sum['pt'][i]   = vsum.pt
        Z['eta'][i]  = vZ.eta;     tau_sum['eta'][i]  = vsum.eta
        Z['phi'][i]  = vZ.phi;     tau_sum['phi'][i]  = vsum.phi
        Z['mass'][i] = vZ.mass;    tau_sum['mass'][i] = vsum.mass

    # Observables and labels
    observables = [
        ('pt',   r'$p_{T}$ [GeV]'),
        ('eta',  r'$\eta$'),
        ('phi',  r'$\phi$ [rad]'),
        ('mass', r'$m$ [GeV]')
    ]
    residual_labels = [
        r'$\Delta p_{T}$ [GeV]',
        r'$\Delta \eta$',
        r'$\Delta \phi$ [rad]',
        r'$\Delta m$ [GeV]'
    ]

    # Create 2×4 grid
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    # Top row: Z vs τ₁+τ₂
    for j, (key, title) in enumerate(observables):
        ax = axs[0, j]
        hist_with_errors(ax, Z[key],       bins, 'blue', 'Z boson')
        hist_with_errors(ax, tau_sum[key], bins, 'red',  r'$\tau_{1}+\tau_{2}$')
        ax.set_title(title)
        ax.set_ylabel('Entries')
        ax.legend(frameon=False)
        ax.grid(linestyle=':', alpha=0.5)

    # Bottom row: per-event residuals
    for j, (key, _) in enumerate(observables):
        ax = axs[1, j]
        residual = Z[key] - tau_sum[key]
        hist_with_errors(ax, residual, bins, 'gray', r'$Z - (\tau_{1}+\tau_{2})$')
        ax.set_title(residual_labels[j])
        ax.set_ylabel('Entries')
        ax.grid(linestyle=':', alpha=0.5)

    fig.suptitle(f'Z vs. τ₁+τ₂: Kinematics and Residuals (first {N} events)', fontsize=16)

    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/z_vs_taus.png', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Z boson and tau-lepton sum kinematics with residuals'
    )
    parser.add_argument('csv_file', help='CSV with Z and tau kinematics')
    parser.add_argument('-n', '--n_events', type=int, default=10000,
                        help='Number of events to process (default: 10000)')
    args = parser.parse_args()
    main(args.csv_file, args.n_events)

