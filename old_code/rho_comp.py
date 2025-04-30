#!/usr/bin/env python3
"""
Compare kinematic distributions of the ρ-meson versus the π⁺+π⁰ sum,
using vector.obj and explicit loop over the first N events.
Produces a 2×4 grid: top row shaded histograms with √N error bars,
bottom row histograms of residuals (ρ – (π⁺+π⁰)) with √N error bars.
Saves the figure to plots/rho_comparison.png.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector

def hist_with_errors(ax, data, bins, color, label):
    """
    Draw a shaded histogram of data on ax with sqrt(N) error bars.
    """
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    errs = np.sqrt(counts)

    width = edges[1] - edges[0]
    ax.bar(centers, counts, width=width, color=color, alpha=0.3, label=label)
    ax.errorbar(centers, counts, yerr=errs, fmt='none',
                ecolor=color, capsize=2, linewidth=1)

def main(csv_file, n_events=10000, bins=60):
    # Load data
    df = pd.read_csv(csv_file)
    N  = min(n_events, len(df))

    # Pre-allocate arrays
    rho = {
        'pt':   np.empty(N),
        'eta':  np.empty(N),
        'phi':  np.empty(N),
        'mass': np.empty(N)
    }
    summ = {
        'pt':   np.empty(N),
        'eta':  np.empty(N),
        'phi':  np.empty(N),
        'mass': np.empty(N)
    }

    # Loop and build four-vectors
    for i in range(N):
        v_rho = vector.obj(
            pt   = df.at[i, 'Rho_pt'],
            eta  = df.at[i, 'Rho_eta'],
            phi  = df.at[i, 'Rho_phi'],
            mass = df.at[i, 'Rho_mass']
        )
        v_ch = vector.obj(
            pt   = df.at[i, 'ChargedPion_pt'],
            eta  = df.at[i, 'ChargedPion_eta'],
            phi  = df.at[i, 'ChargedPion_phi'],
            mass = df.at[i, 'ChargedPion_mass']
        )
        v_pi0 = vector.obj(
            pt   = df.at[i, 'NeutralPion_pt'],
            eta  = df.at[i, 'NeutralPion_eta'],
            phi  = df.at[i, 'NeutralPion_phi'],
            mass = df.at[i, 'NeutralPion_mass']
        )
        v_sum = v_ch + v_pi0

        rho['pt'][i]   = v_rho.pt
        rho['eta'][i]  = v_rho.eta
        rho['phi'][i]  = v_rho.phi
        rho['mass'][i] = v_rho.mass

        summ['pt'][i]   = v_sum.pt
        summ['eta'][i]  = v_sum.eta
        summ['phi'][i]  = v_sum.phi
        summ['mass'][i] = v_sum.mass

    # Observable definitions
    observables = [
        ('pt',   r'Transverse momentum $p_T$ [GeV]',    'blue'),
        ('eta',  r'Pseudorapidity $\eta$',               'blue'),
        ('phi',  r'Azimuthal angle $\phi$ [rad]',        'blue'),
        ('mass', r'Invariant mass [GeV]',                'blue'),
    ]

    residual_labels = [
        (r'$\Delta p_T$ [GeV]'),
        (r'$\Delta \eta$'),
        (r'$\Delta \phi$ [rad]'),
        (r'$\Delta m$ [GeV]'),
    ]

    # Create 2×4 grid
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    # Top row: distributions
    for j, (key, xlabel, color) in enumerate(observables):
        ax = axs[0, j]
        hist_with_errors(ax, rho[key], bins, color='blue', label='ρ')
        hist_with_errors(ax, summ[key], bins, color='red',  label='π⁺+π⁰')
        ax.set_title(xlabel)
        ax.set_ylabel('Entries')
        ax.legend(frameon=False)
        ax.grid(linestyle=':', alpha=0.5)

    # Bottom row: residual histograms
    for j, (key, _, _) in enumerate(observables):
        ax = axs[1, j]
        residual = rho[key] - summ[key]
        hist_with_errors(ax, residual, bins, color='gray', label='ρ − (π⁺+π⁰)')
        ax.set_title(residual_labels[j])
        ax.set_ylabel('Entries')
        ax.grid(linestyle=':', alpha=0.5)

    fig.suptitle(f'ρ vs. π⁺+π⁰: Kinematics and Residuals (first {N} events)', fontsize=16)

    # Save output
    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/rho_comparison.png', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ρ vs. π⁺+π⁰ kinematics and residual histograms with vector.obj'
    )
    parser.add_argument('csv_file', help='Input CSV file (lep_rho_z_valid.csv)')
    parser.add_argument('-n', '--n_events', type=int, default=10000,
                        help='Number of events to process (default: 10000)')
    args = parser.parse_args()
    main(args.csv_file, args.n_events)

