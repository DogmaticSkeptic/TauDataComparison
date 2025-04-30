#!/usr/bin/env python3
"""
Produce raw‐count histograms (with Poisson uncertainties) for a set
of predefined observables from a CSV file, after selecting events
by muon_pt, hadronic tau_pt, di-tau mass, and scattering angle.

Usage:
    python single_histograms_raw.py --input path/to/data.csv [--max-rows N] [--output-dir plots]
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector
from tqdm import tqdm
tqdm.pandas(desc="Computing di-tau kinematics")

# Observable definitions: (x-axis label, column name)
OBSERVABLES = [
    (r"$p_{T}^{\mathrm{lep\tau}}$ [GeV]",        "TauLeptonic_pt"),
    (r"$\eta^{\mathrm{lep\tau}}$",               "TauLeptonic_eta"),
    (r"$\phi^{\mathrm{lep\tau}}$ [rad]",         "TauLeptonic_phi"),
    (r"$m^{\mathrm{lep\tau}}$ [GeV]",            "TauLeptonic_mass"),
    (r"$p_{T}^{\mathrm{had\tau}}$ [GeV]",        "TauHadronic_pt"),
    (r"$\eta^{\mathrm{had\tau}}$",               "TauHadronic_eta"),
    (r"$\phi^{\mathrm{had\tau}}$ [rad]",         "TauHadronic_phi"),
    (r"$m^{\mathrm{had\tau}}$ [GeV]",            "TauHadronic_mass"),
    (r"$p_{T}^{\pi^{\pm}}$ [GeV]",               "ChargedPion_pt"),
    (r"$\eta^{\pi^{\pm}}$",                      "ChargedPion_eta"),
    (r"$\phi^{\pi^{\pm}}$ [rad]",                "ChargedPion_phi"),
    (r"$m^{\pi^{\pm}}$ [GeV]",                   "ChargedPion_mass"),
    (r"$p_{T}^{\pi^{0}}$ [GeV]",                 "NeutralPion_pt"),
    (r"$\eta^{\pi^{0}}$",                        "NeutralPion_eta"),
    (r"$\phi^{\pi^{0}}$ [rad]",                  "NeutralPion_phi"),
    (r"$m^{\pi^{0}}$ [GeV]",                     "NeutralPion_mass"),
    (r"$p_{T}^{\mu}$ [GeV]",                     "Muon_pt"),
    (r"$\eta^{\mu}$",                            "Muon_eta"),
    (r"$\phi^{\mu}$ [rad]",                      "Muon_phi"),
    (r"$m^{\mu}$ [GeV]",                         "Muon_mass"),
    (r"$p_{T}^{Z}$ [GeV]",                       "Z_pt"),
    (r"$\eta^{Z}$",                              "Z_eta"),
    (r"$\phi^{Z}$ [rad]",                        "Z_phi"),
    (r"$m^{Z}$ [GeV]",                           "Z_mass"),
    (r"$p_{T}^{\nu_{\mu}~(\mathrm{lep~\tau})}$ [GeV]", "nu_mu_fromLeptonicTau_pt"),
    (r"$\eta^{\nu_{\mu}~(\mathrm{lep~\tau})}$",       "nu_mu_fromLeptonicTau_eta"),
    (r"$\phi^{\nu_{\mu}~(\mathrm{lep~\tau})}$ [rad]", "nu_mu_fromLeptonicTau_phi"),
    (r"$m^{\nu_{\mu}~(\mathrm{lep~\tau})}$ [GeV]",    "nu_mu_fromLeptonicTau_mass"),
    (r"$p_{T}^{\nu_{\tau}~(\mathrm{lep~\tau})}$ [GeV]", "nu_tau_fromLeptonicTau_pt"),
    (r"$\eta^{\nu_{\tau}~(\mathrm{lep~\tau})}$",       "nu_tau_fromLeptonicTau_eta"),
    (r"$\phi^{\nu_{\tau}~(\mathrm{lep~\tau})}$ [rad]", "nu_tau_fromLeptonicTau_phi"),
    (r"$m^{\nu_{\tau}~(\mathrm{lep~\tau})}$ [GeV]",    "nu_tau_fromLeptonicTau_mass"),
    (r"$p_{T}^{\nu_{\tau}~(\mathrm{had~\tau})}$ [GeV]", "nu_tau_fromHadronicTau_pt"),
    (r"$\eta^{\nu_{\tau}~(\mathrm{had~\tau})}$",       "nu_tau_fromHadronicTau_eta"),
    (r"$\phi^{\nu_{\tau}~(\mathrm{had~\tau})}$ [rad]", "nu_tau_fromHadronicTau_phi"),
    (r"$m^{\nu_{\tau}~(\mathrm{had~\tau})}$ [GeV]",    "nu_tau_fromHadronicTau_mass"),
    (r"$m_{\mathrm{di}\text{-}\tau}$ [GeV]",      "m_ditau"),
    (r"$\theta_{\mathrm{scat}}$ [rad]",          "scattering_angle"),
]

def compute_ditau_kinematics(row: pd.Series) -> pd.Series:
    """
    Compute the di-tau invariant mass and scattering angle.
    Scattering angle is defined as the normalized polar angle of the hadronic tau
    in the di-tau CM frame, scaled to [0,1]: θ = (2/π)*arccos(|cos(θ_lab)|).
    Returns a Series with 'm_ditau' (GeV) and 'scattering_angle' (rad).
    """
    tau_lep = vector.obj(
        pt=row["TauLeptonic_pt"],
        eta=row["TauLeptonic_eta"],
        phi=row["TauLeptonic_phi"],
        mass=row["TauLeptonic_mass"]
    )
    tau_had = vector.obj(
        pt=row["TauHadronic_pt"],
        eta=row["TauHadronic_eta"],
        phi=row["TauHadronic_phi"],
        mass=row["TauHadronic_mass"]
    )
    ditau = tau_lep + tau_had
    m_ditau = ditau.mass

    # Boost hadronic tau into di-tau CM frame
    tau_had_cm = tau_had.boostCM_of(ditau)
    p_vec = np.array([tau_had_cm.x, tau_had_cm.y, tau_had_cm.z])
    p_mag = np.linalg.norm(p_vec)
    if p_mag == 0:
        scattering = np.nan
    else:
        cos_theta = np.clip(tau_had_cm.z / p_mag, -1.0, 1.0)
        scattering = 2 * np.arccos(np.abs(cos_theta)) / np.pi

    return pd.Series({"m_ditau": m_ditau, "scattering_angle": scattering})

def compute_histogram_with_errors(data: np.ndarray, bin_edges: np.ndarray):
    """
    Compute raw counts and Poisson uncertainties for each bin.
    """
    counts, _ = np.histogram(data, bins=bin_edges)
    errors    = np.sqrt(counts)
    return counts, errors

def plot_histogram(bin_edges: np.ndarray,
                   counts: np.ndarray,
                   errors: np.ndarray,
                   xlabel: str,
                   output_path: str):
    """
    Render and save a raw-count histogram with error bars.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = np.diff(bin_edges)

    plt.figure(figsize=(8, 6))
    plt.bar(centers, counts, width=widths, color='C0', alpha=0.7, edgecolor='black')
    plt.errorbar(centers, counts, yerr=errors, fmt='none', ecolor='black', capsize=3)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def determine_bins(col: str, data: np.ndarray):
    """
    Select bin edges according to observable type.
    """
    if col.endswith("_pt"):
        return np.linspace(0, 100, 51)
    if col in {"TauLeptonic_mass", "TauHadronic_mass"}:
        return np.linspace(1.7, 1.85, 51)
    if col in {"ChargedPion_mass", "NeutralPion_mass"}:
        return np.linspace(0.1, 0.25, 51)
    if col == "Muon_mass":
        return np.linspace(0.05, 0.15, 51)
    if col.endswith("mass"):
        return np.linspace(0, 0.1, 51)
    if col == "Z_mass":
        return np.linspace(50, 150, 51)
    if col == "Z_eta":
        return np.linspace(-12, 12, 51)
    if col == "Z_phi":
        return np.linspace(-np.pi, np.pi, 51)
    if col == "m_ditau":
        return np.linspace(80, 100, 41)
    if col == "scattering_angle":
        return np.linspace(0.6, 1.0, 41)
    mn, mx = data.min(), data.max()
    margin = 0.05 * (mx - mn)
    return np.linspace(mn - margin, mx + margin, 51)

def main():
    parser = argparse.ArgumentParser(
        description="Generate raw-count histograms with event selection."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="plots",
        help="Directory to save histogram plots (default: plots)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read and trim
    df = pd.read_csv(args.input)
    # Compute di-tau kinematics
    kin = df.progress_apply(compute_ditau_kinematics, axis=1)
    df = pd.concat([df, kin], axis=1)

    # Apply selection cuts
    df = df[
        (df["Muon_pt"] > 12) &
        (df["TauHadronic_pt"] >= 25) &
        (df["m_ditau"].between(80, 100)) &
        (df["scattering_angle"].between(0.6, 1.0))
    ]

    # Loop over observables and plot
    for xlabel, col in tqdm(OBSERVABLES, desc="Plotting observables"):
        if col not in df.columns:
            continue
        data = df[col].dropna().to_numpy()
        if data.size == 0:
            continue

        bins   = determine_bins(col, data)
        counts, errors = compute_histogram_with_errors(data, bins)

        out_path = os.path.join(args.output_dir, f"{col}.png")
        plot_histogram(bins, counts, errors, xlabel, out_path)

    print(f"All histograms saved in '{args.output_dir}'")

if __name__ == "__main__":
    main()

