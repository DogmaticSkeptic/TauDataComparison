import os
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from numpy.linalg   import eig
from cos_utils import unit_vector, compute_cos_observables_vector, compute_normalized_histogram
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import vector
from tqdm import tqdm
import numpy.linalg as la
import textwrap
import vector   # ensure Vector4D is on the namespace
import argparse
from plotting_functions import (
    overlay_plot_with_fits_B,
    overlay_plot_with_fits_C,
    raw_histogram_plot
)
import plotting_functions
from cos_utils import *

from pdf_summary import create_summary_pdf

# Enable progress_apply for pandas.
tqdm.pandas()

# =============================================================================
# Settings and File Names
# =============================================================================
# now both Aaron and Yulei come from CSV exports
AARON_FILE = "./data/lep_rho_z_valid.csv"
YULEI_FILE = "./data/yulei_code_data.csv"

# Define the six B observables.
B_OBSERVABLES = [
    (r"$\cos\theta_{r}^{\mathrm{lep}}$", "leptonic_cos_r"),
    (r"$\cos\theta_{n}^{\mathrm{lep}}$", "leptonic_cos_n"),
    (r"$\cos\theta_{k}^{\mathrm{lep}}$", "leptonic_cos_k"),
    (r"$\cos\theta_{r}^{\mathrm{had}}$", "hadronic_cos_r"),
    (r"$\cos\theta_{n}^{\mathrm{had}}$", "hadronic_cos_n"),
    (r"$\cos\theta_{k}^{\mathrm{had}}$", "hadronic_cos_k")
]
LEP_KEYS = ["leptonic_cos_r", "leptonic_cos_n", "leptonic_cos_k"]
HAD_KEYS = ["hadronic_cos_r", "hadronic_cos_n", "hadronic_cos_k"]
PRODUCT_OBSERVABLES = [
    (r"C_{rr}", "C_rr"),
    (r"C_{rn}", "C_rn"),
    (r"C_{rk}", "C_rk"),
    (r"C_{nr}", "C_nr"),
    (r"C_{nn}", "C_nn"),
    (r"C_{nk}", "C_nk"),
    (r"C_{kr}", "C_kr"),
    (r"C_{kn}", "C_kn"),
    (r"C_{kk}", "C_kk"),
]
# -----------------------------------------------------------------------------
kappa_lep = -0.34   # For muonic decay
kappa_had = 0.41    # For hadronic (rho) decay
kappa0 = kappa_lep
kappa1 = kappa_had

# =============================================================================
# Main Processing and Summary
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Overlay cosine distributions")
    parser.add_argument("-e", type=int, default=1000,
                        help="Maximum number of events to load from the datasets")
    args = parser.parse_args()

    # Read Aaron and Yulei from CSV
    try:
        df_aaron = pd.read_csv(AARON_FILE).head(args.e)
        df_yulei = pd.read_csv(YULEI_FILE).head(args.e)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Standardize only the Yulei columns so downstream code sees TauLeptonic_*, etc.
    rename_map = {
        "tau_m_PT":   "TauLeptonic_pt",
        "tau_m_Eta":  "TauLeptonic_eta",
        "tau_m_Phi":  "TauLeptonic_phi",
        "tau_m_Mass": "TauLeptonic_mass",
        "tau_p_PT":   "TauHadronic_pt",
        "tau_p_Eta":  "TauHadronic_eta",
        "tau_p_Phi":  "TauHadronic_phi",
        "tau_p_Mass": "TauHadronic_mass",
        "mu_m_PT":    "Muon_pt",
        "mu_m_Eta":   "Muon_eta",
        "mu_m_Phi":   "Muon_phi",
        "mu_m_Mass":  "Muon_mass",
        "rho_p_PT":   "Rho_pt",
        "rho_p_Eta":  "Rho_eta",
        "rho_p_Phi":  "Rho_phi",
        "rho_p_Mass": "Rho_mass",
    }
    # no renaming required: both CSVs use the unified column schema already

    # ——————————————————————————————————————————————————————————————
    # expose precomputed cosines from the PKL
    df_yulei["hadronic_cos_r_pre"] = df_yulei["cos_theta_B_r"]
    df_yulei["hadronic_cos_n_pre"] = df_yulei["cos_theta_B_n"]
    df_yulei["hadronic_cos_k_pre"]   = df_yulei["cos_theta_B_k"]
    df_yulei["leptonic_cos_r_pre"]   = df_yulei["cos_theta_A_r"]
    df_yulei["leptonic_cos_n_pre"]   = df_yulei["cos_theta_A_n"]
    df_yulei["leptonic_cos_k_pre"]   = df_yulei["cos_theta_A_k"]
    # ——————————————————————————————————————————————————————————————

    # Optional debug:
    # print("Aaron columns:", df_aaron.columns.tolist())
    # print("Yulei columns:", df_yulei.columns.tolist())

    # Prepare output directory
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # Compute di-tau kinematics
    kin_m = df_aaron.progress_apply(plotting_functions.compute_ditau_kinematics, axis=1)
    df_aaron[["m_ditau", "scattering_angle"]] = kin_m
    kin_y = df_yulei.progress_apply(plotting_functions.compute_ditau_kinematics, axis=1)
    df_yulei[["m_ditau", "scattering_angle"]] = kin_y

    # Store distributions before cuts
    m_before_m = df_aaron["m_ditau"].to_numpy()
    theta_before_m = df_aaron["scattering_angle"].to_numpy()
    m_before_y = df_yulei["m_ditau"].to_numpy()
    theta_before_y = df_yulei["scattering_angle"].to_numpy()

    # Apply cuts
    mask_m = (
        (df_aaron["m_ditau"] >= 80) & (df_aaron["m_ditau"] <= 100) &
        (df_aaron["scattering_angle"] >= 0.6) & (df_aaron["scattering_angle"] <= 1.0)
    )
    mask_y = (
        (df_yulei["m_ditau"] >= 80) & (df_yulei["m_ditau"] <= 100) &
        (df_yulei["scattering_angle"] >= 0.6) & (df_yulei["scattering_angle"] <= 1.0)
    )
    df_aaron_cut = df_aaron.loc[mask_m].copy()
    df_yulei_cut = df_yulei.loc[mask_y].copy()

    # Store distributions after cuts
    m_after_m = df_aaron_cut["m_ditau"].to_numpy()
    theta_after_m = df_aaron_cut["scattering_angle"].to_numpy()
    m_after_y = df_yulei_cut["m_ditau"].to_numpy()
    theta_after_y = df_yulei_cut["scattering_angle"].to_numpy()

    # Plot before/after distributions
    bins_theta = np.linspace(0, np.pi, 50)
    mass_min = np.nanmin(np.concatenate([m_before_m, m_before_y]))
    mass_max = np.nanmax(np.concatenate([m_before_m, m_before_y]))
    bins_mass = np.linspace(mass_min, mass_max, 50)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Miles: scattering angle
    axes[0, 0].hist(theta_before_m, bins=bins_theta, alpha=0.5, label="Before")
    axes[0, 0].hist(theta_after_m, bins=bins_theta, alpha=0.5, label="After")
    axes[0, 0].set_xlabel("Scattering Angle [rad]")
    axes[0, 0].set_ylabel("Counts")
    axes[0, 0].set_title("Aaron: Scattering Angle")
    axes[0, 0].legend()

    # Miles: di-tau mass
    axes[1, 0].hist(m_before_m, bins=bins_mass, alpha=0.5, label="Before")
    axes[1, 0].hist(m_after_m, bins=bins_mass, alpha=0.5, label="After")
    axes[1, 0].set_xlabel("m_ditau [GeV]")
    axes[1, 0].set_ylabel("Counts")
    axes[1, 0].set_title("Aaron: di-tau Mass")
    axes[1, 0].legend()

    # Yulei: scattering angle
    axes[0, 1].hist(theta_before_y, bins=bins_theta, alpha=0.5, label="Before")
    axes[0, 1].hist(theta_after_y, bins=bins_theta, alpha=0.5, label="After")
    axes[0, 1].set_xlabel("Scattering Angle [rad]")
    axes[0, 1].set_ylabel("Counts")
    axes[0, 1].set_title("Yulei: Scattering Angle")
    axes[0, 1].legend()

    # Yulei: di-tau mass
    axes[1, 1].hist(m_before_y, bins=bins_mass, alpha=0.5, label="Before")
    axes[1, 1].hist(m_after_y, bins=bins_mass, alpha=0.5, label="After")
    axes[1, 1].set_xlabel("m_ditau [GeV]")
    axes[1, 1].set_ylabel("Counts")
    axes[1, 1].set_title("Yulei: di-tau Mass")
    axes[1, 1].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "cut_distributions.png"))
    plt.close(fig)
    print(f"Saved before/after distributions to {os.path.join(outdir, 'cut_distributions.png')}")

    # Replace dataframes with cut versions
    df_aaron = df_aaron_cut
    df_yulei = df_yulei_cut

    # Compute cosine observables
    cos_cols = [
        "leptonic_cos_r", "leptonic_cos_n", "leptonic_cos_k",
        "hadronic_cos_r", "hadronic_cos_n", "hadronic_cos_k"
    ]
    for df in (df_aaron, df_yulei):
        cos_vals = df.progress_apply(
            lambda row: pd.Series(compute_cos_observables_vector(row)),
            axis=1
        )
        for col in cos_cols:
            df[col] = cos_vals[col]

    # Set binning
    bin_edges = np.linspace(-1, 1, 51)

    # Containers for results
    b_results = {}
    c_results = {}

    # Process B observables
    for xlabel, col in tqdm(B_OBSERVABLES, desc="Processing B observables"):
        if col not in df_aaron.columns or col not in df_yulei.columns:
            print(f"Skipping {col}: not found")
            continue
        data_m = df_aaron[col].dropna().to_numpy()
        data_y = df_yulei[col].dropna().to_numpy()
        if data_m.size == 0 or data_y.size == 0:
            print(f"Skipping {col}: no data")
            continue

        # Raw (unnormalized) CSV B-observable histogram
        raw_counts_m = compute_raw_histogram(data_m, bin_edges)
        raw_out = os.path.join(outdir, f"{col}_raw_counts.png")
        raw_histogram_plot(
            bin_edges,
            raw_counts_m,
            xlabel,
            raw_out,
            label="Aaron raw counts"
        )

        norm_m, err_m = compute_normalized_histogram(data_m, bin_edges)
        norm_y, err_y = compute_normalized_histogram(data_y, bin_edges)

        current_kappa = kappa_lep if col.startswith("leptonic") else kappa_had
        outpath = os.path.join(outdir, f"{col}_overlay.png")

        # overlay: Aaron vs Recomputed Yulei
        res = overlay_plot_with_fits_B(
            bin_edges, norm_m, err_m,
            norm_y, err_y,
            xlabel, outpath,
            labels=["Aaron Code", "Recomputed Yulei"],
            kappa=current_kappa
        )
        # build B results dict
        b_results[col] = {
            "Miles":    res["Miles"],
            "Yulei":    res["Yulei"]
        }

        # overlay: Aaron vs Precomputed Yulei
        data_p = df_yulei[f"{col}_pre"].dropna().to_numpy()
        norm_p, err_p = compute_normalized_histogram(data_p, bin_edges)
        out_pre = os.path.join(outdir, f"{col}_overlay_pre.png")
        res_pre = overlay_plot_with_fits_B(
            bin_edges, norm_m, err_m,
            norm_p, err_p,
            xlabel, out_pre,
            labels=["Aaron Code", "Precomputed Yulei"],
            kappa=current_kappa
        )
        # store only the Yulei-pre dict (with asym/avg/fit)
        b_results[col]["Yulei_pre"] = res_pre["Yulei"]

    # Compute C observables for recomputed and for precomputed cosines
    for lep in LEP_KEYS:
        for had in HAD_KEYS:
            base = f"C_{lep.split('_')[-1]}{had.split('_')[-1]}"
            # recomputed
            df_aaron[base] = df_aaron[lep] * df_aaron[had]
            df_yulei[base]  = df_yulei[lep] * df_yulei[had]
            # precomputed
            df_yulei[f"{base}_pre"] = df_yulei[f"{lep}_pre"] * df_yulei[f"{had}_pre"]

    # Process C observables
    for xlabel, col in tqdm(PRODUCT_OBSERVABLES, desc="Processing C observables"):
        if col not in df_aaron.columns or col not in df_yulei.columns:
            print(f"Skipping {col}: not found")
            continue
        data_m = df_aaron[col].dropna().to_numpy()
        data_y = df_yulei[col].dropna().to_numpy()
        if data_m.size == 0 or data_y.size == 0:
            print(f"Skipping {col}: no data")
            continue
        norm_m, err_m = compute_normalized_histogram(data_m, bin_edges)
        norm_y, err_y = compute_normalized_histogram(data_y, bin_edges)
        outpath = os.path.join(outdir, f"{col}_overlay.png")

        # overlay: Aaron vs Recomputed Yulei
        res = overlay_plot_with_fits_C(
            bin_edges, norm_m, err_m,
            norm_y, err_y,
            xlabel, outpath,
            labels=["Aaron Code", "Recomputed Yulei"],
            k0=kappa0, k1=kappa1
        )
        c_results[col] = {
            "Miles":    res["Miles"],
            "Yulei":    res["Yulei"]
        }

        # overlay: Aaron vs Precomputed Yulei
        data_p = df_yulei[f"{col}_pre"].dropna().to_numpy()
        norm_p, err_p = compute_normalized_histogram(data_p, bin_edges)
        out_pre = os.path.join(outdir, f"{col}_overlay_pre.png")
        res_pre = overlay_plot_with_fits_C(
            bin_edges, norm_m, err_m,
            norm_p, err_p,
            xlabel, out_pre,
            labels=["Aaron Code", "Precomputed Yulei"],
            k0=kappa0, k1=kappa1
        )
        # store only the Yulei-pre dict (with asym/avg/fit)
        c_results[col]["Yulei_pre"] = res_pre["Yulei"]

    # Create summary PDF
    create_summary_pdf(b_results, c_results, kappa0, kappa1, B_OBSERVABLES, PRODUCT_OBSERVABLES, outdir)

if __name__ == "__main__":
    main()

