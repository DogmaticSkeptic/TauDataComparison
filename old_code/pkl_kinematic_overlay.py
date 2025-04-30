import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Input files
AARON_FILE  = "./data/aaron_code_data.csv"
YULEI_FILE  = "./data/yulei_code_data.pkl"

# Mapping of observables: (xlabel, csv_col, pkl_col)
OBSERVABLES = [
    # Z boson
    (r"$p_{T}^{Z}$ [GeV]",      "Z_pt",              "Z_PT"),
    (r"$\eta^{Z}$",             "Z_eta",             "Z_Eta"),
    (r"$\phi^{Z}$ [rad]",       "Z_phi",             "Z_Phi"),
    (r"$m^{Z}$ [GeV]",          "Z_mass",            "Z_Mass"),
    # hadronic tau (tau_p)
    (r"$p_{T}^{\mathrm{had\tau}}$ [GeV]",
                                "TauHadronic_pt",    "tau_p_PT"),
    (r"$\eta^{\mathrm{had\tau}}$",
                                "TauHadronic_eta",   "tau_p_Eta"),
    (r"$\phi^{\mathrm{had\tau}}$ [rad]",
                                "TauHadronic_phi",   "tau_p_Phi"),
    (r"$m^{\mathrm{had\tau}}$ [GeV]",
                                "TauHadronic_mass",  "tau_p_Mass"),
    # leptonic tau (tau_m)
    (r"$p_{T}^{\mathrm{lep\tau}}$ [GeV]",
                                "TauLeptonic_pt",    "tau_m_PT"),
    (r"$\eta^{\mathrm{lep\tau}}$",
                                "TauLeptonic_eta",   "tau_m_Eta"),
    (r"$\phi^{\mathrm{lep\tau}}$ [rad]",
                                "TauLeptonic_phi",   "tau_m_Phi"),
    (r"$m^{\mathrm{lep\tau}}$ [GeV]",
                                "TauLeptonic_mass",  "tau_m_Mass"),
    # muon
    (r"$p_{T}^{\mu}$ [GeV]",     "Muon_pt",           "mu_m_PT"),
    (r"$\eta^{\mu}$",            "Muon_eta",          "mu_m_Eta"),
    (r"$\phi^{\mu}$ [rad]",      "Muon_phi",          "mu_m_Phi"),
    (r"$m^{\mu}$ [GeV]",         "Muon_mass",         "mu_m_Mass"),
    # rho meson
    (r"$p_{T}^{\rho^{+}}$ [GeV]", "rho_p_pt",          "rho_p_PT"),
    (r"$\eta^{\rho^{+}}$",        "rho_p_eta",         "rho_p_Eta"),
    (r"$\phi^{\rho^{+}}$ [rad]",  "rho_p_phi",         "rho_p_Phi"),
    (r"$m^{\rho^{+}}$ [GeV]",     "rho_p_mass",        "rho_p_Mass"),
]

def compute_normalized_histogram(data: np.ndarray, bins: np.ndarray):
    counts, _ = np.histogram(data, bins=bins)
    widths = np.diff(bins)
    area = np.sum(counts * widths)
    if area == 0:
        return np.zeros_like(counts, float), np.zeros_like(counts, float)
    norm = counts / area
    errs = np.sqrt(counts) / area
    return norm, errs

def overlay_plot(bins: np.ndarray,
                 norm_a: np.ndarray, err_a: np.ndarray,
                 norm_y: np.ndarray, err_y: np.ndarray,
                 xlabel: str,
                 output_path: str):
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths  = np.diff(bins)
    plt.figure(figsize=(8,6))
    plt.bar(centers, norm_a, width=widths, color="C0", alpha=0.5,
            edgecolor="black", label="Aaron Code")
    plt.errorbar(centers, norm_a, yerr=err_a, fmt="none",
                 color="black", capsize=3)
    plt.bar(centers, norm_y, width=widths, color="C1", alpha=0.5,
            edgecolor="black", label="Yulei Code")
    plt.errorbar(centers, norm_y, yerr=err_y, fmt="none",
                 color="black", capsize=3)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def determine_bins(col: str, data: np.ndarray):
    """
    Define bin edges based on the (lower-cased) column name.
    """
    key = col.lower()
    if key.endswith("_pt"):
        return np.linspace(0, 100, 51)
    if key.endswith("_mass") and "z_" not in key:
        # separate mass windows
        if "hadronic" in key:
            return np.linspace(1.7, 1.85, 51)
        if "rho_p_mass" in key:
            return np.linspace(0.1, 0.25, 51)
        return np.linspace(0, 0.1, 51)
    if key == "z_mass":
        return np.linspace(50, 150, 51)
    if key == "z_eta":
        return np.linspace(-12, 12, 51)
    if key == "z_phi":
        return np.linspace(-np.pi, np.pi, 51)
    # fallback: ±5% margin around data
    lo, hi = np.min(data), np.max(data)
    margin = 0.05 * (hi - lo)
    return np.linspace(lo - margin, hi + margin, 51)

def main():
    # load both tables
    try:
        df_aaron = pd.read_csv(AARON_FILE)
    except Exception as e:
        raise RuntimeError(f"Cannot read '{AARON_FILE}': {e}")
    try:
        df_yulei = pd.read_pickle(YULEI_FILE)
    except Exception as e:
        raise RuntimeError(f"Cannot read '{YULEI_FILE}': {e}")

    # check for missing columns
    missing_in_aaron = []
    missing_in_yulei = []
    for _, csv_col, pkl_col in OBSERVABLES:
        if csv_col not in df_aaron.columns:
            missing_in_aaron.append(csv_col)
        if pkl_col not in df_yulei.columns:
            missing_in_yulei.append(pkl_col)

    if missing_in_aaron:
        print("Warning: missing in Aaron dataset:", missing_in_aaron)
    if missing_in_yulei:
        print("Warning: missing in Yulei dataset:", missing_in_yulei)

    os.makedirs("plots", exist_ok=True)

    # produce overlaid histograms for each matched observable
    for xlabel, csv_col, pkl_col in OBSERVABLES:
        if csv_col not in df_aaron.columns or pkl_col not in df_yulei.columns:
            continue

        a_vals = df_aaron[csv_col].dropna().to_numpy()
        y_vals = df_yulei[pkl_col].dropna().to_numpy()
        if a_vals.size == 0 or y_vals.size == 0:
            continue

        bins = determine_bins(csv_col, np.concatenate([a_vals, y_vals]))
        na, ea = compute_normalized_histogram(a_vals, bins)
        ny, ey = compute_normalized_histogram(y_vals, bins)

        outpath = os.path.join("plots", f"{csv_col}_comparison.png")
        overlay_plot(bins, na, ea, ny, ey, xlabel, outpath)

if __name__ == "__main__":
    main()
