import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vector
from tqdm import tqdm

# Define the CSV file names.
MILES_FILE = "./data/miles_events.csv"
YULEI_FILE = "./data/lep_rho_z_valid.csv"

# List of observables: (x-axis label, column name)
OBSERVABLES = [
    (r"$p_{T}^{\mathrm{lep\tau}}$ [GeV]", "TauLeptonic_pt"),
    (r"$\eta^{\mathrm{lep\tau}}$", "TauLeptonic_eta"),
    (r"$\phi^{\mathrm{lep\tau}}$ [rad]", "TauLeptonic_phi"),
    (r"$m^{\mathrm{lep\tau}}$ [GeV]", "TauLeptonic_mass"),
    (r"$p_{T}^{\mathrm{had\tau}}$ [GeV]", "TauHadronic_pt"),
    (r"$\eta^{\mathrm{had\tau}}$", "TauHadronic_eta"),
    (r"$\phi^{\mathrm{had\tau}}$ [rad]", "TauHadronic_phi"),
    (r"$m^{\mathrm{had\tau}}$ [GeV]", "TauHadronic_mass"),
    (r"$p_{T}^{\pi^{\pm}}$ [GeV]", "ChargedPion_pt"),
    (r"$\eta^{\pi^{\pm}}$", "ChargedPion_eta"),
    (r"$\phi^{\pi^{\pm}}$ [rad]", "ChargedPion_phi"),
    (r"$m^{\pi^{\pm}}$ [GeV]", "ChargedPion_mass"),
    (r"$p_{T}^{\pi^{0}}$ [GeV]", "NeutralPion_pt"),
    (r"$\eta^{\pi^{0}}$", "NeutralPion_eta"),
    (r"$\phi^{\pi^{0}}$ [rad]", "NeutralPion_phi"),
    (r"$m^{\pi^{0}}$ [GeV]", "NeutralPion_mass"),
    (r"$p_{T}^{\mu}$ [GeV]", "Muon_pt"),
    (r"$\eta^{\mu}$", "Muon_eta"),
    (r"$\phi^{\mu}$ [rad]", "Muon_phi"),
    (r"$m^{\mu}$ [GeV]", "Muon_mass"),
    (r"$p_{T}^{Z}$ [GeV]", "Z_pt"),
    (r"$\eta^{Z}$", "Z_eta"),
    (r"$\phi^{Z}$ [rad]", "Z_phi"),
    (r"$m^{Z}$ [GeV]", "Z_mass"),
    (r"$p_{T}^{\nu_{\mu}~(\mathrm{lep~\tau})}$ [GeV]", "nu_mu_fromLeptonicTau_pt"),
    (r"$\eta^{\nu_{\mu}~(\mathrm{lep~\tau})}$", "nu_mu_fromLeptonicTau_eta"),
    (r"$\phi^{\nu_{\mu}~(\mathrm{lep~\tau})}$ [rad]", "nu_mu_fromLeptonicTau_phi"),
    (r"$m^{\nu_{\mu}~(\mathrm{lep~\tau})}$ [GeV]", "nu_mu_fromLeptonicTau_mass"),
    (r"$p_{T}^{\nu_{\tau}~(\mathrm{lep~\tau})}$ [GeV]", "nu_tau_fromLeptonicTau_pt"),
    (r"$\eta^{\nu_{\tau}~(\mathrm{lep~\tau})}$", "nu_tau_fromLeptonicTau_eta"),
    (r"$\phi^{\nu_{\tau}~(\mathrm{lep~\tau})}$ [rad]", "nu_tau_fromLeptonicTau_phi"),
    (r"$m^{\nu_{\tau}~(\mathrm{lep~\tau})}$ [GeV]", "nu_tau_fromLeptonicTau_mass"),
    (r"$p_{T}^{\nu_{\tau}~(\mathrm{had~\tau})}$ [GeV]", "nu_tau_fromHadronicTau_pt"),
    (r"$\eta^{\nu_{\tau}~(\mathrm{had~\tau})}$", "nu_tau_fromHadronicTau_eta"),
    (r"$\phi^{\nu_{\tau}~(\mathrm{had~\tau})}$ [rad]", "nu_tau_fromHadronicTau_phi"),
    (r"$m^{\nu_{\tau}~(\mathrm{had~\tau})}$ [GeV]", "nu_tau_fromHadronicTau_mass"),
]

def compute_normalized_histogram(data, bin_edges):
    counts, _ = np.histogram(data, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    total_area = np.sum(counts * bin_widths)
    if total_area == 0:
        norm_counts = np.zeros_like(counts, dtype=float)
        errors = np.zeros_like(counts, dtype=float)
    else:
        norm_counts = counts / total_area
        errors = np.sqrt(counts) / total_area
    return norm_counts, errors

def overlay_plot(bin_edges, norm1, err1, norm2, err2, xlabel, output_path, labels):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, norm1, width=widths, color='blue', alpha=0.5,
            label=labels[0], edgecolor='black')
    plt.errorbar(bin_centers, norm1, yerr=err1, fmt='none',
                 color='black', capsize=3)
    
    plt.bar(bin_centers, norm2, width=widths, color='red', alpha=0.5,
            label=labels[1], edgecolor='black')
    plt.errorbar(bin_centers, norm2, yerr=err2, fmt='none',
                 color='black', capsize=3)
    
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Counts")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved overlay plot: {output_path}")

def main():
    try:
        df_miles = pd.read_csv(MILES_FILE).head(100000)
    except Exception as e:
        print(f"Error reading {MILES_FILE}: {e}")
        return
    try:
        df_yulei = pd.read_csv(YULEI_FILE).head(100000)
    except Exception as e:
        print(f"Error reading {YULEI_FILE}: {e}")
        return

#    df_yulei = df_yulei[df_yulei["Muon_pt"] > 12]
#    df_yulei = df_yulei[df_yulei["TauHadronic_pt"] >= 25]
    # --- begin override: reconstruct Z from two taus via pt/eta/phi/mass ---
    Z_pt_list, Z_eta_list, Z_phi_list, Z_mass_list = [], [], [], []
    for _, row in df_yulei.iterrows():
        lep_vec = vector.obj(
            pt=row["TauLeptonic_pt"],
            eta=row["TauLeptonic_eta"],
            phi=row["TauLeptonic_phi"],
            mass=row["TauLeptonic_mass"]
        )
        had_vec = vector.obj(
            pt=row["TauHadronic_pt"],
            eta=row["TauHadronic_eta"],
            phi=row["TauHadronic_phi"],
            mass=row["TauHadronic_mass"]
        )
        z_vec = lep_vec + had_vec
        Z_pt_list.append(z_vec.pt)
        Z_eta_list.append(z_vec.eta)
        Z_phi_list.append(z_vec.phi)
        Z_mass_list.append(z_vec.mass)

    df_yulei["Z_pt"]   = Z_pt_list
    df_yulei["Z_eta"]  = Z_eta_list
    df_yulei["Z_phi"]  = Z_phi_list
    df_yulei["Z_mass"] = Z_mass_list
    # --- end override --------------------------------------------------------

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    for xlabel, col in tqdm(OBSERVABLES, desc="Processing observables"):
        if col not in df_miles.columns:
            print(f"Column {col} not found in {MILES_FILE}; skipping overlay.")
            continue
        if col not in df_yulei.columns:
            print(f"Column {col} not found in {YULEI_FILE}; skipping overlay.")
            continue

        data_miles = df_miles[col].dropna().to_numpy()
        data_yulei = df_yulei[col].dropna().to_numpy()
        if data_miles.size == 0 or data_yulei.size == 0:
            print(f"No data for {col}; skipping.")
            continue

        # Binning
        if col.endswith("_pt"):
            bin_edges = np.linspace(0, 100, 51)
        elif col in ["TauLeptonic_mass", "TauHadronic_mass"]:
            bin_edges = np.linspace(1.7, 1.85, 51)
        elif col in ["ChargedPion_mass", "NeutralPion_mass"]:
            bin_edges = np.linspace(0.1, 0.25, 51)
        elif col == "Muon_mass":
            bin_edges = np.linspace(0.05, 0.15, 51)
        elif col.endswith("nu_mu_fromLeptonicTau_mass") or \
             col.endswith("nu_tau_fromLeptonicTau_mass") or \
             col.endswith("nu_tau_fromHadronicTau_mass"):
            bin_edges = np.linspace(0, 0.1, 51)
        elif col == "Z_mass":
            bin_edges = np.linspace(50, 150, 51)
        elif col == "Z_eta":
            bin_edges = np.linspace(-12, 12, 51)
        elif col == "Z_phi":
            bin_edges = np.linspace(-np.pi, np.pi, 51)
        else:
            global_min = min(np.min(data_miles), np.min(data_yulei))
            global_max = max(np.max(data_miles), np.max(data_yulei))
            margin = 0.05 * (global_max - global_min)
            bin_edges = np.linspace(global_min - margin, global_max + margin, 51)

        norm_miles, err_miles = compute_normalized_histogram(data_miles, bin_edges)
        norm_yulei, err_yulei = compute_normalized_histogram(data_yulei, bin_edges)
        output_path = os.path.join(output_dir, f"{col}_overlay.png")
        overlay_plot(bin_edges, norm_miles, err_miles, norm_yulei, err_yulei,
                     xlabel, output_path, labels=["Miles Data", "Yulei Data"])

if __name__ == "__main__":
    main()
