import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import vector
from tqdm import tqdm
from scipy.optimize import curve_fit
import numpy.linalg as la
import textwrap
from cos_utils import compute_cos_observables_vector, compute_normalized_histogram

# Enable progress_apply for pandas.
tqdm.pandas()

# =============================================================================
# Settings and File Names
# =============================================================================
MILES_FILE = "./data/miles_events.csv"
YULEI_FILE = "./data/yulei_code_data.pkl"

# Define the six B observables.
B_OBSERVABLES = [
    (r"$\cos\theta_{r}^{\mathrm{lep}}$", "leptonic_cos_r"),
    (r"$\cos\theta_{n}^{\mathrm{lep}}$", "leptonic_cos_n"),
    (r"$\cos\theta_{k}^{\mathrm{lep}}$", "leptonic_cos_k"),
    (r"$\cos\theta_{r}^{\mathrm{had}}$", "hadronic_cos_r"),
    (r"$\cos\theta_{n}^{\mathrm{had}}$", "hadronic_cos_n"),
    (r"$\cos\theta_{k}^{\mathrm{had}}$", "hadronic_cos_k")
]

# For product (C) observables: nine combinations.
LEP_KEYS = ["leptonic_cos_r", "leptonic_cos_n", "leptonic_cos_k"]
HAD_KEYS = ["hadronic_cos_r", "hadronic_cos_n", "hadronic_cos_k"]
PRODUCT_OBSERVABLES = []
for lep in LEP_KEYS:
    for had in HAD_KEYS:
        colname = f"C_{lep.split('_')[-1]}{had.split('_')[-1]}"
        label = r"$C_{" + f"{lep.split('_')[-1]}{had.split('_')[-1]}" + "}$"
        PRODUCT_OBSERVABLES.append((label, colname))

# -----------------------------------------------------------------------------
# Analyzing Powers (kappa values)
# -----------------------------------------------------------------------------
kappa_lep = -0.34   # For muonic decay
kappa_had = 0.41    # For hadronic (rho) decay
kappa0 = kappa_lep
kappa1 = kappa_had

# =============================================================================
# Utility Functions
# =============================================================================
def compute_ditau_kinematics(row):
    """
    Compute the di-tau invariant mass and scattering angle.
    Scattering angle is defined as the polar angle of the hadronic tau
    momentum in the di-tau CM frame relative to the beam (z) axis.
    Returns a Series with 'm_ditau' (GeV) and 'scattering_angle' (rad).
    """
    tau_lep = vector.obj(
        pt=row["TauLeptonic_pt"], eta=row["TauLeptonic_eta"],
        phi=row["TauLeptonic_phi"], mass=row["TauLeptonic_mass"]
    )
    tau_had = vector.obj(
        pt=row["TauHadronic_pt"], eta=row["TauHadronic_eta"],
        phi=row["TauHadronic_phi"], mass=row["TauHadronic_mass"]
    )
    ditau = tau_lep + tau_had
    m_ditau = ditau.mass

    tau_had_cm = tau_had.boostCM_of(ditau)
    p_vec = np.array([tau_had_cm.x, tau_had_cm.y, tau_had_cm.z])
    p_mag = np.linalg.norm(p_vec)
    if p_mag == 0:
        theta = np.nan
    else:
        cos_theta = tau_had_cm.z / p_mag
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = 2 * np.arccos(np.abs(cos_theta)) / np.pi

    return pd.Series({"m_ditau": m_ditau, "scattering_angle": theta})

# =============================================================================
# Models for Curve Fitting
# =============================================================================
def model_B_k(x, B, kappa):
    return 0.5 * (1 + kappa * B * x)

def model_C(x, C, k0, k1):
    eps = 1e-6
    return -0.5 * (1 + k0 * k1 * C * x) * np.log(np.abs(x) + eps)

# =============================================================================
# Overlay Plotting Functions
# =============================================================================
def overlay_plot_with_fits_B(bin_edges, norm1, err1, norm2, err2,
                             xlabel, output_path, labels, kappa):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    # Dataset 1
    pos_mask = bin_centers > 0
    neg_mask = bin_centers < 0
    P1 = np.sum(norm1[pos_mask] * widths[pos_mask])
    N1 = np.sum(norm1[neg_mask] * widths[neg_mask])
    sigma_P1 = np.sqrt(np.sum((err1[pos_mask] * widths[pos_mask])**2))
    sigma_N1 = np.sqrt(np.sum((err1[neg_mask] * widths[neg_mask])**2))
    T1 = P1 + N1
    A1 = (P1 - N1) / T1
    spin_asym1 = 2 * A1 / kappa
    sigma_A1 = np.sqrt((2 * N1 / T1**2 * sigma_P1)**2 + (2 * P1 / T1**2 * sigma_N1)**2)
    dspin_asym1 = 2 * sigma_A1 / kappa

    avg1 = np.sum(bin_centers * norm1 * widths) / T1
    sigma_avg1 = np.sqrt(np.sum(((bin_centers - avg1)**2 * (err1 * widths)**2))) / T1
    spin_avg1 = 3 * avg1 / kappa
    dspin_avg1 = 3 * sigma_avg1 / kappa

    p0_1 = [spin_avg1]
    try:
        popt1, pcov1 = curve_fit(lambda x, B: model_B_k(x, B, kappa),
                                 bin_centers, norm1, p0=p0_1, sigma=err1,
                                 absolute_sigma=True)
        spin_fit1 = popt1[0]
        dspin_fit1 = np.sqrt(np.diag(pcov1))[0]
    except Exception as e:
        print(f"Fit failed for dataset 1 (B): {e}")
        spin_fit1 = np.nan; dspin_fit1 = np.nan

    # Dataset 2
    P2 = np.sum(norm2[pos_mask] * widths[pos_mask])
    N2 = np.sum(norm2[neg_mask] * widths[neg_mask])
    sigma_P2 = np.sqrt(np.sum((err2[pos_mask] * widths[pos_mask])**2))
    sigma_N2 = np.sqrt(np.sum((err2[neg_mask] * widths[neg_mask])**2))
    T2 = P2 + N2
    A2 = (P2 - N2) / T2
    spin_asym2 = 2 * A2 / kappa
    sigma_A2 = np.sqrt((2 * N2 / T2**2 * sigma_P2)**2 + (2 * P2 / T2**2 * sigma_N2)**2)
    dspin_asym2 = 2 * sigma_A2 / kappa

    avg2 = np.sum(bin_centers * norm2 * widths) / T2
    sigma_avg2 = np.sqrt(np.sum(((bin_centers - avg2)**2 * (err2 * widths)**2))) / T2
    spin_avg2 = 3 * avg2 / kappa
    dspin_avg2 = 3 * sigma_avg2 / kappa

    p0_2 = [spin_avg2]
    try:
        popt2, pcov2 = curve_fit(lambda x, B: model_B_k(x, B, kappa),
                                 bin_centers, norm2, p0=p0_2, sigma=err2,
                                 absolute_sigma=True)
        spin_fit2 = popt2[0]
        dspin_fit2 = np.sqrt(np.diag(pcov2))[0]
    except Exception as e:
        print(f"Fit failed for dataset 2 (B): {e}")
        spin_fit2 = np.nan; dspin_fit2 = np.nan

    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, norm1, width=widths, color='blue', alpha=0.5,
            label=labels[0], edgecolor='black')
    plt.errorbar(bin_centers, norm1, yerr=err1, fmt='none', color='black', capsize=3)
    plt.bar(bin_centers, norm2, width=widths, color='red', alpha=0.5,
            label=labels[1], edgecolor='black')
    plt.errorbar(bin_centers, norm2, yerr=err2, fmt='none', color='black', capsize=3)
    plt.plot(x_fit, model_B_k(x_fit, spin_asym1, kappa), color='blue', linestyle='dashed',
             label=f"{labels[0]} B_asym = {spin_asym1:.3f} ± {dspin_asym1:.3f}")
    plt.plot(x_fit, model_B_k(x_fit, spin_avg1, kappa), color='blue', linestyle='dashdot',
             label=f"{labels[0]} B_avg = {spin_avg1:.3f} ± {dspin_avg1:.3f}")
    plt.plot(x_fit, model_B_k(x_fit, spin_fit1, kappa), color='blue', linestyle='solid',
             label=f"{labels[0]} B_fit = {spin_fit1:.3f} ± {dspin_fit1:.3f}")
    plt.plot(x_fit, model_B_k(x_fit, spin_asym2, kappa), color='red', linestyle='dashed',
             label=f"{labels[1]} B_asym = {spin_asym2:.3f} ± {dspin_asym2:.3f}")
    plt.plot(x_fit, model_B_k(x_fit, spin_avg2, kappa), color='red', linestyle='dashdot',
             label=f"{labels[1]} B_avg = {spin_avg2:.3f} ± {dspin_avg2:.3f}")
    plt.plot(x_fit, model_B_k(x_fit, spin_fit2, kappa), color='red', linestyle='solid',
             label=f"{labels[1]} B_fit = {spin_fit2:.3f} ± {dspin_fit2:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()
    print(f"Saved overlay plot (B) to {output_path}")
    return {
        "Miles": {"asym": (spin_asym1, dspin_asym1), "avg": (spin_avg1, dspin_avg1), "fit": (spin_fit1, dspin_fit1)},
        "Yulei": {"asym": (spin_asym2, dspin_asym2), "avg": (spin_avg2, dspin_avg2), "fit": (spin_fit2, dspin_fit2)}
    }

def overlay_plot_with_fits_C(bin_edges, norm1, err1, norm2, err2,
                             xlabel, output_path, labels, k0, k1):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    # Dataset 1
    pos_mask = bin_centers > 0
    neg_mask = bin_centers < 0
    P1 = np.sum(norm1[pos_mask] * widths[pos_mask])
    N1 = np.sum(norm1[neg_mask] * widths[neg_mask])
    sigma_P1 = np.sqrt(np.sum((err1[pos_mask] * widths[pos_mask])**2))
    sigma_N1 = np.sqrt(np.sum((err1[neg_mask] * widths[neg_mask])**2))
    T1 = P1 + N1
    A1 = (P1 - N1) / T1
    spin_asym1 = 4 * A1 / (k0 * k1)
    sigma_A1 = np.sqrt((2 * N1 / T1**2 * sigma_P1)**2 + (2 * P1 / T1**2 * sigma_N1)**2)
    dspin_asym1 = 4 * sigma_A1 / (k0 * k1)

    avg1 = np.sum(bin_centers * norm1 * widths) / T1
    sigma_avg1 = np.sqrt(np.sum(((bin_centers - avg1)**2 * (err1 * widths)**2))) / T1
    spin_avg1 = 9 * avg1 / (k0 * k1)
    dspin_avg1 = 9 * sigma_avg1 / (k0 * k1)

    p0_1 = [spin_avg1]
    try:
        popt1, pcov1 = curve_fit(lambda x, C: model_C(x, C, k0, k1),
                                 bin_centers, norm1, p0=p0_1, sigma=err1, absolute_sigma=True)
        spin_fit1 = popt1[0]
        dspin_fit1 = np.sqrt(np.diag(pcov1))[0]
    except Exception as e:
        print(f"Fit failed for dataset 1 (C): {e}")
        spin_fit1 = np.nan; dspin_fit1 = np.nan

    # Dataset 2
    P2 = np.sum(norm2[pos_mask] * widths[pos_mask])
    N2 = np.sum(norm2[neg_mask] * widths[neg_mask])
    sigma_P2 = np.sqrt(np.sum((err2[pos_mask] * widths[pos_mask])**2))
    sigma_N2 = np.sqrt(np.sum((err2[neg_mask] * widths[neg_mask])**2))
    T2 = P2 + N2
    A2 = (P2 - N2) / T2
    spin_asym2 = 4 * A2 / (k0 * k1)
    sigma_A2 = np.sqrt((2 * N2 / T2**2 * sigma_P2)**2 + (2 * P2 / T2**2 * sigma_N2)**2)
    dspin_asym2 = 4 * sigma_A2 / (k0 * k1)

    avg2 = np.sum(bin_centers * norm2 * widths) / T2
    sigma_avg2 = np.sqrt(np.sum(((bin_centers - avg2)**2 * (err2 * widths)**2))) / T2
    spin_avg2 = 9 * avg2 / (k0 * k1)
    dspin_avg2 = 9 * sigma_avg2 / (k0 * k1)

    p0_2 = [spin_avg2]
    try:
        popt2, pcov2 = curve_fit(lambda x, C: model_C(x, C, k0, k1),
                                 bin_centers, norm2, p0=p0_2, sigma=err2, absolute_sigma=True)
        spin_fit2 = popt2[0]
        dspin_fit2 = np.sqrt(np.diag(pcov2))[0]
    except Exception as e:
        print(f"Fit failed for dataset 2 (C): {e}")
        spin_fit2 = np.nan; dspin_fit2 = np.nan

    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, norm1, width=widths, color='blue', alpha=0.5,
            label=labels[0], edgecolor='black')
    plt.errorbar(bin_centers, norm1, yerr=err1, fmt='none', color='black', capsize=3)
    plt.bar(bin_centers, norm2, width=widths, color='red', alpha=0.5,
            label=labels[1], edgecolor='black')
    plt.errorbar(bin_centers, norm2, yerr=err2, fmt='none', color='black', capsize=3)
    plt.plot(x_fit, model_C(x_fit, spin_asym1, k0, k1), color='blue', linestyle='dashed',
             label=f"{labels[0]} C_asym = {spin_asym1:.3f} ± {dspin_asym1:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_avg1, k0, k1), color='blue', linestyle='dashdot',
             label=f"{labels[0]} C_avg = {spin_avg1:.3f} ± {dspin_avg1:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_fit1, k0, k1), color='blue', linestyle='solid',
             label=f"{labels[0]} C_fit = {spin_fit1:.3f} ± {dspin_fit1:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_asym2, k0, k1), color='red', linestyle='dashed',
             label=f"{labels[1]} C_asym = {spin_asym2:.3f} ± {dspin_asym2:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_avg2, k0, k1), color='red', linestyle='dashdot',
             label=f"{labels[1]} C_avg = {spin_avg2:.3f} ± {dspin_avg2:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_fit2, k0, k1), color='red', linestyle='solid',
             label=f"{labels[1]} C_fit = {spin_fit2:.3f} ± {dspin_fit2:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()
    print(f"Saved overlay plot (C) to {output_path}")
    return {
        "Miles": {"asym": (spin_asym1, dspin_asym1), "avg": (spin_avg1, dspin_avg1), "fit": (spin_fit1, dspin_fit1)},
        "Yulei": {"asym": (spin_asym2, dspin_asym2), "avg": (spin_avg2, dspin_avg2), "fit": (spin_fit2, dspin_fit2)}
    }

# =============================================================================
# Density Matrix and Entanglement Functions
# =============================================================================
def compute_density_matrix(BA, BB, C):
    sigma_x = np.array([[0, 1],[1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0],[0, -1]], dtype=complex)
    paulis = {"r": sigma_x, "n": sigma_y, "k": sigma_z}
    I2 = np.eye(2, dtype=complex)
    I4 = np.eye(4, dtype=complex)
    rho = I4.copy()
    for i, key in enumerate(["r", "n", "k"]):
        rho += BA[i] * np.kron(paulis[key], I2)
        rho += BB[i] * np.kron(I2, paulis[key])
        for j, key2 in enumerate(["r", "n", "k"]):
            rho += C[i, j] * np.kron(paulis[key], paulis[key2])
    return rho / 4.0

def compute_concurrence(rho):
    sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    R = rho @ np.kron(sigma_y, sigma_y) @ np.conjugate(rho) @ np.kron(sigma_y, sigma_y)
    eigvals = np.sort(np.sqrt(np.abs(la.eigvals(R))))[::-1]
    return max(0, eigvals[0] - np.sum(eigvals[1:]))

def compute_best_bell_violation(C_vals, C_unc):
    best_violation = -np.inf
    best_pair = None
    best_sign = None
    best_unc = None
    for i in range(3):
        for j in range(i+1, 3):
            for sign in [1, -1]:
                value = abs(C_vals[i, i] + sign * C_vals[j, j]) - np.sqrt(2)
                unc = np.sqrt(C_unc[i, i]**2 + C_unc[j, j]**2)
                if value > best_violation:
                    best_violation = value
                    best_pair = (i, j)
                    best_sign = '+' if sign == 1 else '-'
                    best_unc = unc
    return best_pair, best_sign, best_violation, best_unc

def assemble_C_numeric_and_uncertainty(c_results, dataset, method):
    C_vals = np.zeros((3, 3))
    C_unc = np.zeros((3, 3))
    idx_map = {"r": 0, "n": 1, "k": 2}
    for _, col in PRODUCT_OBSERVABLES:
        letters = col.split("_")[-1]
        i = idx_map[letters[0]]
        j = idx_map[letters[1]]
        try:
            val, unc = c_results[col][dataset][method]
        except KeyError:
            val, unc = np.nan, np.nan
        C_vals[i, j] = val
        C_unc[i, j] = unc
    return C_vals, C_unc

# =============================================================================
# Main Processing and Summary
# =============================================================================
def main():
    # Read Aaron (CSV) and Yulei (PKL) datasets
    try:
        df_miles = pd.read_csv(MILES_FILE).iloc[:-1].head(100000)
        df_yulei = pd.read_pickle(YULEI_FILE).head(100000)
        # — rename to unified column scheme (as in yulei_only_cos_overlay.py)
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
        df_yulei.rename(columns=rename_map, inplace=True)
    except Exception as e:
        print(f"Error reading input files: {e}")
        return

    # Prepare output directory
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # Compute di-tau kinematics
    kin_m = df_miles.progress_apply(compute_ditau_kinematics, axis=1)
    df_miles[["m_ditau", "scattering_angle"]] = kin_m
    kin_y = df_yulei.progress_apply(compute_ditau_kinematics, axis=1)
    df_yulei[["m_ditau", "scattering_angle"]] = kin_y

    # Store distributions before cuts
    m_before_m = df_miles["m_ditau"].to_numpy()
    theta_before_m = df_miles["scattering_angle"].to_numpy()
    m_before_y = df_yulei["m_ditau"].to_numpy()
    theta_before_y = df_yulei["scattering_angle"].to_numpy()

    # Apply cuts
    min_tau_had_pt = 25.0  # GeV
    min_muon_pt = 12.0     # GeV

    mask_m = (
        (df_miles["m_ditau"] >= 80) & (df_miles["m_ditau"] <= 100) &
        (df_miles["scattering_angle"] >= 0.6) & (df_miles["scattering_angle"] <= 1.0) &
        (df_miles["TauHadronic_pt"] >= min_tau_had_pt) & (df_miles["Muon_pt"] >= min_muon_pt)
    )
    mask_y = (
        (df_yulei["m_ditau"] >= 80) & (df_yulei["m_ditau"] <= 100) &
        (df_yulei["scattering_angle"] >= 0.6) & (df_yulei["scattering_angle"] <= 1.0) &
        (df_yulei["TauHadronic_pt"] >= min_tau_had_pt) & (df_yulei["Muon_pt"] >= min_muon_pt)
    )
    df_miles_cut = df_miles.loc[mask_m].copy()
    df_yulei_cut = df_yulei.loc[mask_y].copy()

    # Store distributions after cuts
    m_after_m = df_miles_cut["m_ditau"].to_numpy()
    theta_after_m = df_miles_cut["scattering_angle"].to_numpy()
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
    axes[0, 0].set_title("Miles: Scattering Angle")
    axes[0, 0].legend()

    # Miles: di-tau mass
    axes[1, 0].hist(m_before_m, bins=bins_mass, alpha=0.5, label="Before")
    axes[1, 0].hist(m_after_m, bins=bins_mass, alpha=0.5, label="After")
    axes[1, 0].set_xlabel("m_ditau [GeV]")
    axes[1, 0].set_ylabel("Counts")
    axes[1, 0].set_title("Miles: di-tau Mass")
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
    df_miles = df_miles_cut
    df_yulei = df_yulei_cut

    # Compute cosine observables
    cos_cols = [
        "leptonic_cos_r", "leptonic_cos_n", "leptonic_cos_k",
        "hadronic_cos_r", "hadronic_cos_n", "hadronic_cos_k"
    ]
    for df in (df_miles, df_yulei):
        cos_vals = df.progress_apply(lambda row: pd.Series(compute_cos_observables_vector(row)), axis=1)
        for col in cos_cols:
            df[col] = cos_vals[col]

    # Set binning
    bin_edges = np.linspace(-1, 1, 51)

    # Containers for results
    b_results = {}
    c_results = {}

    # Process B observables
    for xlabel, col in tqdm(B_OBSERVABLES, desc="Processing B observables"):
        if col not in df_miles.columns or col not in df_yulei.columns:
            print(f"Skipping {col}: not found")
            continue
        data_m = df_miles[col].dropna().to_numpy()
        data_y = df_yulei[col].dropna().to_numpy()
        if data_m.size == 0 or data_y.size == 0:
            print(f"Skipping {col}: no data")
            continue
        norm_m, err_m = compute_normalized_histogram(data_m, bin_edges)
        norm_y, err_y = compute_normalized_histogram(data_y, bin_edges)
        current_kappa = kappa_lep if col.startswith("leptonic") else kappa_had
        outpath = os.path.join(outdir, f"{col}_overlay.png")
        res = overlay_plot_with_fits_B(bin_edges, norm_m, err_m, norm_y, err_y,
                                     xlabel, outpath, labels=["Miles Data", "Yulei Data"],
                                     kappa=current_kappa)
        b_results[col] = res

    # Compute C observables
    for lep in LEP_KEYS:
        for had in HAD_KEYS:
            new_col = f"C_{lep.split('_')[-1]}{had.split('_')[-1]}"
            df_miles[new_col] = df_miles[lep] * df_miles[had]
            df_yulei[new_col] = df_yulei[lep] * df_yulei[had]

    # Process C observables
    for xlabel, col in tqdm(PRODUCT_OBSERVABLES, desc="Processing C observables"):
        if col not in df_miles.columns or col not in df_yulei.columns:
            print(f"Skipping {col}: not found")
            continue
        data_m = df_miles[col].dropna().to_numpy()
        data_y = df_yulei[col].dropna().to_numpy()
        if data_m.size == 0 or data_y.size == 0:
            print(f"Skipping {col}: no data")
            continue
        norm_m, err_m = compute_normalized_histogram(data_m, bin_edges)
        norm_y, err_y = compute_normalized_histogram(data_y, bin_edges)
        outpath = os.path.join(outdir, f"{col}_overlay.png")
        res = overlay_plot_with_fits_C(bin_edges, norm_m, err_m, norm_y, err_y,
                                     xlabel, outpath, labels=["Miles Data", "Yulei Data"],
                                     k0=kappa0, k1=kappa1)
        c_results[col] = res

    # =============================================================================
    # Assemble and summarize
    # =============================================================================
    def assemble_B_vector(dataset, method):
        BA, BB = [], []
        for _, col in B_OBSERVABLES:
            val, unc = b_results[col][dataset][method]
            if col.startswith("leptonic"):
                BA.append(f"{val:.3f} ± {unc:.3f}")
            else:
                BB.append(f"{val:.3f} ± {unc:.3f}")
        return BA, BB

    def assemble_B_numeric(dataset, method):
        BA_vals, BB_vals = [], []
        for _, col in B_OBSERVABLES:
            val = b_results[col][dataset][method][0]
            if col.startswith("leptonic"):
                BA_vals.append(val)
            else:
                BB_vals.append(val)
        return np.array(BA_vals), np.array(BB_vals)

    methods = ["asym", "avg", "fit"]
    datasets = ["Miles", "Yulei"]
    summary_lines = []
    summary_lines.append("Spin Observables Summary\n")
    summary_lines.append("=" * 80 + "\n")
    for ds in datasets:
        summary_lines.append(f"Dataset: {ds}\n" + "-" * 80)
        for m in methods:
            BA_str, BB_str = assemble_B_vector(ds, m)
            BA_num, BB_num = assemble_B_numeric(ds, m)
            C_vals, C_unc = assemble_C_numeric_and_uncertainty(c_results, ds, m)
            rho = compute_density_matrix(BA_num, BB_num, C_vals)
            conc = compute_concurrence(rho)
            pair, sign, bell_val, bell_unc = compute_best_bell_violation(C_vals, C_unc)
            summary_lines.append(f"Method: {m}")
            summary_lines.append("  Leptonic (BA): " + ", ".join(BA_str))
            summary_lines.append("  Hadronic (BB): " + ", ".join(BB_str))
            summary_lines.append("  Spin Correlation (C) matrix:")
            for row in C_vals:
                summary_lines.append("    " + "  ".join(f"{x:.3f}" for x in row))
            summary_lines.append("  Density Matrix (ρ):")
            for row in rho:
                summary_lines.append("    " + "  ".join(f"{elem.real:+.3f}{elem.imag:+.3f}j" for elem in row))
            summary_lines.append(f"  Concurrence: {conc:.3f}")
            summary_lines.append("  Bell Violation Measure:")
            if pair is not None:
                i0, j0 = pair
                basis = {0: 'r', 1: 'n', 2: 'k'}
                summary_lines.append(f"    Best pair: ({basis[i0]}, {basis[j0]}) sign '{sign}'")
                summary_lines.append(f"    Violation: {bell_val:.3f} ± {bell_unc:.3f}")
            else:
                summary_lines.append("    Not available")
            summary_lines.append("-" * 80 + "\n")

    summary_text = "\n".join(summary_lines)
    wrapped_text = "\n".join(textwrap.fill(line, width=90) for line in summary_text.split("\n"))
    summary_pdf = os.path.join(outdir, "Summary.pdf")
    pp = PdfPages(summary_pdf)
    fig = plt.figure(figsize=(11, 17))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.text(0.01, 0.99, wrapped_text, va="top", ha="left",
             fontfamily="monospace", fontsize=8, wrap=True)
    pp.savefig(fig)
    plt.close(fig)
    pp.close()
    print(f"Summary PDF saved to {summary_pdf}")

if __name__ == "__main__":
    main()

