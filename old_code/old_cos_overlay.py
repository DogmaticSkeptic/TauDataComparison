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

# Enable progress_apply for pandas.
tqdm.pandas()

# =============================================================================
# Settings and File Names
# =============================================================================
MILES_FILE = "./data/miles_events.csv"
YULEI_FILE = "./data/yulei_events.csv"

# Define the six B observables.
# Each tuple: (xlabel for plot, column name)
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
kappa_lep = -0.34   # For muonic decay (adjust as needed)
kappa_had = 0.41    # For hadronic (rho) decay (adjust as needed)
# For C observables
kappa0 = kappa_lep
kappa1 = kappa_had

# =============================================================================
# Utility Functions
# =============================================================================
def unit_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def compute_cos_observables_vector(row):
    """
    Compute the six cosine observables from the tau decay kinematics.
    Returns a dict with keys:
      leptonic_cos_r, leptonic_cos_n, leptonic_cos_k,
      hadronic_cos_r, hadronic_cos_n, hadronic_cos_k.
    """
    tau_lep = vector.obj(
        pt=row["TauLeptonic_pt"], eta=row["TauLeptonic_eta"],
        phi=row["TauLeptonic_phi"], mass=row["TauLeptonic_mass"]
    )
    tau_had = vector.obj(
        pt=row["TauHadronic_pt"], eta=row["TauHadronic_eta"],
        phi=row["TauHadronic_phi"], mass=row["TauHadronic_mass"]
    )
    lepton = vector.obj(
        pt=row["Muon_pt"], eta=row["Muon_eta"],
        phi=row["Muon_phi"], mass=row["Muon_mass"]
    )
    pion_ch = vector.obj(
        pt=row["ChargedPion_pt"], eta=row["ChargedPion_eta"],
        phi=row["ChargedPion_phi"], mass=row["ChargedPion_mass"]
    )
    pion0 = vector.obj(
        pt=row["NeutralPion_pt"], eta=row["NeutralPion_eta"],
        phi=row["NeutralPion_phi"], mass=row["NeutralPion_mass"]
    )
    rho = pion_ch + pion0
    ditau = tau_lep + tau_had

    # -- Leptonic tau --
    tau_lep_cm = tau_lep.boostCM_of(ditau)
    k_hat = unit_vector(np.array([tau_lep_cm.x, tau_lep_cm.y, tau_lep_cm.z]))
    z_axis = np.array([0, 0, 1])
    r_hat = unit_vector(z_axis - np.dot(z_axis, k_hat) * k_hat)
    n_hat = np.cross(k_hat, r_hat)
    lepton_rest = lepton.boostCM_of(tau_lep)
    lepton_mom = unit_vector(np.array([lepton_rest.x, lepton_rest.y, lepton_rest.z]))
    leptonic_cos_r = np.dot(lepton_mom, r_hat)
    leptonic_cos_n = np.dot(lepton_mom, n_hat)
    leptonic_cos_k = np.dot(lepton_mom, k_hat)

    # -- Hadronic tau --
    tau_had_cm = tau_had.boostCM_of(ditau)
    k_hat2 = unit_vector(np.array([tau_had_cm.x, tau_had_cm.y, tau_had_cm.z]))
    r_hat2 = unit_vector(z_axis - np.dot(z_axis, k_hat2) * k_hat2)
    n_hat2 = np.cross(k_hat2, r_hat2)
    rho_rest = rho.boostCM_of(tau_had)
    rho_mom = unit_vector(np.array([rho_rest.x, rho_rest.y, rho_rest.z]))
    hadronic_cos_r = np.dot(rho_mom, r_hat2)
    hadronic_cos_n = np.dot(rho_mom, n_hat2)
    hadronic_cos_k = np.dot(rho_mom, k_hat2)

    return {
        "leptonic_cos_r": leptonic_cos_r,
        "leptonic_cos_n": leptonic_cos_n,
        "leptonic_cos_k": leptonic_cos_k,
        "hadronic_cos_r": hadronic_cos_r,
        "hadronic_cos_n": hadronic_cos_n,
        "hadronic_cos_k": hadronic_cos_k
    }

def compute_normalized_histogram(data, bin_edges):
    counts, _ = np.histogram(data, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    total_area = np.sum(counts * bin_widths)
    if total_area == 0:
        return np.zeros_like(counts, dtype=float), np.zeros_like(counts, dtype=float)
    norm = counts / total_area
    err = np.sqrt(counts) / total_area
    return norm, err

# =============================================================================
# Models for Curve Fitting
# =============================================================================
def model_B_k(x, B, kappa):
    return 0.5 * (1 + kappa * B * x)

def model_C(x, C, kappa0, kappa1):
    epsilon = 1e-6
    return -0.5 * (1 + kappa0 * kappa1 * C * x) * np.log(np.abs(x) + epsilon)

# =============================================================================
# Overlay Plotting Functions
# =============================================================================
def overlay_plot_with_fits_B(bin_edges, norm1, err1, norm2, err2, xlabel, output_path, labels, kappa):
    """
    For a B observable, overlay normalized histograms (Miles vs. Yulei)
    and extract:
      B_asym = 2A/kappa,
      B_avg = 3<x>/kappa,
      B_fit via nonlinear fit with f_B(x; B) = 0.5*(1+kappa*B*x).
    Uncertainties are propagated.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    # --- Dataset 1 ---
    total1 = np.sum(norm1 * widths)
    pos_mask = bin_centers > 0
    neg_mask = bin_centers < 0
    P1 = np.sum(norm1[pos_mask] * widths[pos_mask])
    N1 = np.sum(norm1[neg_mask] * widths[neg_mask])
    sigma_P1 = np.sqrt(np.sum((err1[pos_mask] * widths[pos_mask])**2))
    sigma_N1 = np.sqrt(np.sum((err1[neg_mask] * widths[neg_mask])**2))
    T1 = P1 + N1
    A1 = (P1 - N1) / T1
    spin_asym1 = 2 * A1 / kappa
    dA_dP = 2 * N1 / (T1**2)
    dA_dN = 2 * P1 / (T1**2)
    sigma_A1 = np.sqrt((dA_dP * sigma_P1)**2 + (dA_dN * sigma_N1)**2)
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

    # --- Dataset 2 ---
    total2 = np.sum(norm2 * widths)
    P2 = np.sum(norm2[pos_mask] * widths[pos_mask])
    N2 = np.sum(norm2[neg_mask] * widths[neg_mask])
    sigma_P2 = np.sqrt(np.sum((err2[pos_mask] * widths[pos_mask])**2))
    sigma_N2 = np.sqrt(np.sum((err2[neg_mask] * widths[neg_mask])**2))
    T2 = P2 + N2
    A2 = (P2 - N2) / T2
    spin_asym2 = 2 * A2 / kappa
    dA_dP2 = 2 * N2 / (T2**2)
    dA_dN2 = 2 * P2 / (T2**2)
    sigma_A2 = np.sqrt((dA_dP2 * sigma_P2)**2 + (dA_dN2 * sigma_N2)**2)
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
    plt.savefig(output_path, format='png')
    plt.close()
    print(f"Saved overlay plot (B) to {output_path}")
    return {
        "Miles": {"asym": (spin_asym1, dspin_asym1), "avg": (spin_avg1, dspin_avg1), "fit": (spin_fit1, dspin_fit1)},
        "Yulei": {"asym": (spin_asym2, dspin_asym2), "avg": (spin_avg2, dspin_avg2), "fit": (spin_fit2, dspin_fit2)}
    }

def overlay_plot_with_fits_C(bin_edges, norm1, err1, norm2, err2, xlabel, output_path, labels, kappa0, kappa1):
    """
    For a C observable, overlay normalized histograms (Miles vs. Yulei)
    and extract:
      C_asym = 4A/(kappa0*kappa1),
      C_avg = 9<x>/(kappa0*kappa1),
      C_fit via nonlinear fit using f_C(x; C) = -0.5*(1+kappa0*kappa1*C*x)*ln(|x|+ε).
    Uncertainties are propagated.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    # --- Dataset 1 ---
    total1 = np.sum(norm1 * widths)
    pos_mask = bin_centers > 0
    neg_mask = bin_centers < 0
    P1 = np.sum(norm1[pos_mask] * widths[pos_mask])
    N1 = np.sum(norm1[neg_mask] * widths[neg_mask])
    sigma_P1 = np.sqrt(np.sum((err1[pos_mask] * widths[pos_mask])**2))
    sigma_N1 = np.sqrt(np.sum((err1[neg_mask] * widths[neg_mask])**2))
    T1 = P1 + N1
    A1 = (P1 - N1) / T1
    spin_asym1 = 4 * A1 / (kappa0 * kappa1)
    dA_dP = 2 * N1 / (T1**2)
    dA_dN = 2 * P1 / (T1**2)
    sigma_A1 = np.sqrt((dA_dP * sigma_P1)**2 + (dA_dN * sigma_N1)**2)
    dspin_asym1 = 4 * sigma_A1 / (kappa0 * kappa1)
    avg1 = np.sum(bin_centers * norm1 * widths) / T1
    sigma_avg1 = np.sqrt(np.sum(((bin_centers - avg1)**2 * (err1 * widths)**2)))/T1
    spin_avg1 = 9 * avg1 / (kappa0 * kappa1)
    dspin_avg1 = 9 * sigma_avg1 / (kappa0 * kappa1)
    p0_1 = [spin_avg1]
    try:
        popt1, pcov1 = curve_fit(lambda x, C: model_C(x, C, kappa0, kappa1),
                                 bin_centers, norm1, p0=p0_1, sigma=err1, absolute_sigma=True)
        spin_fit1 = popt1[0]
        dspin_fit1 = np.sqrt(np.diag(pcov1))[0]
    except Exception as e:
        print(f"Fit failed for dataset 1 (C): {e}")
        spin_fit1 = np.nan; dspin_fit1 = np.nan

    # --- Dataset 2 ---
    total2 = np.sum(norm2 * widths)
    P2 = np.sum(norm2[pos_mask] * widths[pos_mask])
    N2 = np.sum(norm2[neg_mask] * widths[neg_mask])
    sigma_P2 = np.sqrt(np.sum((err2[pos_mask] * widths[pos_mask])**2))
    sigma_N2 = np.sqrt(np.sum((err2[neg_mask] * widths[neg_mask])**2))
    T2 = P2 + N2
    A2 = (P2 - N2) / T2
    spin_asym2 = 4 * A2 / (kappa0 * kappa1)
    dA_dP2 = 2 * N2 / (T2**2)
    dA_dN2 = 2 * P2 / (T2**2)
    sigma_A2 = np.sqrt((dA_dP2 * sigma_P2)**2 + (dA_dN2 * sigma_N2)**2)
    dspin_asym2 = 4 * sigma_A2 / (kappa0 * kappa1)
    avg2 = np.sum(bin_centers * norm2 * widths) / T2
    sigma_avg2 = np.sqrt(np.sum(((bin_centers - avg2)**2 * (err2 * widths)**2)))/T2
    spin_avg2 = 9 * avg2 / (kappa0 * kappa1)
    dspin_avg2 = 9 * sigma_avg2 / (kappa0 * kappa1)
    p0_2 = [spin_avg2]
    try:
        popt2, pcov2 = curve_fit(lambda x, C: model_C(x, C, kappa0, kappa1),
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
    plt.plot(x_fit, model_C(x_fit, spin_asym1, kappa0, kappa1), color='blue', linestyle='dashed',
             label=f"{labels[0]} C_asym = {spin_asym1:.3f} ± {dspin_asym1:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_avg1, kappa0, kappa1), color='blue', linestyle='dashdot',
             label=f"{labels[0]} C_avg = {spin_avg1:.3f} ± {dspin_avg1:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_fit1, kappa0, kappa1), color='blue', linestyle='solid',
             label=f"{labels[0]} C_fit = {spin_fit1:.3f} ± {dspin_fit1:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_asym2, kappa0, kappa1), color='red', linestyle='dashed',
             label=f"{labels[1]} C_asym = {spin_asym2:.3f} ± {dspin_asym2:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_avg2, kappa0, kappa1), color='red', linestyle='dashdot',
             label=f"{labels[1]} C_avg = {spin_avg2:.3f} ± {dspin_avg2:.3f}")
    plt.plot(x_fit, model_C(x_fit, spin_fit2, kappa0, kappa1), color='red', linestyle='solid',
             label=f"{labels[1]} C_fit = {spin_fit2:.3f} ± {dspin_fit2:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='png')
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
    """
    Constructs the 4x4 spin density matrix:
      ρ = 1/4 [I₄ + Σ_i BA[i]*(σ_i ⊗ I₂) + Σ_j BB[j]*(I₂ ⊗ σ_j)
               + Σ_{ij} C[i,j]*(σ_i ⊗ σ_j)]
    with i, j over the basis [r, n, k].
    """
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
    """
    Computes concurrence from a 4x4 density matrix using the standard formula.
    """
    sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    R = rho @ np.kron(sigma_y, sigma_y) @ np.conjugate(rho) @ np.kron(sigma_y, sigma_y)
    eigvals = np.sort(np.sqrt(np.abs(la.eigvals(R))))[::-1]
    return max(0, eigvals[0] - np.sum(eigvals[1:]))

# =============================================================================
# Bell Violation Measure with Uncertainty Propagation
# =============================================================================
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
    """
    Assemble numeric 3x3 C matrix and corresponding uncertainty matrix for a given
    dataset ('Miles' or 'Yulei') and method ('asym', 'avg', or 'fit').
    """
    C_vals = np.zeros((3, 3))
    C_unc = np.zeros((3, 3))
    idx_map = {"r": 0, "n": 1, "k": 2}
    for (xlabel, col) in PRODUCT_OBSERVABLES:
        letters = col.split("_")[-1]
        row_char = letters[0]
        col_char = letters[1]
        try:
            val, unc = c_results[col][dataset][method]
        except KeyError:
            val, unc = np.nan, np.nan
        C_vals[idx_map[row_char], idx_map[col_char]] = val
        C_unc[idx_map[row_char], idx_map[col_char]] = unc
    return C_vals, C_unc

# =============================================================================
# Main Processing and Summary
# =============================================================================
def main():
    # Read CSV files.
    try:
        df_miles = pd.read_csv(MILES_FILE).iloc[:-1].head(10000)
    except Exception as e:
        print(f"Error reading {MILES_FILE}: {e}")
        return
    try:
        df_yulei = pd.read_csv(YULEI_FILE).iloc[:-1].head(10000)
    except Exception as e:
        print(f"Error reading {YULEI_FILE}: {e}")
        return

    # (Optional) Apply event-level cuts on Yulei if needed.
    # df_yulei = df_yulei[(df_yulei["Muon_pt"] > 12) & (df_yulei["TauHadronic_pt"] >= 25)]
    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)

    # Compute cosine observables.
    cos_cols = ["leptonic_cos_r", "leptonic_cos_n", "leptonic_cos_k",
                "hadronic_cos_r", "hadronic_cos_n", "hadronic_cos_k"]
    for df in [df_miles, df_yulei]:
        cos_vals = df.progress_apply(lambda row: pd.Series(compute_cos_observables_vector(row)), axis=1)
        for col in cos_cols:
            df[col] = cos_vals[col]

    # Set fixed binning from -1 to 1.
    bin_edges = np.linspace(-1, 1, 51)

    # Dictionaries to store extracted parameters.
    b_results = {}
    c_results = {}

    # Process B observables.
    for xlabel, col in tqdm(B_OBSERVABLES, desc="Processing B observables"):
        if (col not in df_miles.columns) or (col not in df_yulei.columns):
            print(f"Skipping {col} (not found).")
            continue
        data_miles = df_miles[col].dropna().to_numpy()
        data_yulei = df_yulei[col].dropna().to_numpy()
        if (data_miles.size == 0) or (data_yulei.size == 0):
            print(f"Skipping {col} (no data).")
            continue
        norm_miles, err_miles = compute_normalized_histogram(data_miles, bin_edges)
        norm_yulei, err_yulei = compute_normalized_histogram(data_yulei, bin_edges)
        if col.startswith("leptonic"):
            current_kappa = kappa_lep
        elif col.startswith("hadronic"):
            current_kappa = kappa_had
        else:
            current_kappa = 1.0
        outpath = os.path.join(outdir, f"{col}_overlay.png")
        res = overlay_plot_with_fits_B(bin_edges, norm_miles, err_miles,
                                        norm_yulei, err_yulei, xlabel,
                                        outpath, labels=["Miles Data", "Yulei Data"],
                                        kappa=current_kappa)
        b_results[col] = res

    # Compute product (C) observables.
    for lep in LEP_KEYS:
        for had in HAD_KEYS:
            new_col = f"C_{lep.split('_')[-1]}{had.split('_')[-1]}"
            df_miles[new_col] = df_miles[lep] * df_miles[had]
            df_yulei[new_col] = df_yulei[lep] * df_yulei[had]

    # Process C observables.
    for xlabel, col in tqdm(PRODUCT_OBSERVABLES, desc="Processing C observables"):
        if (col not in df_miles.columns) or (col not in df_yulei.columns):
            print(f"Skipping {col} (not found).")
            continue
        data_miles = df_miles[col].dropna().to_numpy()
        data_yulei = df_yulei[col].dropna().to_numpy()
        if (data_miles.size == 0) or (data_yulei.size == 0):
            print(f"Skipping {col} (no data).")
            continue
        norm_miles, err_miles = compute_normalized_histogram(data_miles, bin_edges)
        norm_yulei, err_yulei = compute_normalized_histogram(data_yulei, bin_edges)
        outpath = os.path.join(outdir, f"{col}_overlay.png")
        res = overlay_plot_with_fits_C(bin_edges, norm_miles, err_miles,
                                        norm_yulei, err_yulei, xlabel,
                                        outpath, labels=["Miles Data", "Yulei Data"],
                                        kappa0=kappa0, kappa1=kappa1)
        c_results[col] = res

    # =============================================================================
    # Assemble extracted values and compute density matrices and entanglement measures.
    # BA (for leptonic) and BB (for hadronic) are taken from the B observables.
    # For C observables, assemble a 3x3 matrix in the order [r, n, k].
    # =============================================================================
    def assemble_B_vector(dataset, method):
        BA = []
        BB = []
        for (xlabel, col) in B_OBSERVABLES:
            if col.startswith("leptonic"):
                val, unc = b_results[col][dataset][method]
                BA.append(f"{val:.3f} ± {unc:.3f}")
            elif col.startswith("hadronic"):
                val, unc = b_results[col][dataset][method]
                BB.append(f"{val:.3f} ± {unc:.3f}")
        return BA, BB

    def assemble_B_numeric(dataset, method):
        BA = []
        BB = []
        for (xlabel, col) in B_OBSERVABLES:
            if col.startswith("leptonic"):
                BA.append(b_results[col][dataset][method][0])
            elif col.startswith("hadronic"):
                BB.append(b_results[col][dataset][method][0])
        return np.array(BA), np.array(BB)

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
            # Assemble C matrix values and uncertainties
            C_vals, C_unc = assemble_C_numeric_and_uncertainty(c_results, ds, m)
            # Compute density matrix, concurrence, etc.
            rho = compute_density_matrix(BA_num, BB_num, C_vals)
            conc = compute_concurrence(rho)
            # Compute best Bell violation measure
            best_pair, best_sign, bell_violation, bell_unc = compute_best_bell_violation(C_vals, C_unc)
            summary_lines.append(f"Method: {m}")
            summary_lines.append("  Leptonic (BA): " + ", ".join(BA_str))
            summary_lines.append("  Hadronic (BB): " + ", ".join(BB_str))
            summary_lines.append("  Spin Correlation (C) matrix:")
            for row in C_vals:
                summary_lines.append("    " + "  ".join(f"{elem:.3f}" for elem in row))
            summary_lines.append("  Density Matrix (rho):")
            for row in rho:
                row_str = "  ".join([f"{elem.real:+.3f}{elem.imag:+.3f}j" for elem in row])
                summary_lines.append("    " + row_str)
            summary_lines.append(f"  Concurrence: {conc:.3f}")
            summary_lines.append("  Bell Violation Measure:")
            # Map indices to basis letters.
            idx_to_basis = {0: 'r', 1: 'n', 2: 'k'}
            if best_pair is not None:
                i_letter = idx_to_basis[best_pair[0]]
                j_letter = idx_to_basis[best_pair[1]]
                summary_lines.append(f"    Best pair: ({i_letter}, {j_letter}) with sign '{best_sign}'")
                summary_lines.append(f"    Violation: {bell_violation:.3f} ± {bell_unc:.3f}")
            else:
                summary_lines.append("    Not available")
            summary_lines.append("-" * 80 + "\n")
    
    # Create a human-readable summary.
    summary_text = "\n".join(summary_lines)
    wrapped_text = "\n".join(textwrap.fill(line, width=90) for line in summary_text.split("\n"))
    
    summary_pdf_path = os.path.join(outdir, "Summary.pdf")
    ppSummary = PdfPages(summary_pdf_path)
    
    # Create a new figure to accommodate the summary text.
    figSum = plt.figure(figsize=(11, 17))
    figSum.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    figSum.text(0.01, 0.99, wrapped_text, va="top", ha="left",
                fontfamily="monospace", fontsize=8, wrap=True)
    
    ppSummary.savefig(figSum)
    plt.close(figSum)
    ppSummary.close()
    print(f"Summary PDF saved to {summary_pdf_path}")

if __name__ == "__main__":
    main()
