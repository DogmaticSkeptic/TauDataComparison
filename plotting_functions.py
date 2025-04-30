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
from scipy.optimize import curve_fit
import numpy.linalg as la
import textwrap
import vector   # ensure Vector4D is on the namespace
from fit_functions import model_B_k, model_C


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
def overlay_plot_with_three_B(bin_edges,
                              norm1, err1, norm2, err2, norm3, err3,
                              xlabel, output_path, labels, kappa):
    """
    Same as overlay_plot_with_fits_B but simply draws three normalized histograms
    (no fits) for Aaron, recomputed Yulei, and precomputed Yulei.
    """
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    widths     = np.diff(bin_edges)

    # three‐way
    plt.figure(figsize=(8,6))
    plt.bar(bin_centers, norm1, width=widths, color='blue' , alpha=0.5,
            label=labels[0], edgecolor='black')
    plt.errorbar(bin_centers, norm1, yerr=err1, fmt='none', color='black', capsize=3)
    plt.bar(bin_centers, norm2, width=widths, color='red'  , alpha=0.5,
            label=labels[1], edgecolor='black')
    plt.errorbar(bin_centers, norm2, yerr=err2, fmt='none', color='black', capsize=3)
    plt.bar(bin_centers, norm3, width=widths, color='green', alpha=0.5,
            label=labels[2], edgecolor='black')
    plt.errorbar(bin_centers, norm3, yerr=err3, fmt='none', color='black', capsize=3)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized Counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()
    print(f"Saved 3‐way overlay to {output_path}")
    return None

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


# -----------------------------------------------------------------------------
# Raw histogram plotting (Counts only)
def raw_histogram_plot(bin_edges: np.ndarray,
                       counts: np.ndarray,
                       xlabel: str,
                       output_path: str,
                       label: str = "Counts") -> None:
    """
    Plot unnormalized histogram counts and save as a PNG.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths      = np.diff(bin_edges)

    plt.figure(figsize=(8,6))
    plt.bar(bin_centers, counts, width=widths,
            alpha=0.5, label=label, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.close()
    print(f"Saved raw histogram to {output_path}")


